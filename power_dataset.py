# HAT: Hardware-Aware Transformers for Efficient Natural Language Processing
# Hanrui Wang, Zhanghao Wu, Zhijian Liu, Han Cai, Ligeng Zhu, Chuang Gan and Song Han
# The 58th Annual Meeting of the Association for Computational Linguistics (ACL), 2020.
# Paper: https://arxiv.org/abs/2005.14187
# Project page: https://hanruiwang.me/project_pages/hat/

import torch
import time
import pdb

import numpy as np

from fairseq import checkpoint_utils, distributed_utils, options, tasks, utils
from tqdm import tqdm

import threading
import pynvml


def main(args):
    utils.import_user_module(args)

    assert args.max_tokens is not None or args.max_sentences is not None, \
        'Must specify batch size either with --max-tokens or --max-sentences'

    # Initialize CUDA and distributed training
    if torch.cuda.is_available() and not args.cpu:
        torch.cuda.set_device(args.device_id)
    torch.manual_seed(args.seed)

    # Print args
    print(args)

    # Setup task
    task = tasks.setup_task(args)

    # Build model
    model = task.build_model(args)
    print(model)

    # specify the length of the dummy input for profile
    # for iwslt, the average length is 23, for wmt, that is 30
    dummy_sentence_length_dict = {'iwslt': 23, 'wmt': 30}
    if 'iwslt' in args.arch:
        dummy_sentence_length = dummy_sentence_length_dict['iwslt']
    elif 'wmt' in args.arch:
        dummy_sentence_length = dummy_sentence_length_dict['wmt']
    else:
        raise NotImplementedError

    dummy_src_tokens = [2] + [7] * (dummy_sentence_length - 1)
    dummy_prev = [7] * (dummy_sentence_length - 1) + [2]

    # for power predictor: power dataset generation
    with open(args.lat_dataset_path, 'w') as fid:
        src_tokens_test = torch.tensor([dummy_src_tokens], dtype=torch.long)
        src_lengths_test = torch.tensor([dummy_sentence_length])
        prev_output_tokens_test_with_beam = torch.tensor([dummy_prev] * args.beam, dtype=torch.long)
        if args.latcpu:
            model.cpu()
            print('Measuring model power on CPU for dataset generation...')
        elif args.latgpu:
            model.cuda()
            src_tokens_test = src_tokens_test.cuda()
            src_lengths_test = src_lengths_test.cuda()
            prev_output_tokens_test_with_beam = prev_output_tokens_test_with_beam.cuda()
            src_tokens_test.get_device()
            print('Measuring model power on GPU for dataset generation...')
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

        feature_info = utils.get_feature_info()
        fid.write(','.join(feature_info) + ',')
        power_info = ['power_mean_encoder', 'power_mean_decoder', 'power_std_encoder', 'power_std_decoder']
        fid.write(','.join(power_info) + '\n')

        for i in range(args.lat_dataset_size):
            print(i)
            config_sam = utils.sample_configs(utils.get_all_choices(args), reset_rand_seed=False, super_decoder_num_layer=args.decoder_layers)

            features = utils.get_config_features(config_sam)
            fid.write(','.join(map(str, features)) + ',')

            model.set_sample_config(config_sam)

            # dry runs
            for _ in range(5):
                encoder_out_test = model.encoder(src_tokens=src_tokens_test, src_lengths=src_lengths_test)

            encoder_powers = []
            print('Measuring encoder for dataset generation...')
            for _ in tqdm(range(args.latiter)):
                if args.latgpu:
                    start.record()
                elif args.latcpu:
                    start = time.time()

                powers = []
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(1)

                thread_encoder = threading.Thread(target=model.encoder, args=(src_tokens_test, src_lengths_test))         
                thread_encoder.start()

                while(thread_encoder.is_alive()):
                    powers.append(pynvml.nvmlDeviceGetPowerUsage(handle))
                    time.sleep(0.001)
                pynvml.nvmlShutdown()
                power = np.average(powers)

                if args.latgpu:
                    end.record()
                    torch.cuda.synchronize()
                    encoder_powers.append(power/1000) #the result is W
                    if not args.latsilent:
                        print('Encoder one run on GPU (for dataset generation): ', power/1000)  

                elif args.latcpu:
                    end = time.time()
                    encoder_powers.append(power)
                    if not args.latsilent:
                        print('Encoder one run on CPU (for dataset generation): ', power)

            # only use the 10% to 90% powers to avoid outliers
            encoder_powers.sort()
            encoder_powers = encoder_powers[int(args.latiter * 0.1): -max(1, int(args.latiter * 0.1))]
            print(f'Encoder power for dataset generation: Mean: {np.mean(encoder_powers)} W; \t Std: {np.std(encoder_powers)} W')

            bsz = 1
            new_order = torch.arange(bsz).view(-1, 1).repeat(1, args.beam).view(-1).long()
            if args.latgpu:
                new_order = new_order.cuda()

            encoder_out_test_with_beam = model.encoder.reorder_encoder_out(encoder_out_test, new_order)

            # dry runs
            for _ in range(5):
                model.decoder(prev_output_tokens=prev_output_tokens_test_with_beam,
                                   encoder_out=encoder_out_test_with_beam)

            # decoder is more complicated because we need to deal with incremental states and auto regressive things
            decoder_iterations_dict = {'iwslt': 23, 'wmt': 30}
            if 'iwslt' in args.arch:
                decoder_iterations = decoder_iterations_dict['iwslt']
            elif 'wmt' in args.arch:
                decoder_iterations = decoder_iterations_dict['wmt']

            decoder_powers = []
            print('Measuring decoder for dataset generation...')
            for _ in tqdm(range(args.latiter)):
                if args.latgpu:
                    start.record()
                elif args.latcpu:
                    start = time.time()
                
                powers = []
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(1)

                incre_states = {}
                for k_regressive in range(decoder_iterations):
                    thread_decoder = threading.Thread(target=model.decoder, args=((prev_output_tokens_test_with_beam[:, :k_regressive + 1], encoder_out_test_with_beam, incre_states)))         
                    thread_decoder.start()
                    while(thread_decoder.is_alive()):
                        powers.append(pynvml.nvmlDeviceGetPowerUsage(handle))
                        time.sleep(0.001)
                
                pynvml.nvmlShutdown()
                # print(powers)
                power = np.average(powers)

                if args.latgpu:
                    end.record()
                    torch.cuda.synchronize()
                    decoder_powers.append(power/1000) #the result is W
                    if not args.latsilent:
                        print('Decoder one run on GPU (for dataset generation): ', power/1000)  

                elif args.latcpu:
                    end = time.time()
                    decoder_powers.append(power)
                    if not args.latsilent:
                        print('Decoder one run on CPU (for dataset generation): ', power)

            # only use the 10% to 90% powers to avoid outliers
            decoder_powers.sort()
            decoder_powers = decoder_powers[int(args.latiter * 0.1): -max(1, int(args.latiter * 0.1))]

            print(decoder_powers)
            print(f'Decoder power for dataset generation: Mean: {np.mean(decoder_powers)} W; \t Std: {np.std(decoder_powers)} W')

            lats = [np.mean(encoder_powers), np.mean(decoder_powers), np.std(encoder_powers), np.std(decoder_powers)]
            fid.write(','.join(map(str, lats)) + '\n')

def cli_main():
    parser = options.get_training_parser()

    parser.add_argument('--latgpu', action='store_true', help='measure SubTransformer power on GPU')
    parser.add_argument('--latcpu', action='store_true', help='measure SubTransformer power on CPU')
    parser.add_argument('--latiter', type=int, default=300, help='how many iterations to run when measure the power')
    parser.add_argument('--latsilent', action='store_true', help='keep silent when measure power')

    parser.add_argument('--lat-dataset-path', type=str, default='./power_dataset/lat.tmp', help='the path to write power dataset')
    parser.add_argument('--lat-dataset-size', type=int, default=200, help='number of data points for the dataset')

    options.add_generation_args(parser)

    args = options.parse_args_and_arch(parser)

    if args.latcpu:
        args.cpu = True
        args.fp16 = False

    if args.pdb:
        pdb.set_trace()

    main(args)

if __name__ == '__main__':
    cli_main()
