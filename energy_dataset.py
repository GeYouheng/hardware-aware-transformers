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

delay = 0.1

def measureEnergy(runningThread, delay):
    powers = []

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(1)
    # while(~runningThread.is_alive()):
    i=0
    while(i<10):
        powers.append(pynvml.nvmlDeviceGetPowerUsage(handle))
        time.sleep(delay)
        i+=1
    pynvml.nvmlShutdown()

    energy = np.sum(powers)*delay

    print(powers)
    print(energy)

    return energy

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

    # for energy predictor: energy dataset generation
    with open(args.lat_dataset_path, 'w') as fid:
        src_tokens_test = torch.tensor([dummy_src_tokens], dtype=torch.long)
        src_lengths_test = torch.tensor([dummy_sentence_length])
        prev_output_tokens_test_with_beam = torch.tensor([dummy_prev] * args.beam, dtype=torch.long)
        if args.latcpu:
            model.cpu()
            print('Measuring model energy on CPU for dataset generation...')
        elif args.latgpu:
            model.cuda()
            src_tokens_test = src_tokens_test.cuda()
            src_lengths_test = src_lengths_test.cuda()
            prev_output_tokens_test_with_beam = prev_output_tokens_test_with_beam.cuda()
            src_tokens_test.get_device()
            print('Measuring model energy on GPU for dataset generation...')
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

        feature_info = utils.get_feature_info()
        fid.write(','.join(feature_info) + ',')
        energy_info = ['energy_mean_encoder', 'energy_mean_decoder', 'energy_std_encoder', 'energy_std_decoder']
        fid.write(','.join(energy_info) + '\n')

        for i in range(args.lat_dataset_size):
            print(i)
            config_sam = utils.sample_configs(utils.get_all_choices(args), reset_rand_seed=False, super_decoder_num_layer=args.decoder_layers)

            features = utils.get_config_features(config_sam)
            fid.write(','.join(map(str, features)) + ',')

            model.set_sample_config(config_sam)

            # dry runs
            for _ in range(5):
                encoder_out_test = model.encoder(src_tokens=src_tokens_test, src_lengths=src_lengths_test)

            encoder_energies = []
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
                    encoder_energies.append(start.elapsed_time(end)*power/1000000) #the result is J
                    if not args.latsilent:
                        print('Encoder one run on GPU (for dataset generation): ', start.elapsed_time(end)*power/1000000)  

                elif args.latcpu:
                    end = time.time()
                    encoder_energies.append((end - start) * power / 1000)
                    if not args.latsilent:
                        print('Encoder one run on CPU (for dataset generation): ', (end - start)*power/1000)

            # only use the 10% to 90% energies to avoid outliers
            encoder_energies.sort()
            encoder_energies = encoder_energies[int(args.latiter * 0.1): -max(1, int(args.latiter * 0.1))]
            print(f'Encoder energy for dataset generation: Mean: {np.mean(encoder_energies)} J; \t Std: {np.std(encoder_energies)} J')

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

            decoder_energies = []
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
                    decoder_energies.append(start.elapsed_time(end)*power/1000000) #the result is J
                    if not args.latsilent:
                        print('Decoder one run on GPU (for dataset generation): ', start.elapsed_time(end)*power/1000000)  

                elif args.latcpu:
                    end = time.time()
                    decoder_energies.append((end - start) * power / 1000)
                    if not args.latsilent:
                        print('Decoder one run on CPU (for dataset generation): ', (end - start)*power)

            # only use the 10% to 90% energies to avoid outliers
            decoder_energies.sort()
            decoder_energies = decoder_energies[int(args.latiter * 0.1): -max(1, int(args.latiter * 0.1))]

            print(decoder_energies)
            print(f'Decoder energy for dataset generation: Mean: {np.mean(decoder_energies)} J; \t Std: {np.std(decoder_energies)} J')

            lats = [np.mean(encoder_energies), np.mean(decoder_energies), np.std(encoder_energies), np.std(decoder_energies)]
            fid.write(','.join(map(str, lats)) + '\n')

def cli_main():
    parser = options.get_training_parser()

    parser.add_argument('--latgpu', action='store_true', help='measure SubTransformer energy on GPU')
    parser.add_argument('--latcpu', action='store_true', help='measure SubTransformer energy on CPU')
    parser.add_argument('--latiter', type=int, default=300, help='how many iterations to run when measure the energy')
    parser.add_argument('--latsilent', action='store_true', help='keep silent when measure energy')

    parser.add_argument('--lat-dataset-path', type=str, default='./energy_dataset/lat.tmp', help='the path to write energy dataset')
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
