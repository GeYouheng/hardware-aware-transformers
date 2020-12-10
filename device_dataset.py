# HAT: Hardware-Aware Transformers for Efficient Natural Language Processing
# Hanrui Wang, Zhanghao Wu, Zhijian Liu, Han Cai, Ligeng Zhu, Chuang Gan and Song Han
# The 58th Annual Meeting of the Association for Computational Linguistics (ACL), 2020.
# Paper: https://arxiv.org/abs/2005.14187
# Project page: https://hanruiwang.me/project_pages/hat/

import torch
import time
import pdb
import random
import subprocess
import json
import numpy as np

from fairseq import checkpoint_utils, distributed_utils, options, tasks, utils
from tqdm import tqdm


dry_run_iters = 3
dev_conf_dir = 'device_dataset/device_configs'
dynamorio_dir = '/home/parallels/Desktop/DynamoRIO-Linux-8.0.18565'


def main(args):
    utils.import_user_module(args)

    assert args.max_tokens is not None or args.max_sentences is not None, \
        'Must specify batch size either with --max-tokens or --max-sentences'

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

    # for device predictor: device dataset generation 
    # The temporary format is (SubTransformer, DeviceFeature) -> CacheMisses
    with open(args.lat_dataset_path, 'w') as fid:
        src_tokens_test = torch.tensor([dummy_src_tokens], dtype=torch.long)
        src_lengths_test = torch.tensor([dummy_sentence_length])
        prev_output_tokens_test_with_beam = torch.tensor([dummy_prev] * args.beam, dtype=torch.long)
        if args.latcpu:
            model.cpu()
            print('Measuring model cache misses on CPU for dataset generation...')
        elif args.latgpu:
            return

        feature_info = utils.get_feature_info()
        fid.write(','.join(feature_info) + ',')
        device_feature_info = ['l1d_size', 'l1i_size', 'l2_size', 'l3_size', 'num_cores']  # unit for cache sizes is KB
        fid.write(','.join(device_feature_info) + ',')
        misses_info = ['l1d_misses', 'l1i_misses', 'l2_misses', 'l3_misses']
        fid.write(','.join(misses_info) + '\n')

        device_feature_set = [[32, 32, 256, 4096, 1], [32, 32, 512, 4096, 1], [32, 32, 512, 8192, 1], [64, 64, 512, 8192, 1]]
        num_devices = len(device_feature_set)

        for i in range(args.lat_dataset_size):
            print(i)

            config_sam = utils.sample_configs(utils.get_all_choices(args), reset_rand_seed=False, super_decoder_num_layer=args.decoder_layers)
            features = utils.get_config_features(config_sam)
            fid.write(','.join(map(str, features)) + ',')

            device_id = random.randint(0, num_devices-1)  # randomly select data
            device_features = device_feature_set[device_id]
            fid.write(','.join(map(str, device_features)) + ',')
            
            model.set_sample_config(config_sam)
            # dry runs
            for _ in range(dry_run_iters):
                encoder_out_test = model.encoder(src_tokens=src_tokens_test, src_lengths=src_lengths_test)

            # pass the model(features&device_features) as parameters into a new python process
            arguments = [dynamorio_dir+'/bin64/drrun', '-t', 'drcachesim', '-config_file', dev_conf_dir+'/dev'+str(device_id)+'.conf', '--', 'python', '-u', 'encoder_infer.py', str(dummy_sentence_length)]
            print(arguments)
            enproc = subprocess.Popen([dynamorio_dir+'/bin64/drrun', '-t', 'drcachesim', '-config_file', dev_conf_dir+'/dev'+str(device_id)+'.conf', '--', 'python', '-u', 'encoder_infer.py', str(dummy_sentence_length)], stdout=subprocess.PIPE)
            while True:
                line = enproc.stdout.readline()
                if not line:
                    break
                print('line: ' + line.decode('utf-8'))
            encoder_misses = []
            print('Measuring encoder for dataset generation...')
            #TODO run model.encoder(src_tokens=src_tokens_test, src_lengths=src_lengths_test) in simulator and write results into encoder_misses
            print(encoder_misses)

            bsz = 1
            new_order = torch.arange(bsz).view(-1, 1).repeat(1, args.beam).view(-1).long()

            encoder_out_test_with_beam = model.encoder.reorder_encoder_out(encoder_out_test, new_order)

            # dry runs
            for _ in range(dry_run_iters):
                model.decoder(prev_output_tokens=prev_output_tokens_test_with_beam,
                                   encoder_out=encoder_out_test_with_beam)

            # decoder is more complicated because we need to deal with incremental states and auto regressive things
            decoder_iterations_dict = {'iwslt': 23, 'wmt': 30}
            if 'iwslt' in args.arch:
                decoder_iterations = decoder_iterations_dict['iwslt']
            elif 'wmt' in args.arch:
                decoder_iterations = decoder_iterations_dict['wmt']

            decoder_misses = []
            print('Measuring decoder for dataset generation...')
            #TODO run the below commands in simulator and write results into decoder_misses
            #incre_states = {}
            #for k_regressive in range(decoder_iterations):
            #    model.decoder(prev_output_tokens=prev_output_tokens_test_with_beam[:, :k_regressive + 1],
            #                  encoder_out=encoder_out_test_with_beam, incremental_state=incre_states)
            print(decoder_misses)

            #TODO Do we need to add the encoder/decoder misses together
            fid.write(','.join(map(str, encoder_misses)) + '\n')  #temp
            #lats = [np.mean(encoder_latencies), np.mean(decoder_latencies), np.std(encoder_latencies), np.std(decoder_latencies)]
            #fid.write(','.join(map(str, lats)) + '\n')

def cli_main():
    parser = options.get_training_parser()

    parser.add_argument('--latgpu', action='store_true', help='measure SubTransformer cache misses on GPU')
    parser.add_argument('--latcpu', action='store_true', help='measure SubTransformer cache misses on CPU')
    parser.add_argument('--latiter', type=int, default=300, help='how many iterations to run when measure the latency')  #TODO remove it later
    parser.add_argument('--latsilent', action='store_true', help='keep silent when measure cache misses')

    parser.add_argument('--lat-dataset-path', type=str, default='./device_dataset/lat.tmp', help='the path to write device dataset')
    parser.add_argument('--lat-dataset-size', type=int, default=200, help='number of data points for the dataset')

    options.add_generation_args(parser)

    args = options.parse_args_and_arch(parser)

    if args.latcpu:
        args.cpu = True
        args.fp16 = False
    else:
        print('GPU mode not supported yet')
        return

    if args.pdb:
        pdb.set_trace()

    main(args)

if __name__ == '__main__':
    cli_main()
