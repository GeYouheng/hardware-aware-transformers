import sys

import torch
from fairseq import checkpoint_utils, distributed_utils, options, tasks, utils, models

def main():
	argv = sys.argv
	print(argv)
	model = torch.load('/home/parallels/Desktop/hardware-aware-transformers/temp.pt')
	
	dummy_sentence_length = int(argv[1])
	dummy_src_tokens = [2] + [7] * (dummy_sentence_length - 1)
	dummy_prev = [7] * (dummy_sentence_length - 1) + [2]
	src_tokens_test = torch.tensor([dummy_src_tokens], dtype=torch.long)
	src_lengths_test = torch.tensor([dummy_sentence_length])
	model.encoder(src_tokens=src_tokens_test, src_lengths=src_lengths_test)  # comment this line if want to measure the offset

if __name__ == '__main__':
    main()
