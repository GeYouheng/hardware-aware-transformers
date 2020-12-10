import configargparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

# hard-coded parameters(should be changed into parameters)
feature_num = 10
feature_norm = [640, 6, 2048, 6, 640, 6, 2048, 6, 6, 2, 64, 64, 512, 8192, 1]
output_norm = 10000000
header = 'encoder_embed_dim,encoder_layer_num,encoder_ffn_embed_dim_avg,encoder_self_attention_heads_avg,' + 'decoder_embed_dim,decoder_layer_num,decoder_ffn_embed_dim_avg,decoder_self_attention_heads_avg,decoder_ende_attention_heads_avg,decoder_arbitrary_ende_attn_avg,' +'l1d_misses,l1i_misses,l2_misses,l3_misses,'+'latency_mean_encoder,latency_mean_decoder,latency_std_encoder,latency_std_decoder'
device_info = [32,32,256,3072,1]

class Net(nn.Module):
    def __init__(self, feature_dim=15, hidden_dim=400, hidden_layer_num=3, output_dim=4):
        super(Net, self).__init__()

        self.first_layer = nn.Linear(feature_dim, hidden_dim)

        self.layers = nn.ModuleList()

        for i in range(hidden_layer_num):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))

        self.predict = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.first_layer(x))

        for i in range(len(self.layers)):
            x = F.relu(self.layers[i](x))

        x = self.predict(x)

        return x


def main():
	parser = configargparse.ArgumentParser()
	parser.add_argument('--device-predictor-path', type=str)
	parser.add_argument('--dataset-path', type=str)
	parser.add_argument('--augmented-path', type=str)
	args = parser.parse_args()
	print(parser)
	
	model = Net()
	model.load_state_dict(torch.load(args.device_predictor_path))
	df = pd.read_csv(args.dataset_path)
	print(df.head())
	
	net_feature = df[df.columns[:feature_num]].to_numpy()
	device_feature = np.array([device_info for _ in range(len(net_feature))])  # features of current running device

	total_feature = np.hstack((net_feature, device_feature)) / feature_norm
	x_tensor = torch.Tensor(total_feature)
	print(x_tensor.shape)

	cache_misses = model.forward(x_tensor) * output_norm
	print(cache_misses[0])
	augmented_dataset = np.hstack((net_feature, cache_misses.detach().numpy(), df[df.columns[feature_num:]].to_numpy()))
	print(augmented_dataset[0])
	print(len(augmented_dataset))
	
	with open(args.augmented_path, "w") as f:
		f.write(header + '\n')
		for line in augmented_dataset:
			f.write(','.join(map(str, line.tolist())) + '\n')
	return 
			
	
		
# python device_augment_dataset.py --device-predictor-path=device_dataset/predictors/wmt14ende_cpu_xeon.pt --dataset-path=latency_dataset/wmt14ende_cpu_xeon_all.csv --augmented-path=latency_dataset/wmt14ende_cpu_xeon_augmented.csv
if __name__ == '__main__':
	main()
