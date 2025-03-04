import argparse
import numpy as np
import torch
import torch.nn as nn
import random

import models
from dataset import ValDataset


class InputPadder:
	""" Pads images such that dimensions are divisible by 8 """
	def __init__(self, dims, mode='sintel'):
		self.ht, self.wd = dims[-2:]
		pad_ht = (((self.ht // 8) + 1) * 8 - self.ht) % 8
		pad_wd = (((self.wd // 8) + 1) * 8 - self.wd) % 8
		if mode == 'sintel':
			self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
		else:
			self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]

	def pad(self, *inputs):
		return [nn.functional.pad(x, self._pad, mode='replicate') for x in inputs]

	def unpad(self,x):
		ht, wd = x.shape[-2:]
		c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
		return x[..., c[0]:c[1], c[2]:c[3]]


parser = argparse.ArgumentParser(description="Quantitative evaluation")
parser.add_argument("--data_dir", type=str, default="/mnt/4TB/datasets/DAVIS2Share/val", help='path of data dir')
parser.add_argument('--models', nargs='+', help='paths for model checkpoints', default=[
	'/mnt/4TB/MA_results/L_CGNet_CA_withCascade_1/ckpt_e79.pth',
	'/mnt/4TB/MA_results/L_CGNet_CA_withCascade_FIP_1/ckpt_e79.pth',
]
)
parser.add_argument("--temp_distance", type=int, default=1, help="Can be 1 or 2")
parser.add_argument("--samples_per_seq", type=int, default=200)
argspar = vars(parser.parse_args())


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

patch_size = 128
stride = 64
n_frames = 5
cpf = 3
mid = n_frames // 2

# change noise_std parameter here to produce results at various noise levels
noise_std = 50

device_ids = [0]

w, h, w1, h1 = 128, 128, 128, 128
dataset_val = ValDataset(argspar['data_dir'], gray_mode=False, num_input_frames=100)
crop_sides = 40
x1 = 0
y1 = 0
ps = np.array([x1 + w1 // 2])
qs = np.array([y1 + h1 // 2])
p = ps[0]
q = qs[0]
span = 1
r1, r2, r3 = 2, 2, 20
samples_per_seq = argspar['samples_per_seq']
models_list = []

paths = argspar['models']
for path in paths:
	model = models.CGNet_D2()
	model = nn.DataParallel(model, device_ids=device_ids).cuda()
	model.load_state_dict(torch.load(path)['state_dict'])
	model.eval()
	models_list.append(model)
EPE = [[] for _ in paths]
ofs = []
temp_distance = argspar['temp_distance']
of_dir = '/val_of21/' if temp_distance == 1 else '/val_of31/'
for n_seq, seq in enumerate(dataset_val):
	print(seq[1])
	padder = InputPadder(seq[0][:1].shape)
	for _ in range(samples_per_seq):
		while True:
			num, y, x = random.randrange(1 + temp_distance, seq[0].shape[0] - 2 - temp_distance), random.randrange(0, seq[0].shape[2] - 128 - 1), random.randrange(0, seq[0].shape[3] - 128 - 1)
			of_all = padder.unpad(torch.load(seq[1].replace('/val/', of_dir) + '/' + str(num).zfill(5) + '.pt'))
			of = of_all[0, :, y:y + h, x:x + w].cpu().numpy()
			of_len = ((of[:, q, p]) ** 2).sum() ** 0.5
			if r2 < of_len < r3:
				break

		ofs.append(of_len)
		grad_maps = []
		outputs = [0 for _ in paths]

		sample = torch.stack([seq[0][num + 0 - 2:num + 0 + 3][i, :, y:y + h, x:x + w] for i in range(5)]).contiguous().view(-1, 15, h,w).to(device)
		clean_image = sample[:, (mid * cpf):((mid + 1) * cpf), :, :]
		N, C, H, W = sample.shape
		noise_map = (noise_std / 255) * torch.ones(N, 1, H, W).to(device)

		fixed_noises = torch.FloatTensor(sample.size()).normal_(mean=0, std=1).cuda()
		noise = (noise_std / 255.0) * fixed_noises
		noisy_inputs = noise + sample

		noisy_inputs = noisy_inputs.requires_grad_(True)

		for model_num, model in enumerate(models_list):
			output = model(noisy_inputs, noise_map)
			# outputs[model_num] += output
			loss = 100 * output[:, :, q, p].mean()

			model.zero_grad()

			loss.backward()

			grads = noisy_inputs.grad.cpu().detach()
			noisy_inputs.grad.zero_()
			grad_maps.append(grads)

		# Create x and y coordinate grids
		x_coords = torch.linspace(0, W - 1, W)  # Range from 0 to 1 with W points
		y_coords = torch.linspace(0, H - 1, H)  # Range from 0 to 1 with H points

		# Create meshgrid
		x_grid, y_grid = torch.meshgrid(x_coords, y_coords)

		# Stack x and y grids along the last axis to create the tensor
		positions_tensor = torch.stack((x_grid, y_grid), dim=-1)

		acceptance = 0.2
		print(of[:, q, p], ((of[:, q, p]) ** 2).sum() ** 0.5)
		for model_num, path in enumerate(paths):
			argmax_positions = []
			grads = []
			grad = torch.sum(grad_maps[model_num][0, (mid-temp_distance) * cpf:(mid+1 - temp_distance) * cpf, y1:y1 + h1, x1:x1 + w1], dim=0).reshape(w1, h1)
			grads.append(grad)
			valid_pixels = grad > grad.max() * acceptance
			argmax_positions.append([
				(positions_tensor[:, :, 0][valid_pixels] * grad[valid_pixels]).sum() / grad[valid_pixels].sum(),
				(positions_tensor[:, :, 1][valid_pixels] * grad[valid_pixels]).sum() / grad[valid_pixels].sum()
			])
			argmax_position = torch.tensor(argmax_positions).mean(dim=0)
			estimated_of = argmax_position - torch.tensor([q * 1.0, p * 1.0])
			distance = (((estimated_of[[1, 0]].numpy() - (of[:, q, p])) ** 2).sum() ** 0.5).round(3)
			print('\t', estimated_of[[1, 0]], distance)
			EPE[model_num].append(distance)


print(len(ofs))
top2_per_model = [0 for _ in EPE]
top2 = 0
for of_num, of in enumerate(ofs):
	top2 += 1
	for m_num, m in enumerate(EPE):
		if EPE[m_num][of_num] < r1:
			top2_per_model[m_num] += 1
for m, (score_per_model, path) in enumerate(zip(top2_per_model, paths)):
	print(path, "PCIF", score_per_model / top2)
