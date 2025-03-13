import os
import torch
import torch.nn as nn
import models
from dataset import ValDataset
import torchvision
from skimage.metrics import peak_signal_noise_ratio
import argparse


parser = argparse.ArgumentParser(description="Flow maps generation")
# Dirs
parser.add_argument("--data_dir", type=str, default="/mnt/4TB/datasets/DAVIS/frames/val", help='path of data dir')
parser.add_argument("--output_dir", type=str, default='PCIF_vis_CGNET/', help='path of output dir')
parser.add_argument('--models', nargs='+', help='paths for model checkpoints', default=[
	'/mnt/4TB/MA_results/L_CGNet_CA_withCascade_1/ckpt_e79.pth',
	'/mnt/4TB/MA_results/L_CGNet_CA_withCascade_FIP_1/ckpt_e79.pth',
])
parser.add_argument('--seq_num', type=int, default=26)
parser.add_argument('--frame_num', type=int, default=20)
argspar = vars(parser.parse_args())


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device_ids = [0]

n_frames = 5
cpf = 3
mid = n_frames // 2

# Editable
patch_size = 256
noise_std = 50
x = 300
y = 80
w, h, w1, h1 = patch_size, patch_size, patch_size, patch_size


crop_sides = 30
dil = 1

models_list = []
paths = argspar['models']
for path in paths:
	model = models.CGNet_D2()
	model = nn.DataParallel(model, device_ids=device_ids).cuda()
	model.load_state_dict(torch.load(path)['state_dict'])
	model = model.module
	model.eval()
	models_list.append(model)

seq_num = argspar['seq_num']
frame_num = argspar['frame_num']

print(seq_num)

dataset_val = ValDataset(argspar['data_dir'], gray_mode=False, num_input_frames=200, seq_num=seq_num)
os.makedirs(argspar['output_dir'], exist_ok=True)
print(frame_num)
of21 = torch.load(dataset_val[0][1].replace('/frames/val/', '/val_of21/') + '.pt')[frame_num-1][:,y:y+h, x:x+w].cpu()
of31 = torch.load(dataset_val[0][1].replace('/frames/val/', '/val_of31/') + '/' + str(frame_num).zfill(5)+ '.pt')[0,:,y:y+h, x:x+w].cpu()


outputs = [0 for _ in paths]
sample = torch.stack([dataset_val[0][0][frame_num - 2:frame_num + 3][i, :, y:y + h, x:x + w] for i in range(n_frames)]).contiguous().view(-1, n_frames*cpf, h, w).to(device)
torchvision.utils.save_image(sample[:, mid*cpf:(mid+1)*cpf, crop_sides:-crop_sides, crop_sides:-crop_sides], os.path.join(argspar['output_dir'], f'patch_{seq_num}_{frame_num}.png'))
clean_image = sample[:, mid * cpf:(mid + 1) * cpf, :, :]
N, C, H, W = sample.shape
noise_map = (noise_std / 255) * torch.ones(N, 1, H, W).to(device)

x_coords = torch.linspace(0, W-1, W)  # Range from 0 to 1 with W points
y_coords = torch.linspace(0, H-1, H)  # Range from 0 to 1 with H points
x_grid, y_grid = torch.meshgrid(x_coords, y_coords)
positions_tensor = torch.stack((x_grid, y_grid), dim=-1)
estimated_of_map = [torch.zeros(2, patch_size, patch_size) for _ in paths]
estimated_of_map_m2 = [torch.zeros(2, patch_size, patch_size) for _ in paths]


fixed_noises = torch.FloatTensor(sample.size()).normal_(mean=0, std=1).cuda()
noise = (noise_std / 255.0) * fixed_noises
noisy_inputs = noise + sample
noisy_inputs = noisy_inputs.requires_grad_(True)

for model_num, model in enumerate(models_list):

	output = model(noisy_inputs, noise_map)
	outputs[model_num] = output
	print(peak_signal_noise_ratio(torch.clip(output[:,:,crop_sides: -crop_sides], 0, 1).cpu().detach().numpy(), sample[:, mid * cpf:(mid + 1) * cpf, crop_sides: -crop_sides].cpu().detach().numpy()))
	for p in range(crop_sides, patch_size - crop_sides, dil):
		for q in range(crop_sides, patch_size - crop_sides, dil):
			loss = 100 * output[:, :, q, p].mean()

			loss.backward(retain_graph=True)

			grad_maps = noisy_inputs.grad.cpu().detach()
			noisy_inputs.grad.zero_()

			acceptance = 0.2
			grad = torch.sum(grad_maps[0, 1 * 3:(1 + 1) * 3, :h1, :w1], dim=0).reshape(w1, h1)
			valid_pixels = grad > grad.max() * acceptance
			argmax_position = [
				(positions_tensor[:, :, 0][valid_pixels] * grad[valid_pixels]).sum() / grad[valid_pixels].sum(),
				(positions_tensor[:, :, 1][valid_pixels] * grad[valid_pixels]).sum() / grad[valid_pixels].sum()
			]
			argmax_position = torch.tensor(argmax_position)
			estimated_of = argmax_position - torch.tensor([q * 1.0, p * 1.0])
			estimated_of_map[model_num][:, q, p] = estimated_of[[1, 0]]

			grad = torch.sum(grad_maps[0, :3, :h1, :w1], dim=0).reshape(w1, h1)
			valid_pixels = grad > grad.max() * acceptance
			argmax_position = [
				(positions_tensor[:, :, 0][valid_pixels] * grad[valid_pixels]).sum() / grad[valid_pixels].sum(),
				(positions_tensor[:, :, 1][valid_pixels] * grad[valid_pixels]).sum() / grad[valid_pixels].sum()
			]
			argmax_position = torch.tensor(argmax_position)
			estimated_of = argmax_position - torch.tensor([q * 1.0, p * 1.0])
			estimated_of_map_m2[model_num][:, q, p] = estimated_of[[1, 0]]

optical_flows = [
	of21,
	estimated_of_map[0], estimated_of_map[1],
	of31,
	estimated_of_map_m2[0], estimated_of_map_m2[1],
]
epe0 = (((of21 - estimated_of_map[0])[:,crop_sides:-crop_sides:dil, crop_sides:-crop_sides:dil]**2).sum(0)**0.5).mean()
epe1 = (((of21 - estimated_of_map[1])[:,crop_sides:-crop_sides:dil, crop_sides:-crop_sides:dil]**2).sum(0)**0.5).mean()
print('EPE: ', ((of21**2).sum(0)**0.5).mean(), epe0 , epe1)

torch.save(torch.stack(optical_flows, dim=0), os.path.join(argspar['output_dir'], f'{seq_num}_{frame_num}_flows.pt'))
torch.save(sample[:, :9], os.path.join(argspar['output_dir'], f'{seq_num}_{frame_num}_GT.pt'))
torch.save(noisy_inputs, os.path.join(argspar['output_dir'], f'{seq_num}_{frame_num}_noisy.pt'))

for model_num, model in enumerate(models_list):
	torch.save(outputs[model_num], os.path.join(argspar['output_dir'], f'{seq_num}_{frame_num}_model_{model_num}.pt'))

optical_flows_vis = torchvision.utils.flow_to_image(torch.stack(optical_flows, dim=0)[:,:, crop_sides:-crop_sides:dil, crop_sides:-crop_sides:dil].float()) /255.0
optical_flows_vis = torch.cat([
	sample[:, :9].cpu().view(3,3, patch_size,patch_size)[:,:, crop_sides:-crop_sides:dil, crop_sides:-crop_sides:dil],
	noisy_inputs[:, :9].cpu().view(3,3, patch_size,patch_size)[:,:, crop_sides:-crop_sides:dil, crop_sides:-crop_sides:dil],
	optical_flows_vis[:6],
], dim=0)
torchvision.utils.save_image(torchvision.utils.make_grid(optical_flows_vis, nrow=3), os.path.join(argspar['output_dir'], f'{seq_num}_{frame_num}_flows.png'))
