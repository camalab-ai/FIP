import argparse
import torch.nn as nn
from models import CGNet_D3
from dataset import ValDataset
import os
import time
import torch
from utils import batch_psnr
from infer_utils import denoise_seq
import math


def main(**args):
	r"""Performs the main training loop
	"""
	ISO_list = [1600, 3200, 6400, 12800, 25600]
	# Load dataset
	print('> Loading datasets ...')
	dataset_val = [ValDataset(ISO=ISO, data_dir=args['valset_dir'], gray_mode=False, num_input_frames=25) for ISO in ISO_list]


	# Define GPU devices
	device_ids = [0]
	torch.backends.cudnn.benchmark = True # CUDNN optimization

	# Create model
	model = CGNet_D3()
	model = nn.DataParallel(model, device_ids=device_ids).cuda()
	resumef = os.path.join(args['log_dir'], 'ckpt.pth')

	checkpoint = torch.load(resumef)
	model.load_state_dict(checkpoint['state_dict'])

	model.eval()

	a_list = [3.513262, 6.955588, 13.486051, 26.585953, 52.032536]
	b_list = [11.917691, 38.117816, 130.818508, 484.539790, 1819.818657]

	t1 = time.time()

	levels = [0, 1, 2, 3, 4]
	with torch.no_grad():
		avg_psnr = 0
		avg_psnr_SFD = 0
		for level in levels:
			a = torch.tensor(a_list[level], dtype=torch.float).cuda()
			b = torch.tensor(math.sqrt(b_list[level]), dtype=torch.float).cuda()

			psnr_val, psnr_val_SFD = 0, 0
			for seqn_val, seq_val in dataset_val[level]:
				seq_val = seq_val.cuda()
				seqn_val = (seqn_val.cuda() - 240) / (2 ** 12 - 1 - 240)

				out_val, out_val_SFD = denoise_seq(seq=seqn_val, \
												   noise_p=a, \
												   noise_g=b, \
												   temp_psz=7, \
												   model_temporal=model)

				psnr = batch_psnr(out_val.cpu(), (seq_val - 240) / (2 ** 12 - 1 - 240), 1.)
				psnr_val += psnr
				psnr_val_SFD += psnr - batch_psnr(out_val_SFD.cpu(), (seq_val - 240) / (2 ** 12 - 1 - 240), 1.)

			psnr_val /= len(dataset_val[level])
			psnr_val_SFD /= len(dataset_val[level])
			avg_psnr_SFD += psnr_val_SFD
			avg_psnr += psnr_val
			t2 = time.time()
			print("\nPSNR_val: %.4f, on %.2f sec" % (psnr_val, (t2 - t1)))
			print("PSNR_SFD_val: %.4f, on %.2f sec" % (psnr_val_SFD, (t2 - t1)))


	avg_psnr = avg_psnr / len(levels)
	avg_psnr_SFD = avg_psnr_SFD / len(levels)
	print(avg_psnr, avg_psnr_SFD)

if __name__ == "__main__":

	parser = argparse.ArgumentParser(description="Train the denoiser")

	#Training parameters

	# Dirs
	parser.add_argument("--log_dir", type=str, default="/mnt/4TB/MA_results/I_raw_reCRVD_CGNet_D3_FIP", \
					 help='path of log files')
	parser.add_argument("--valset_dir", type=str, default='/mnt/4TB/datasets/ReCRVD/reCRVD', \
						 help='path of validation set')
	argspar = parser.parse_args()

	print("> Parameters:")
	for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
		print('\t{}: {}'.format(p, v))
	print('\n')

	main(**vars(argspar))
