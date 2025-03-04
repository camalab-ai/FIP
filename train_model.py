import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from models import CGNet_D2
from dataset import ValDataset
from dataloaders import train_dali_loader
from utils import close_logger, init_logging, normalize_augment
from train_common import resume_training, log_train_psnr, \
					validate_and_log, save_model_checkpoint, CosineAnnealingRestartLR


def main(**args):
	r"""Performs the main training loop
	"""
	# Load dataset
	print('> Loading datasets ...')
	dataset_val = ValDataset(valsetdir=args['valset_dir'], gray_mode=False)
	ctrl_fr_idx = (args['temp_patch_size'] - 1) // 2

	wandb.init(project='FIP', entity=args['wandb_entity'])
	wandb.run.name = args['log_dir'].split('/')[-1]
	wandb.run.save()
	# Init loggers
	logger = init_logging(args)

	# Define GPU devices
	device_ids = [0]
	torch.backends.cudnn.benchmark = True # CUDNN optimization

	# Create model
	model = CGNet_D2()
	model = nn.DataParallel(model, device_ids=device_ids).cuda()
	print(sum(p.numel() for p in model.parameters()))
	# Define loss

	class PSNRLoss(nn.Module):

		def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
			super(PSNRLoss, self).__init__()
			assert reduction == 'mean'
			self.loss_weight = loss_weight
			self.scale = 10 / 2.302585092994046
			self.toY = toY
			self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
			self.first = True

		def forward(self, pred, target):
			assert len(pred.size()) == 4
			if self.toY:
				if self.first:
					self.coef = self.coef.to(pred.device)
					self.first = False

				pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
				target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

				pred, target = pred / 255., target / 255.
				pass
			assert len(pred.size()) == 4

			return self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()

	criterion = PSNRLoss()
	criterion.cuda()

	# Optimizer
	optimizer = optim.AdamW(model.parameters(), lr=args['lr'], betas=[0.9, 0.9])
	scheduler = CosineAnnealingRestartLR(optimizer, periods=[args['pretraining_epochs'], args['milestone'][0]-args['pretraining_epochs'], args['epochs']-args['milestone'][0]], restart_weights=[1, 1, 1], eta_min=[0.0003,0.0003,0.000001])

	# Resume training or start anew
	start_epoch, training_params = resume_training(args, model, optimizer, scheduler)

	# Training
	start_time = time.time()
	for epoch in range(start_epoch, args['epochs']):
		# Set learning rate
		print(epoch)
		is_pretraining = epoch < args['pretraining_epochs']
		for g in optimizer.param_groups:
			current_lr = g['lr']
			break
		print('\nlearning rate %f' % current_lr)
		for num, milestone in enumerate(args['milestone']):
			if epoch < milestone:
				train_loader = train_dali_loader(batch_size=int(args['batch_size'][num]), \
												  file_root=args['trainset_dir'], \
												  sequence_length=args['temp_patch_size'], \
												  crop_size=args['patch_size'][num], \
												  epoch_size=args['max_number_patches'], \
												  random_shuffle=True, \
												  temp_stride=3,
												  seed=1)
				num_minibatches = int(args['max_number_patches'] // args['batch_size'][num])
				print("\t# of training samples: %d\n" % int(args['max_number_patches']))
				break

		# train
		for i, orig_data in enumerate(train_loader):
			datas = [orig_data]
			data = [{'data': torch.cat([d[0]['data'] for d in datas], dim=0)}]

			# Pre-training step
			model.train()

			# When optimizer = optim.Optimizer(net.parameters()) we only zero the optim's grads
			optimizer.zero_grad()

			# convert inp to [N, num_frames*C. H, W] in  [0., 1.] from [N, num_frames, C. H, W] in [0., 255.]
			# extract ground truth (central frame)
			img_train, gt_train = normalize_augment(data[0]['data'], ctrl_fr_idx)
			N, _, H, W = img_train.size()

			# std dev of each sequence
			stdn = torch.empty((N, 1, 1, 1)).cuda().uniform_(args['noise_ival'][0], to=args['noise_ival'][1])
			# draw noise samples from std dev tensor
			noise = torch.zeros_like(img_train)
			noise = torch.normal(mean=noise, std=stdn.expand_as(noise))

			if is_pretraining:
				imgn_train = img_train
			else:
				imgn_train = img_train + noise

			# Send tensors to GPU
			gt_train = gt_train.cuda(non_blocking=True)
			imgn_train = imgn_train.cuda(non_blocking=True)
			noise = noise.cuda(non_blocking=True)
			noise_map = stdn.expand((N, 1, H, W)).cuda(non_blocking=True) # one channel per image

			# Evaluate model and optimize it
			if is_pretraining:
				out_train, inter = model(imgn_train, noise_map, noise=noise, epoch=epoch, pretraining_epochs=args['pretraining_epochs'])
				loss = (criterion(gt_train, out_train) + criterion(gt_train, inter)) / (N * 2)
			elif epoch == args['pretraining_epochs'] and args['pretraining_epochs'] > 0 and not args['no_postfipepoch']:
				out_train = model(imgn_train, noise_map, factor=float(i + 1) / num_minibatches)
				loss = criterion(gt_train, out_train) / (N * 2)
			else:
				out_train = model(imgn_train, noise_map)
				loss = criterion(gt_train, out_train) / (N*2)
			loss.backward()
			optimizer.step()

			# Results
			if training_params['step'] % args['save_every'] == 0:
				# Apply regularization by orthogonalizing filters
				if training_params['step'] % (50*args['save_every']) == 0:
					# Compute training PSNR
					log_train_psnr(out_train, \
									gt_train, \
									loss, \
									epoch, \
									i, \
									num_minibatches, \
									training_params)
			# update step counter
			training_params['step'] += 1
		if epoch + 1 < args['epochs']:
			scheduler.step()

		# Call to model.eval() to correctly set the BN layers before inference
		model.eval()

		# Validation and log images
		psnr_val = validate_and_log(
						model_temp=model, \
						dataset_val=dataset_val, \
						valnoisestd=args['val_noiseL'], \
						temp_psz=args['temp_patch_size'], \
						epoch=epoch, \
						lr=current_lr
						)
		print(psnr_val)
		# save model and checkpoint
		training_params['start_epoch'] = epoch + 1
		save_model_checkpoint(model, args, optimizer, training_params, epoch)

	# Print elapsed time
	elapsed_time = time.time() - start_time
	print('Elapsed time {}'.format(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))

	# Close logger file
	close_logger(logger)


if __name__ == "__main__":

	parser = argparse.ArgumentParser(description="Train the denoiser")

	#Training parameters
	parser.add_argument("--batch_size", type=int, nargs=6, default=[8,8,8,4,2,2], 	\
					 help="Training batch size")
	parser.add_argument("--epochs", "--e", type=int, default=80, \
					 help="Number of total training epochs")
	parser.add_argument("--pretraining_epochs", "--pe", type=int, default=10)
	parser.add_argument("--no_postfipepoch", "--no_pfe", action='store_true')
	parser.add_argument("--resume_training", "--r", action='store_true',\
						help="resume training from a previous checkpoint")
	parser.add_argument("--milestone", type=int, nargs=6, default=[17, 37, 54, 66, 74, 80], \
						help="When to decay learning rate; should be lower than 'epochs'")
	parser.add_argument("--lr", type=float, default=1e-3, \
					 help="Initial learning rate")
	parser.add_argument("--no_orthog", action='store_true',\
						help="Don't perform orthogonalization as regularization")
	parser.add_argument("--save_every", type=int, default=10,\
						help="Number of training steps to log psnr and perform \
						orthogonalization")
	parser.add_argument("--save_every_epochs", type=int, default=1,\
						help="Number of training epochs to save state")
	parser.add_argument("--noise_ival", nargs=2, type=int, default=[5, 50], \
					 help="Noise training interval")
	parser.add_argument("--val_noiseL", type=float, default=25, \
						help='noise level used on validation set')
	# Preprocessing parameters
	parser.add_argument("--patch_size", "--p", type=int, nargs=6, default=[128, 160, 192, 256, 320, 384], help="Patch size")
	parser.add_argument("--temp_patch_size", "--tp", type=int, default=5, help="Temporal patch size")
	parser.add_argument("--max_number_patches", "--m", type=int, default=38400, \
						help="Maximum number of patches")
	# Dirs
	parser.add_argument("--log_dir", type=str, default="/mnt/4TB/MA_results/L_CGNet_D2_FIP", \
					 help='path of log files')
	parser.add_argument("--trainset_dir", type=str, default='/mnt/4TB/datasets/DAVIS2Share/videos/train_official', \
					 help='path of trainset')
	parser.add_argument("--valset_dir", type=str, default='/mnt/4TB/datasets/DAVIS2Share/val', \
						 help='path of validation set')
	parser.add_argument("--wandb_entity", type=str, default='PKO-personal')
	argspar = parser.parse_args()

	# Normalize noise between [0, 1]
	argspar.val_noiseL /= 255.
	argspar.noise_ival[0] /= 255.
	argspar.noise_ival[1] /= 255.

	print("\n### Training CGNet-D2 ###")
	print("> Parameters:")
	for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
		print('\t{}: {}'.format(p, v))
	print('\n')

	main(**vars(argspar))
