import os
import time
import torch
from utils import batch_psnr
from infer_utils import denoise_seq
import wandb

def	resume_training(argdict, model, optimizer, scheduler):
	""" Resumes previous training or starts anew
	"""
	if argdict['resume_training']:
		resumef = os.path.join(argdict['log_dir'], 'ckpt.pth')
		if os.path.isfile(resumef):
			checkpoint = torch.load(resumef)
			print("> Resuming previous training")
			model.load_state_dict(checkpoint['state_dict'])
			optimizer.load_state_dict(checkpoint['optimizer'])
			new_epoch = argdict['epochs']
			new_milestone = argdict['milestone']
			current_lr = argdict['lr']
			argdict = checkpoint['args']
			training_params = checkpoint['training_params']
			start_epoch = training_params['start_epoch']
			for i in range(start_epoch):
				scheduler.step()
			argdict['epochs'] = new_epoch
			argdict['milestone'] = new_milestone
			argdict['lr'] = current_lr
			print("=> loaded checkpoint '{}' (epoch {})"\
				  .format(resumef, start_epoch))
			print("=> loaded parameters :")
			print("==> checkpoint['optimizer']['param_groups']")
			print("\t{}".format(checkpoint['optimizer']['param_groups']))
			print("==> checkpoint['training_params']")
			for k in checkpoint['training_params']:
				print("\t{}, {}".format(k, checkpoint['training_params'][k]))
			argpri = checkpoint['args']
			print("==> checkpoint['args']")
			for k in argpri:
				print("\t{}, {}".format(k, argpri[k]))

			argdict['resume_training'] = False
		else:
			raise Exception("Couldn't resume training with checkpoint {}".\
				   format(resumef))
	else:
		start_epoch = 0
		training_params = {}
		training_params['step'] = 0
		training_params['current_lr'] = 0

	return start_epoch, training_params

import math
from torch.optim.lr_scheduler import _LRScheduler

def get_position_from_periods(iteration, cumulative_period):
	"""Get the position from a period list.

	It will return the index of the right-closest number in the period list.
	For example, the cumulative_period = [100, 200, 300, 400],
	if iteration == 50, return 0;
	if iteration == 210, return 2;
	if iteration == 300, return 2.

	Args:
		iteration (int): Current iteration.
		cumulative_period (list[int]): Cumulative period list.

	Returns:
		int: The position of the right-closest number in the period list.
	"""
	for i, period in enumerate(cumulative_period):
		if iteration < period:
			return i

class CosineAnnealingRestartLR(_LRScheduler):
	""" Cosine annealing with restarts learning rate scheme.

	An example of config:
	periods = [10, 10, 10, 10]
	restart_weights = [1, 0.5, 0.5, 0.5]
	eta_min=1e-7

	It has four cycles, each has 10 iterations. At 10th, 20th, 30th, the
	scheduler will restart with the weights in restart_weights.

	Args:
		optimizer (torch.nn.optimizer): Torch optimizer.
		periods (list): Period for each cosine anneling cycle.
		restart_weights (list): Restart weights at each restart iteration.
			Default: [1].
		eta_min (list): The minimum lr. Default: 0.
		last_epoch (int): Used in _LRScheduler. Default: -1.
	"""

	def __init__(self, optimizer, periods, restart_weights=(1,), eta_min=(1e-6,), last_epoch=-1):
		self.periods = periods
		self.restart_weights = restart_weights
		self.eta_min = eta_min
		assert (len(self.periods) == len(
			self.restart_weights)), 'periods and restart_weights should have the same length.'
		self.cumulative_period = [sum(self.periods[0:i + 1]) for i in range(0, len(self.periods))]
		super(CosineAnnealingRestartLR, self).__init__(optimizer, last_epoch)

	def get_lr(self):
		idx = get_position_from_periods(self.last_epoch, self.cumulative_period)
		current_weight = self.restart_weights[idx]
		nearest_restart = 0 if idx == 0 else self.cumulative_period[idx - 1]
		current_period = self.periods[idx]

		return [
			self.eta_min[idx] + current_weight * 0.5 * (base_lr - self.eta_min[idx]) *
			(1 + math.cos(math.pi * ((self.last_epoch - nearest_restart) / (current_period-1))))
			for base_lr in self.base_lrs
		]


def	log_train_psnr(result, imsource, loss, epoch, idx, num_minibatches, training_params):
	'''Logs trai loss.
	'''
	#Compute pnsr of the whole batch
	psnr_train = batch_psnr(result, imsource, 1., val=False)

	# Log the scalar values
	wandb_logs = {
		'loss': loss.item(),
		'train_psnr':psnr_train
	}
	wandb.log(wandb_logs)
# 	writer.add_scalar('PSNR on training data', psnr_train, \
# 		  training_params['step'])
	print("[epoch {}][{}/{}] loss: {:1.4f} PSNR_train: {:1.4f}".\
		  format(epoch+1, idx+1, num_minibatches, loss.item(), psnr_train))

def save_model_checkpoint(model, argdict, optimizer, train_pars, epoch):
	"""Stores the model parameters under 'argdict['log_dir'] + '/net.pth'
	Also saves a checkpoint under 'argdict['log_dir'] + '/ckpt.pth'
	"""
	torch.save(model.state_dict(), os.path.join(argdict['log_dir'], 'net.pth'))
	save_dict = { \
		'state_dict': model.state_dict(), \
		'optimizer' : optimizer.state_dict(), \
		'training_params': train_pars, \
		'args': argdict\
		}
	torch.save(save_dict, os.path.join(argdict['log_dir'], 'ckpt.pth'))

	if epoch % argdict['save_every_epochs'] == 0:
		torch.save(save_dict, os.path.join(argdict['log_dir'], 'ckpt_e{}.pth'.format(epoch)))
	del save_dict

def validate_and_log(model_temp, dataset_val, temp_psz, \
					 epoch, lr):
	"""Validation step after the epoch finished
	"""
	iso_list = [1600, 3200, 6400, 12800, 25600]
	a_list = [3.513262, 6.955588, 13.486051, 26.585953, 52.032536]
	b_list = [11.917691, 38.117816, 130.818508, 484.539790, 1819.818657]

	t1 = time.time()

	levels = [0,1,2,3,4]
	wandb_logs = {
		'Learning rate': lr,
		'epoch': epoch,
	}
	with torch.no_grad():
		avg_psnr = 0
		avg_psnr_SFD = 0
		for level in levels:
			a = torch.tensor(a_list[level], dtype=torch.float).cuda()
			b = torch.tensor(math.sqrt(b_list[level ]), dtype=torch.float).cuda()

			psnr_val, psnr_val_SFD = 0, 0
			for seqn_val, seq_val in dataset_val[level]:
				seq_val = seq_val.cuda()
				seqn_val = (seqn_val.cuda() - 240) / (2 ** 12 - 1 - 240)

				out_val, out_val_SFD = denoise_seq(seq=seqn_val, \
												 noise_p=a, \
												 noise_g=b, \
												 temp_psz=temp_psz, \
												 model_temporal=model_temp)

				psnr = batch_psnr(out_val.cpu(), (seq_val - 240) / (2 ** 12 - 1 - 240), 1.)
				psnr_val += psnr
				psnr_val_SFD += psnr - batch_psnr(out_val_SFD.cpu(), (seq_val - 240) / (2 ** 12 - 1 - 240), 1.)

			psnr_val /= len(dataset_val[level])
			psnr_val_SFD /= len(dataset_val[level])
			avg_psnr_SFD += psnr_val_SFD
			avg_psnr += psnr_val
			t2 = time.time()
			print("\n[epoch %d] PSNR_val: %.4f, on %.2f sec" % (epoch+1, psnr_val, (t2-t1)))
			print("PSNR_SFD_val: %.4f, on %.2f sec" % (psnr_val_SFD, (t2-t1)))

			wandb_logs[f'val/PSNR_level_{level}'] = psnr_val

	avg_psnr = avg_psnr / len(levels)
	avg_psnr_SFD = avg_psnr_SFD / len(levels)
	wandb_logs[f'val/PSNR_mean'] = avg_psnr
	wandb_logs[f'val/PSNR_SFD_mean'] = avg_psnr_SFD
	wandb.log(wandb_logs)
	print(avg_psnr, avg_psnr_SFD)
	return avg_psnr, avg_psnr_SFD