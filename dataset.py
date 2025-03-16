import os
import glob
import torch
from torch.utils.data.dataset import Dataset
from utils import open_sequence, open_raw_sequence
import random
import numpy as np

NUMFRXSEQ_VAL = 15	# number of frames of each sequence to include in validation dataset

test_sequences = [
	'FlowerKids',
	'RiverBank',
	'CityAlley',
	'HoneyBee',
	'IMG_0058',
	'car-turn',
	'classic-car',
	'cows',
	'bike-packing',
	'breakdance',
	'drone',
	'horsejump-high',
	'swing',
	'monkeys',
	'varanus-tree',
	'mallard-water',
	'lions',
	'loading',
	'crossing',
	'soapbox',
	'dance-twirl',
	'dog-agility',
	'flamingo',
	'goat',
	'bear',
	'train',
	'skydive',
	'choreography',
	'bus',
	'dogs-scale'
]


class ValDataset(Dataset):
	"""Validation dataset. Loads all the images in the dataset folder on memory.
	"""
	def __init__(self, ISO, data_dir=None, gray_mode=False, num_input_frames=NUMFRXSEQ_VAL):
		self.gray_mode = gray_mode

		self.sequences = []
		for dir in glob.glob(os.path.join(data_dir, 'wb_scene_noisy', '*', f'iso{ISO}')):
			if dir.split('/')[-2] not in test_sequences:
				continue
			raw_seq_gt, _, _ = open_sequence(dir.replace('wb_scene_noisy', 'wb_scene_clean_postprocessed').replace(f'/iso{ISO}', ''),
											 gray_mode=True, expand_if_needed=False, max_num_fr=num_input_frames)
			raw_seq_noisy, _, _ = open_raw_sequence(dir, gray_mode=True, expand_if_needed=False, max_num_fr=num_input_frames, file_num=1)
			self.sequences.append((torch.from_numpy(raw_seq_noisy.astype(np.float32)),
								   torch.from_numpy(raw_seq_gt.astype(np.float32))))

	def __getitem__(self, index):
		return self.sequences[index]

	def __len__(self):
		return len(self.sequences)


class TrainDataset(Dataset):
	def __init__(self, data_dir, sequence_length, crop_size, epoch_size,
				 gray_mode=False, create_data=False):
		self.sequences = []
		self.sequence_length = sequence_length
		self.crop_size = crop_size
		self.epoch_size = epoch_size
		self.shorts = []
		for ISO in [1600, 3200, 6400, 12800, 25600]:
			for dir in glob.glob(os.path.join(data_dir, 'wb_scene_noisy', '*', f'iso{ISO}')):
				if dir.split('/')[-2] in test_sequences:
					continue
				raw_sequence_gt, _, _ = open_sequence(
					dir.replace('wb_scene_noisy', 'wb_scene_clean_postprocessed').replace(f'/iso{ISO}', ''),
					gray_mode=True,
					expand_if_needed=False, max_num_fr=100000)
				file_num = random.randint(0, 9)

				raw_sequence_noisy, _, _ = open_raw_sequence(dir,
														gray_mode=True,
														expand_if_needed=False, max_num_fr=100000, file_num=file_num)
				# for i in range(sequence_length, len(raw_sequence_gt)):
				self.shorts.append((raw_sequence_gt, raw_sequence_noisy, ISO))

	def __getitem__(self, index):
		ind = random.randrange(0, len(self.shorts))
		i = random.randrange(self.sequence_length, len(self.shorts[ind][0]))
		H, W = 1080, 1920
		x = random.randrange(0, int((W - self.crop_size - 1) / 2)) * 2
		y = random.randrange(0, int((H - self.crop_size - 1) / 2)) * 2
		cropped_raw_gt = self.shorts[ind][0][i - self.sequence_length: i][:, y:y + self.crop_size + 1, x:x + self.crop_size + 1].astype(np.float32)
		cropped_raw_noise = self.shorts[ind][1][i - self.sequence_length: i][:, y:y + self.crop_size + 1, x:x + self.crop_size + 1].astype(np.float32)
		return (torch.from_numpy(cropped_raw_noise), torch.from_numpy(cropped_raw_gt), torch.tensor(np.array(self.shorts[ind][2])))

	def __len__(self):
		return self.epoch_size