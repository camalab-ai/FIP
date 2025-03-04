import os
import glob
import torch
from torch.utils.data.dataset import Dataset
from utils import open_sequence

NUMFRXSEQ_VAL = 15	# number of frames of each sequence to include in validation dataset
VALSEQPATT = '*' # pattern for name of validation sequence

class ValDataset(Dataset):
	"""Validation dataset. Loads all the images in the dataset folder on memory.
	"""
	def __init__(self, valsetdir=None, gray_mode=False, num_input_frames=NUMFRXSEQ_VAL, seq_num=None):
		self.gray_mode = gray_mode

		# Look for subdirs with individual sequences
		seqs_dirs = sorted(glob.glob(os.path.join(valsetdir, VALSEQPATT)))
		print([(n,s) for n, s in enumerate(seqs_dirs)])
		if seq_num is not None:
			seqs_dirs = seqs_dirs[seq_num: seq_num + 1]

		# open individual sequences and append them to the sequence list
		self.sequences = []
		self.sequences_paths = []
		for seq_dir in seqs_dirs:
			seq, _, _ = open_sequence(seq_dir, gray_mode, expand_if_needed=False, \
							 max_num_fr=num_input_frames)
			# seq is [num_frames, C, H, W]
			self.sequences.append(seq)
			self.sequences_paths.append(seq_dir)

	def __getitem__(self, index):
		return torch.from_numpy(self.sequences[index]), self.sequences_paths[index]

	def __len__(self):
		return len(self.sequences)
