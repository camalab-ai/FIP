Code for training CGNet-D2
### Environment

The code runs on Python 3.8. You can create a virtualenv by running
```
virtualenv -p /usr/bin/python3.8 venv
source venv/bin/activate
pip install -r requirements.txt
pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-cuda100==1.2.0
```

### Training

If you want to train models first:

1. Login to wandb from your console
2. Train model:

a) with FIP and PostFip epoch:

```
train_model.py \
	--trainset_dir <path_to_input_mp4s> \
	--valset_dir <path_to_val_sequences> \
	--log_dir <path_to_dir_for_chkecpoints> \
	--wandb_entity <name_of_wandb_entity>
```

b) with FIP and without PostFip epoch:
```
train_model.py \
	--trainset_dir <path_to_input_mp4s> \
	--valset_dir <path_to_val_sequences> \
	--log_dir <path_to_dir_for_chkecpoints> \
	--wandb_entity <name_of_wandb_entity>
	--no_pfe
```
c) with FIP and without PostFip epoch:
```
train_model.py \
	--trainset_dir <path_to_input_mp4s> \
	--valset_dir <path_to_val_sequences> \
	--log_dir <path_to_dir_for_chkecpoints> \
	--wandb_entity <name_of_wandb_entity>
	--pe 0
```
