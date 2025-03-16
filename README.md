Code for training CGNet-D3 on reCRVD dataset
### Environment

The code runs on Python 3.8. You can create a virtualenv by running
```
virtualenv -p /usr/bin/python3.8 venv
source venv/bin/activate
pip install -r requirements.txt
```

### Training

If you want to train models first:

1. Login to wandb from your console
2. Train model:

a) with FIP and PostFip epoch:

```
python train_model.py \
	--trainset_dir <path_to_ReCRVD_dataset> \
	--valset_dir <path_to_ReCRVD_dataset> \
	--log_dir <path_to_dir_for_chkecpoints> \
	--wandb_entity <name_of_wandb_entity>
```

b) without FIP:
```
python train_model.py \
	--trainset_dir <path_to_ReCRVD_dataset> \
	--valset_dir <path_to_ReCRVD_dataset> \
	--log_dir <path_to_dir_for_chkecpoints> \
	--wandb_entity <name_of_wandb_entity>
	--pe 0
```

### Quantitative evaluation:
```
python test_model.py
    --valset_dir <path_to_ReCRVD_dataset> \
	--log_dir <path_to_dir_with_chkecpoints>... <path_to_modelN>
```

### Data
ReCRVD dataset can be downloaded following instructions from the authors:
https://github.com/cao-cong/RViDeformer