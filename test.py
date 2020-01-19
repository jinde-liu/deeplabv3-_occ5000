import os
import numpy as np
from dataloaders import make_data_loader
from args import Args_occ5000
args = Args_occ5000()
kwargs = {'num_workers': args.workers, 'pin_memory': True}
train_loader, val_loader, test_loader, nclass = make_data_loader(args, **kwargs)
for sample in train_loader:
    image = sample['image']
    target = sample['label']
    print(target)