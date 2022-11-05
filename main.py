import os
import torch
import yaml

from pathlib import Path
from dataloader.dataset import *
from tools.evaluation import evaluation


def main():
    args = yaml.safe_load(Path('args/main.yaml').read_text())
    
    train_item, test_item = bulid_dataset('item', args['dataset'])
    train_user, test_user = bulid_dataset('user', args['dataset'])
    
    print("Item dataset (train, test):", len(train_item), len(test_item))
    print("User dataset (train, test):", len(train_user), len(test_user))
    
    train_item_dataloader = bulid_dataloader(train_item, args['dataloader'])
    train_user_dataloader = bulid_dataloader(train_user, args['dataloader'])
    
    
    print(len(train_item_dataloader))
    print(len(train_user_dataloader))


    
if __name__=='__main__':
    main()