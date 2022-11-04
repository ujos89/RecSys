import os
import torch

from dataloader.dataset import ItemDataset, UserDataset, dataset_split
from tools.evaluation import evaluation

DATA_PATH = '/home/zealot/zealot/RecSys/data/preprocessed/prepared'

def main():
    dataset_item = ItemDataset(os.path.join(DATA_PATH, 'item.pkl'))
    train_item, test_item = dataset_split(dataset_item, 0.8)
    
    dataset_user = UserDataset(os.path.join(DATA_PATH, 'user.pkl'))
    train_user, test_user = dataset_split(dataset_user, 0.8)
    
    print("Item dataset (train, test):", len(train_item), len(test_item))
    print("User dataset (train, test):", len(train_user), len(test_user))
    
    
    
    
if __name__=='__main__':
    main()