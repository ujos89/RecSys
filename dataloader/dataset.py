import os
import pandas as pd
import torch

from torch.utils.data import Dataset, random_split


class ItemDataset(Dataset):
    def __init__(self, data_path):
        if data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
        elif data_path.endswith('.pkl'):
            df = pd.read_pickle(data_path)
        else:
            raise NotImplementedError
        
        self.df = df
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        sample = self.df.iloc[idx].to_dict()
        
        # columns: id, category, comment_count, like_count, visit_count, user_id, create_date, update_date 
        counts = torch.FloatTensor([sample['comment_count'], sample['like_count'], sample['visit_count']])
        
        return sample

class UserDataset(Dataset):
    def __init__(self, data_path):
        if data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
        elif data_path.endswith('.pkl'):
            df = pd.read_pickle(data_path)
        else:
            raise NotImplementedError
        
        self.df = df
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        sample = self.df.iloc[idx].to_dict()
        
        # columns: id, login_count, login_last, bookmark, follower, following, project, projectAll
        
        return sample
    

class SessionDataset(Dataset):
    def __init__(self, data_path):
        if data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
        elif data_path.endswith('.pkl'):
            df = pd.read_pickle(data_path)
        else:
            raise NotImplementedError
        
        self.df = df
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        return 1

def dataset_split(dataset, ratio, seed=42):
    train_size = int(len(dataset)*0.8)
    test_size = len(dataset)-train_size
    trainset, testset = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(seed))

    return trainset, testset

DATA_PATH = '/home/zealot/zealot/RecSys/data/preprocessed/prepared'

def main():    
    dataset = ItemDataset(os.path.join(DATA_PATH, 'item.pkl'))
    print(len(dataset))
    
if __name__=='__main__':
    main()