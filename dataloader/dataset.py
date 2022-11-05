import os
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader,random_split

DATA_PATH = '/home/zealot/zealot/RecSys/data/preprocessed/prepared'

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

class MergeDataset(Dataset):
    def __init__(self, user_dataloader, item_dataloader, args):
        df_session = pd.read_pickle(os.path.join(args['root_path'], 'seesion.pkl'))
        
        if data_path.endswith('.csv'):
            df_session = pd.read_csv(data_path)
        elif data_path.endswith('.pkl'):
            df_session = pd.read_pickle(data_path)
        else:
            raise NotImplementedError
        
        self.user_dataloader = user_dataloader
        self.item_dataloader = item_dataloader
        self.df_session = df_session
        
    def __len__(self):
        return len(self.user_dataloader) * len(self.item_dataloader)

    def __getitem__(self, idx):
        # user_idx = 
        # item_idx = 

def dataset_split(dataset, ratio, seed=42):
    train_size = int(len(dataset)*0.8)
    test_size = len(dataset)-train_size
    trainset, testset = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(seed))

    return trainset, testset

def bulid_dataset(data_type, args):
    if data_type == 'item':
        dataset = ItemDataset(os.path.join(args['root_path'], 'item.pkl'))
    elif data_type == 'user':
        dataset = UserDataset(os.path.join(args['root_path'], 'user.pkl'))
    else:
        raise NotImplementedError
    
    trainset, testset = dataset_split(dataset, ratio=args['split']['ratio'], seed=args['split']['seed'])
    
    return trainset, testset

def bulid_dataloader(dataset, args):
    return DataLoader(dataset, batch_size=args['batch_size'], shuffle=args['shuffle'])

def main():    
    dataset = ItemDataset(os.path.join(DATA_PATH, 'item.pkl'))
    print(len(dataset))
    
if __name__=='__main__':
    main()