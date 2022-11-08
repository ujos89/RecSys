import pandas as pd
import os
import torch
import time
import random

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

def session2pair(df_session):
    pair_dict = {}
    for idx, row in df_session.iterrows():
        user_id, item_id = row['user_id'], row['item_id']
        if row['user_id'] not in pair_dict:
            pair_dict[user_id] = [item_id]
        else:
            pair_dict[user_id].append(item_id)
    return pair_dict

def category2int(target):
    CATEGORY = {'etc':0, 'game':1, 'living':2, 'storytelling':3, 'arts':4, 'storytelling':5}
    if target in CATEGORY:
        return CATEGORY[target]
    else:
        return 0

class PNDataset(Dataset):
    def __init__(self, args):
        self.df_user = pd.read_pickle(os.path.join(args['root_path'], 'user.pkl'))
        self.df_item = pd.read_pickle(os.path.join(args['root_path'], 'item.pkl'))
        self.df_session = pd.read_pickle(os.path.join(args['root_path'], 'session.pkl'))
        self.neg_samples = args['posneg']['neg_samples']
        # self.pair_dict = session2pair(self.df_session)
        
    def __len__(self):
        return len(self.df_session) * (1 + self.neg_samples)

    
    def __getitem__(self, idx):
        # positive sample
        if idx < len(self.df_session):
            session = self.df_session.iloc[idx].to_dict()
            user_id, item_id = session['user_id'], session['item_id']
        
            label = torch.FloatTensor([1])
        
        else:
            # negative samples can be duplicated
            while True:
                user_idx = random.randint(0, len(self.df_user)-1)
                item_idx = random.randint(0, len(self.df_item)-1)
                user_id = self.df_user.iloc[user_idx]['id']
                item_id = self.df_item.iloc[item_idx]['id']
                
                if self.df_session[(self.df_session['user_id']==user_id)&(self.df_session['item_id']==item_id)].empty:
                    break
        
            label = torch.FloatTensor([0])
            
        user_info = self.df_user[self.df_user['id'] == user_id].to_dict('records')[0]
        item_info = self.df_item[self.df_item['id'] == item_id].to_dict('records')[0]
        
        
        # USER INFO {login_count, bookmark, follower, following, project, projectAll}
        user = torch.FloatTensor([user_info['login_count'], user_info['bookmark'], user_info['follower'], user_info['following'], user_info['project'], user_info['projectAll']])
        # ITEM INFO {category, comment_count, like_count, visit_count
        item = torch.FloatTensor([category2int(item_info['category']), item_info['comment_count'], item_info['like_count'], item_info['visit_count']])
        
        return (user, item), label


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
    elif data_type == 'posneg':
        dataset = PNDataset(args)
    else:
        raise NotImplementedError
    
    trainset, testset = dataset_split(dataset, ratio=args['split']['ratio'], seed=args['split']['seed'])
    
    return trainset, testset

def bulid_dataloader(dataset, args):
    return DataLoader(dataset, batch_size=args['batch_size'], shuffle=args['shuffle'])

def main():    
    df_session = pd.read_pickle(os.path.join(DATA_PATH, 'session.pkl'))
    start_time = time.time()
    pair_dict = session2pair(df_session)
    print(len(pair_dict), time.time()-start_time)
    
if __name__=='__main__':
    main()