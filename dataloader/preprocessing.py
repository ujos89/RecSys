import os
import json
import time
import pandas as pd
import datetime
import csv

from pprint import pprint

DATA_PATH  = '/data3/data/naver/naver_connect_v2'
SAVE_PATH  = '/home/zealot/zealot/RecSys/data/preprocessed'

def user_preprocessing(line_data):
    if 'status' not in line_data:
        return False
    if not ('bookmark' and 'follower' and 'following' and 'project' and 'projectAll') in line_data['status']:
        return False
    
    # data = {}
    # data['id'] = line_data['_id']['$oid']
    # data['login_count'] = line_data['loginCount']
    # data['login_last'] = datetime.datetime.strptime(line_data['lastLogin']['$date'][:16], "%Y-%m-%dT%H:%M")
    # data['bookmark'] = line_data['status']['bookmark']['project']
    # data['follower'] = line_data['status']['follower']
    # data['following'] = line_data['status']['following']
    # data['project'] = line_data['status']['project']
    # data['projectAll'] = line_data['status']['projectAll']

    data = []
    data.append(line_data['_id']['$oid'])
    data.append(line_data['loginCount'])
    data.append(datetime.datetime.strptime(line_data['lastLogin']['$date'][:16], "%Y-%m-%dT%H:%M"))
    data.append(line_data['status']['bookmark']['project'])
    data.append(line_data['status']['follower'])
    data.append(line_data['status']['following'])
    data.append(line_data['status']['project'])
    data.append(line_data['status']['projectAll'])
    
    return data

def item_preprocessing(line_data):
    if 'likeCnt' not in line_data:
        return False
    elif line_data['likeCnt'] < 1:
        return False
            
    
    if not ('_id' and 'categoryCode' and 'comment' and 'likeCnt' and 'visit' and 'user' and 'created' and 'updated') in line_data:
        return False

    data = []
    data.append(line_data['_id']['$oid'])
    data.append(line_data['categoryCode'])
    data.append(line_data['comment'])
    data.append(line_data['likeCnt'])
    data.append(line_data['visit'])
    data.append(line_data['user']['$oid'])
    data.append(datetime.datetime.strptime(line_data['created']['$date'][:16], "%Y-%m-%dT%H:%M"))
    data.append(datetime.datetime.strptime(line_data['updated']['$date'][:16], "%Y-%m-%dT%H:%M"))
    
    return data

def preprocessing(name, extension='_202205.json'):
    file_name = os.path.join(DATA_PATH, name+extension)
    
    if name == 'users':
        USER_COL = ['id', 'login_count', 'login_last', 'bookmark', 'follower', 'following', 'project', 'projectAll']
        
        user_idx = 0
        start_time = time.time()
        user_ids = []
        user_dict = {}
                
        with open(file_name, 'r') as f:
            f_save = open(os.path.join(SAVE_PATH, 'user.csv'), 'w', newline='')
            writer = csv.writer(f_save)
            writer.writerow(USER_COL)
            
            for line_idx, line in enumerate(f):
                line_data = json.loads(line)    
                
                # if line_data['role'] != 'member':
                if line_data['role'] != 'member' or line_data['loginCount'] < 3:
                    pass
                else:
                    data_preprocessed = user_preprocessing(line_data)
                    if data_preprocessed:
                        writer.writerow(data_preprocessed)
                        user_idx += 1
                
                if line_idx%10000==0:
                    print(line_idx, '->', user_idx)
        f_save.close()                        
        
        print(name, 'preprocessing time:\t', round(time.time()-start_time, 3))  # ~52s
        print(name, 'preproceeing data:\t', line_idx+1, '->', user_idx+1)       # 5212303 -> 1826516
        # df_user = df_user.reset_index(drop=True)
        # df_user.to_pickle(os.path.join(SAVE_PATH, 'user.pkl'))
    
    elif name == 'projects':
        PROJ_COL = ['id', 'category', 'comment_count', 'like_count', 'visit_count', 'user_id', 'create_date', 'update_date']
        
        start_time = time.time()
        data_preprocessed = {}        
        start_time = time.time()
        
        with open(file_name, 'r') as f:
            for line_idx, line in enumerate(f):
                line_data = json.loads(line)
            
                
                data_preprocessed = item_preprocessing(line_data)
                if data_preprocessed:
                    pprint(data_preprocessed)
                
                if line_idx>100:
                    break
                
        print(name, 'preprocessing:\t', round(time.time()-start_time, 3))
        print(name, 'line counts:\t', line_idx+1)
        
        # ################
        # projects preprocessing:  186.794
        # projects line counts:    21479342
        # projects target counts:  906065 (410107, likeCnt>=2) 
        # projects not include likeCnt:    3
        # ################
        
    elif name == 'likes':
        start_time = time.time()
        target_subjects = []
        wrong_cnt = 0 
        
        data_preprocessed = {}
        
        with open(file_name, 'r') as f:
            for line_idx, line in enumerate(f):
                line_data = json.loads(line)
                
                if 'targetSubject' not in line_data:
                    wrong_cnt += 1
                else: 
                    if line_data['targetSubject'] not in target_subjects:
                        target_subjects.append(line_data['targetSubject'])
                
        print(name, 'preprocessing:\t', round(time.time()-start_time, 3))
        print(name, 'line counts:\t', line_idx+1)
        print(name, 'target subjects:\t', target_subjects)
        print(name, 'not include target subjects:\t', wrong_cnt)
        
        # ################
        # likes preprocessing:     19.555
        # likes line counts:       6979858
        # likes target subjects:   ['project', 'comment', 'discuss', 'lecture', 'discovery', 'curriculum', 'EntryStory', '글삭기관총']
        # likes not include target subjects:       5329
        # ################
        
    elif name == 'favorites':
        start_time = time.time()
        target_subjects = []
        wrong_cnt = 0         
        data_preprocessed = {}
        
        with open(file_name, 'r') as f:
            for line_idx, line in enumerate(f):
                line_data = json.loads(line)
                
                if 'targetSubject' not in line_data:
                    wrong_cnt += 1
                else: 
                    pprint(line_data)
                    if line_data['targetSubject'] not in target_subjects:
                        target_subjects.append(line_data['targetSubject'])
                        
                        
                if line_idx > 5:
                    break
                
                
        print(name, 'preprocessing:\t', round(time.time()-start_time, 3))
        print(name, 'line counts:\t', line_idx+1)
        print(name, 'target subjects:\t', target_subjects)
        print(name, 'not include target subjects:\t', wrong_cnt)
        # ################
        # favorites preprocessing:         8.293
        # favorites line counts:   3069707
        # favorites target subjects:       ['project', 'lecture', 'curriculum']
        # favorites not include target subjects:   75
        # ################

def main():
    # preprocessing('users')
    preprocessing('projects')
    # preprocessing('likes')

if __name__=='__main__':
    main()