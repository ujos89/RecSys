import os
import json
import time
import pandas as pd
import datetime

from pprint import pprint

DATA_PATH  = '/data3/data/naver/naver_connect_v2'
SAVE_PATH  = '/data3/data/naver/preprocessing'

def user_preprocessing(line_data):
    if 'status' not in line_data:
        return False
    if not ('bookmark' and 'follower' and 'following' and 'project' and 'projectAll') in line_data['status']:
        return False
    
    data = {}
    data['id'] = line_data['_id']['$oid']
    data['login_count'] = line_data['loginCount']
    data['login_last'] = datetime.datetime.strptime(line_data['lastLogin']['$date'].split('.')[0], "%Y-%m-%dT%H:%M:%S")
    
    data['bookmark'] = line_data['status']['bookmark']['project']
    data['follower'] = line_data['status']['follower']
    data['following'] = line_data['status']['following']
    data['project'] = line_data['status']['project']
    data['projectAll'] = line_data['status']['projectAll']

    return data

def preprocessing(name, extension='_202205.json'):
    file_name = os.path.join(DATA_PATH, name+extension)
    
    if name == 'users':
        USER_COL = ['id', 'login_count', 'login_last', 'bookmark', 'follower', 'following', 'project', 'projectAll']
        
        df_user = pd.DataFrame(columns=USER_COL)
        
        target_cnt = 0
        start_time = time.time()
        user_ids = []
                
        with open(file_name, 'r') as f:
            for line_idx, line in enumerate(f):
                line_data = json.loads(line)    
                
                # if line_data['role'] != 'member':
                if line_data['role'] != 'member' or line_data['loginCount'] < 3:
                    pass
                else:
                    data_preprocessed = user_preprocessing(line_data)
                    
                    if data_preprocessed:
                        df_line = pd.DataFrame([data_preprocessed])
                        print(df_user)
                        print(df_line)
                    
                        break
                        
                    
                # if target_cnt > 50:
                #     break
                                    
        print(name, 'preprocessing time:\t', round(time.time()-start_time, 3))  # ~52s
        print(name, 'preproceeing data:\t', line_idx+1, '->', target_cnt)       # 5212303 -> 1826516
    
    elif name == 'projects':
        start_time = time.time()
        data_preprocessed = {}
        target_cnt = 0
        wrong_cnt = 0
        
        start_time = time.time()
        with open(file_name, 'r') as f:
            for line_idx, line in enumerate(f):
                line_data = json.loads(line)
            
                # if 10 <= line_idx <= 15:
                #     pprint(line_data)
                
                
                # if 'childCnt' in line_data:
                #     if 10 < line_data['childCnt'] < 1000 and not line_data['isForStudy']:
                #         pprint(line_data)
                #         break
                
                if 'likeCnt' not in line_data:
                    wrong_cnt += 1
                else:
                    if line_data['likeCnt'] >= 1:
                        target_cnt += 1
                
        print(name, 'preprocessing:\t', round(time.time()-start_time, 3))
        print(name, 'line counts:\t', line_idx+1)
        print(name, 'target counts:\t', target_cnt)
        print(name, 'not include likeCnt:\t', wrong_cnt)
        
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
    preprocessing('users')
    # preprocessing('projects')
    # preprocessing('likes')

if __name__=='__main__':
    main()