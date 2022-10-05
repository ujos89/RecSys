import os
import json
import time
import pprint

DATA_PATH  = '/data3/data/naver/naver_connect_v2'
SAVE_PATH  = '/data3/data/naver/preprocessing'


def preprocessing(name, extension='_202205.json'):
    file_name = os.path.join(DATA_PATH, name+extension)
    
    if name == 'users':
        line_cnt = 0
        target_cnt = 0
        role_cnt = [0,0,0,0] # admin, member, teacher, student
        
        start_time = time.time()
        with open(file_name, 'r') as f:
            for line in f:
                line_data = json.loads(line)
            
                ################################
                if '_id' not in line_data:
                    print(line_data)
                if 'role' not in line_data:
                    pprint.pprint(line_data)
                ###############################        
                # if line_data['loginCount'] >= 3 :
                #     target_cnt += 1
                #     if line_data['role'] == 'admin':
                #         role_cnt[0] += 1 
                #     elif line_data['role'] == 'member':
                #         role_cnt[1] += 1 
                #     elif line_data['role'] == 'teacher':
                #         role_cnt[2] += 1 
                #     elif line_data['role'] == 'student':
                #         role_cnt[3] += 1
                        
                if 'status' in line_data and line_data['role']=='member' and line_data['loginCount'] >= 3:
                    pprint.pprint(line_data)
                    print()
                    target_cnt += 1
                
                if target_cnt > 4:
                    break
                
                # if line_data['role'] == 'student':
                #     pprint.pprint(line_data)
                #     break
                
                # if line_data['username'] != line_data['nickname']:
                #     pprint.pprint(line_data)
                #     break
                
                
                # if 'studyCurriculum' in line_data:
                #     if len(line_data['studyCurriculum']) > 1:
                #         pprint.pprint(line_data)
                #         break
                    
                
                line_cnt += 1 
                
                
        print(name, 'preprocessing:\t', round(time.time()-start_time, 3))
        print(name, 'line counts:\t', line_cnt)
        print(target_cnt)
        print(role_cnt)
    
    elif name == 'favorites':
        line_cnt = 0
        wrong_cnt = 0
        group_cnt = 0
        indiv_cnt = 0
        target_cnt = [0] * 3 # projects, lecture, curriculum
        
        start_time = time.time()
        with open(file_name, 'r') as f:
            for line in f:
                line_data = json.loads(line)
                line_cnt += 1
                
                # if 'targetType' not in line_data:
                #     wrong_cnt += 1
                # else: 
                #     if line_data['targetType'] == 'individual':
                #         indiv_cnt += 1
                #     if line_data['targetType'] == 'group':
                #         group_cnt += 1
                
                if 'targetSubject' not in line_data:
                    wrong_cnt += 1
                    pprint.pprint(line_data)
                else:
                    if line_data['targetSubject'] == 'project':
                        target_cnt[0] += 1
                    elif line_data['targetSubject'] == 'lecture':
                        target_cnt[1] += 1
                    elif line_data['targetSubject'] == 'curriculum':
                        target_cnt[2] += 1
                        
        print(name, 'preprocessing:\t', round(time.time()-start_time, 3))
        print(name, 'line counts:\t', line_cnt)
        print(line_cnt)
        print(wrong_cnt)
        print(target_cnt)

def main():
    preprocessing('users')
    # preprocessing('favorites')

if __name__=='__main__':
    main()