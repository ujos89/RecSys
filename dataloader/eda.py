import os
import pandas as pd
import json

DATA_PATH  = '/data3/data/naver/naver_connect_v2'

def read_jsonl(name, extension='_202205.json'):
    file_name = os.path.join(DATA_PATH, name+extension)
    
    # users
    cnt = 0
    target_cnt = 0
    with open(file_name, 'r') as f:
        for line in f:
            line_data = json.loads(line)
    
            
            if 'status' in line_data:
                target_cnt += 1
            
            cnt +=1 

    print(cnt)
    print(target_cnt)

def main():
    # file_xlsx = os.path.join(DATA_PATH, 'NAVER_Connect_Entry_Dataset.List.v2.0.0.xlsx')
    # df_xlsx = pd.read_excel(file_xlsx, engine='openpyxl')
    
    read_jsonl('users')

if __name__=='__main__':
    main()