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
    if not ('bookmark' or 'follower' or 'following' or 'project' or 'projectAll') in line_data['status']:
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
            
    if not ('_id' or 'categoryCode' or 'comment' or 'likeCnt' or 'visit' or 'user' or 'created' or 'updated') in line_data:
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

def interaction_preprocessing(line_data):
    if 'targetSubject' not in line_data:
        return False
    if line_data['targetSubject'] != 'project':
        return False
    if not ('target' or 'user' or 'created') in line_data:
        return False
    
    if not '$oid' in line_data['user']:
        return False
    if not '$oid' in line_data['target']:
        return False
    if not '$date' in line_data['created']:
        return False
        
    
    data = []
    data.append(line_data['user']['$oid'])
    data.append(line_data['target']['$oid'])
    data.append(datetime.datetime.strptime(line_data['created']['$date'][:16], "%Y-%m-%dT%H:%M"))
    
    return data
    

def blocktype2info(block_type, input_data):
    info = []
    
    for block in block_type:
        if block in input_data:
            info.append(1)
        else:
            info.append(0)
    
    return info

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
        item_cnt = 0
        
        with open(file_name, 'r') as f:
            f_save = open(os.path.join(SAVE_PATH, 'item.csv'), 'w', newline='')
            writer = csv.writer(f_save)
            writer.writerow(PROJ_COL)
            for line_idx, line in enumerate(f):
                line_data = json.loads(line)
            
                if 'categoryCode' not in line_data:
                    pass
                else:
                    data_preprocessed = item_preprocessing(line_data)
                    if data_preprocessed:
                        writer.writerow(data_preprocessed)
                        item_cnt += 1
                        
                if line_idx%10000==0:
                    print(line_idx, '->', item_cnt)
        f_save.close()
        
        print(name, 'preprocessing:\t', round(time.time()-start_time, 3))       #
        print(name, 'preproceeing data:\t', line_idx+1, '->', item_cnt+1)       # 
        
        # ################
        # projects preprocessing:  186.794
        # projects line counts:    21479342
        # projects target counts:  906065 (410107, likeCnt>=2) 
        # projects not include likeCnt:    3
        # ################
        
    elif name == 'likes':
        INTER_COL = ['user_id', 'item_id', 'time']
        
        start_time = time.time()
        inter_cnt = 0 
        
        with open(file_name, 'r') as f:
            f_save = open(os.path.join(SAVE_PATH, 'interaction.csv'), 'w', newline='')
            writer = csv.writer(f_save)
            writer.writerow(INTER_COL)
            
            for line_idx, line in enumerate(f):
                line_data = json.loads(line)
                
                data_preprocessed = interaction_preprocessing(line_data)
                if not data_preprocessed:
                    pass
                else:
                    writer.writerow(data_preprocessed)
                    inter_cnt += 1
                
                if line_idx%10000==0:
                    print(line_idx, '->', inter_cnt)
        f_save.close()
                
        print(name, 'preprocessing:\t', round(time.time()-start_time, 3))
        print(name, 'line counts:\t', line_idx+1, '->', inter_cnt+1)
        
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
        
    elif name == 'likes':
        INTER_COL = ['user_id', 'item_id', 'time']
        
        start_time = time.time()
        inter_cnt = 0 
        
        with open(file_name, 'r') as f:
            f_save = open(os.path.join(SAVE_PATH, 'interaction.csv'), 'w', newline='')
            writer = csv.writer(f_save)
            writer.writerow(INTER_COL)
            
            for line_idx, line in enumerate(f):
                line_data = json.loads(line)
                
                data_preprocessed = interaction_preprocessing(line_data)
                if not data_preprocessed:
                    pass
                else:
                    writer.writerow(data_preprocessed)
                    inter_cnt += 1
                
                if line_idx%10000==0:
                    print(line_idx, '->', inter_cnt)
        f_save.close()
        
    elif name == 'logprojectblocks':
        start_time = time.time()
        block_type = set()
        
        block_type = sorted(['hamster_buzzer', 'RichShield_DHT', 'jikko_makeBuzzer', 'trueRobot_control', 'microbitExtSound', 'byrobot_dronefighter_controller_boolean_input', 'RGBLED_Set', 'uoalbert_eye', 'cheese_step_motor', 'jikko_basicPin', 'MechanicblockControllerGet', 'mp3', 'dplay_set', 'byrobot_dronefighter_controller_buzzer', 'hamster_s_sensor', 'byrobot_petrone_v2_flight_control_flight', 'Objects', 'CODEino_ultrasonic', 'Camera', 'MicrobitLed', 'cheese_sensor', 'WriteBlock', 'hamster_s_led', 'RuncodingGet', 'PingpongG1', 'roduino_value', 'remote', 'kamibot_sensor', 'Ozobot_Line_trace', 'jikko_makeLed', 'altino_lite_expert', 'MindpiggyNomalBlock', 'robotis_servo_motor', 'hamster_port', 'hummingbird_motor', 'freearduino_basic', 'brush_clear', 'albertai_sensor', 'cheese_neopixel', 'boolean_collision', '0lecoboard_led', 'playcode_set', 'kamibot_mapboard', 'Debug', 'Ozobot_LED', 'smartBoard_servo_motor', 'DavinciLed', 'number', 'sound_wait', 'Ozobot_Sound', 'byrobot_petrone_v2_flight_buzzer', 'robotis_motor', 'ssboard_nano_LV4', 'cp_moving', 'CODEino_custom_buzzer_mode', 'decision', 'dplay_get', 'site', 'calc_date', 'kamibot_control_stop', 'altino_expert', 'LineCoding_LINE1', 'Materials', 'calc_distance', 'xbot_motor', 'sound_stop', 'Digital_Get', 'altino_output', 'microbitExtMove', 'save_delete', 'ThinkBoard_RGB', 'setting', 'Rendering', 'neobot_purple_servo', 'Set', 'byrobot_dronefighter_controller_controller_light', 'calc_timer', 'mkboardMatrix', 'walk', 'smartBoard_button', 'PingpongG1_motor', 'ask', 'bingles_rgb', 'byrobot_petrone_v2_drive_drone_light', 'COM', 'CodingBoxArduinoWrite', 'chocopi', 'ssboard_nano_LV3', 'byrobot_petrone_v2_controller_controller_light', 'CodeWiz_Servo', 'default', 'CodingBoxWrite', 'dadublockget', 'neobot_purple_output', 'hamsterlite_sensor', 'brush_control', 'practical_course_sound', 'hamster_s_port', 'LecoboardAnalogRead', 'neobot_think_car_servo', 'JDCode_Sensor', 'problock', 'DavinciMusic', 'jikko_makeGet', 'hamsterlite_wheel', 'microbit2liteSensor', 'Dash_light', 'neobot_purple_melody', 'Krypton0_sensor', 'CODEino_Adumode', 'CodeWiz_buzzer', 'hamsterlite_board', 'codestar_input_sensor', 'kamibot_control', 'byrobot_dronefighter_flight_monitor', 'littlebits', 'speaker', 'microphone', 'p_types_method', 'byrobot_petrone_v2_drive_control_drive', 'delay', 'sound_play', 'flip', 'infrared', 'web', 'albert_buzzer', 'set_motor', 'file', 'jikko_basicBuzzer', 'lecoboardAvr_servo', 'scale', '0lecoboardAvr_led', 'Mechatronics_4D', 'kamibot_topmotor_control', 'block', 'plrun', 'MechanicblockUnoS', 'wait', 'xbot_sensor', 'microbit2Servo', 'RichShield_set', 'CodeWiz_DIGITAL_OUTPUT', 'toast', 'geni_output', 'dial', 'RichShield_OLED', 'lecoboard_servo', 'OrangeGet', 'brush_color', 'button', 'jikkoSet', 'choco_command', 'neobot_note', 'GBotLED', 'RichShield_LCD', 'codestar_output_sensor', 'palmkit_input', 'aibot', 'littlebits_set', 'robotis_carCont_cm', 'arduino_ori', 'hasseamGet', 'uoalbert_sound', 'smartBoard_sensor', 'Dash_senor', 'robotis_temperature', 'buttonclick', 'brush_opacity', 'computerinfo', 'LCD', 'microbitExtDigital', 'brown_led', 'Mechatro_d_out', 'list_visibility', 'jikko_basicGet', 'ThinkBoard_LASER', 'ai_learning', 'weather_legacy', 'hamster_s_sound', 'Dash_drive', 'byrobot_petrone_v2_controller_buzzer', 'chocopi_motion', 'CodingBoxRead', 'practical_course_touch', 'turtle_sensor', 'MicrobitButton', 'other', 'ThinkBoard_BUZ', 'GetBlock', 'EduMaker', 'alertpromptconfirm', 'camera', 'dot', 'practical_course_light', 'neobot_think_car_operation', 'microbit2liteSound', 'byrobot_petrone_v2_drive_controller_light', 'elio', 'boolean_input', 'variable_visibility', 'geni_input', 'microbitExtLed', 'JDKit_Sensor', 'albert_wheel', 'palmkit_buzzer', 'dadublock_car_set', 'byrobot_dronefighter_flight_buzzer', 'PingpongG3_Music', 'coconut_buzzer', 'byrobot_dronefighter_controller_monitor', 'albertai_sound', 'sound_volume', 'aibot_remote', 'PingpongG4', 'Runcoding', 'move_rotate', 'hamster_sensor', 'ThinkBoard_TMP', 'MicrobitSensor', 'robotis_userbutton', 'robotis_openCM70_custom', 'CodingBoxArduinoRead', 'memakerGet', 'byrobot_dronefighter_flight_boolean_input', 'dc_motor', 'joysticksensor', 'practical_course_irs', 'behaviorConductDisaster', 'cobl', 'pmsensor', 'microbit2Sound', 'Digital_DHT_Get', 'turtle_led', 'PingpongG2_peripheral_LED', 'cheese_serial', 'lecoboard_buzzer', 'Buzzer', 'cheese_pid', 'Kingcoding', 'coconut_sensor', 'robotis_melody', 'byrobot_petrone_v2_drive_buzzer', 'MicrobitDigital', 'hamster_led', 'ev3_sensor', 'arduino_set', 'iboard', 'sensorBoard', 'Cue_Sound', 'microbit2v2', 'byrobot_petrone_v2_flight_controller_display', 'ebs', 'neobot_purple_motor', 'RichShield_rgbled', 'calc_string', 'neo', 'CodeWiz_HuskyLens', 'robotis_irs', 'ssboard_nano_LV1', 'clone', 'lecoboardble', 'brush_thickness', 'byrobot_dronefighter_controller_vibrator', 'led', 'roduino_set', 'LineCoding_LINE2', 'neobot_output', 'CODEino_sensor', 'color', 'uoalbert_sensor', 'CODEino_RGBLED_mode', 'byrobot_petrone_v2_drive_boolean_input', 'scene', 'neobot_purple_sensor', 'jikkoLed', 'DavinciSensor', 'palmkit_led', 'CODEino_custom_neopixel_mode', 'joystickGet', 'RobotamiCoding', 'robotis_touch', 'altino_sensor', 'neobot_purple_decision', 'Cue_senor', 'CODEino_default_neopixel_mode', 'ArduinoExt', 'output', 'CodeWiz_default_sensor', 'altino_lite_output', 'coconut_wheel', 'littlebits_value', 'altino_lite_sensor', 'CodeWiz_DotMatrix', 'MicrobitAnalog', 'SetBlock', 'robotori_sensor', 'jikkoGet', 'jikkoPin', 'NEOPIXEL_Set', 'Dancing', 'p_types_literal', 'testblock', 'NeoSpiderGet', 'cheese_servo_motor', 'CODEino_analogSensor', 'variable', 'PingpongG3', 'UglyBot_Command', 'ssboard_nano_LV2', 'JDCode_Command', 'gyroscope', 'RichShieldGet', 'console', 'playcode_get', 'calc_user', 'Dash_Sound', 'boolean_compare', 'chocopi_touch', 'Ultrasonic_Set', 'microbitExtServo', 'PingpongG3_motor', 'PingpongG2_Music', 'chocopi_control', 'neobot_servo', 'JDCode_CodeRC', 'control_position', 'Digital_Set', 'log', 'arduino_value', 'environment', 'move_absolute', 'lecoboardLcd', 'getBlock', 'practical_course_servo', 'hamster_wheel', 'message', 'Mechatronics_4D_Get', 'CODEino_extmode', 'SERVO_Set', 'ThinkBoard_ANA', 'trim', 'byrobot_petrone_v2_flight_drone_light', 'microbitExtSensor', 'p_type_structs', 'microbit2Sensor', 'CODEino_digitalSensor', 'robotis_color', 'neobot_think_car_melody', 'iboard_sensor', 'mkboard', 'Orange', 'monitor', 'microbitExtButton', 'control_flight', 'festival', 'MechatroGet', 'byrobot_petrone_v2_flight_motor', 'ThinkBoard_BTN', 'microbit2Radio', 'byrobot_petrone_v2_controller_vibrator', 'LineCoding_LINE3', 'controller_display', 'LineCoding_EASY', 'PingpongG1_Music', 'terminate', 'CODEino_default_buzzer_mode', 'event', 'microbitExtAnalog', 'drone_light', 'cheese_digital_output', 'practical_course_motor', 'hamster_s_wheel', 'buzzer', 'byrobot_petrone_v2_flight_vibrator', 'cheese_hat010', 'hamsterlite_led', 'LineCoding_LINE5', 'turtle_wheel', 'byrobot_petrone_v2_flight_boolean_input', 'CodeWiz_neopixel', 'Cue_drive', 'translate', 'byrobot_petrone_v2_flight_controller_light', 'stamp', 'video', 'funboardsetmatrix', 'vibrator', 'calc_duration', 'p_types', 'hummingbird_led', 'say', 'coconut_led', 'message_type', 'trueRobot_sensor', 'UglyBot_Sensor', 'PingpongG2', 'robotis_light', 'robotori_motor', 'calc', 'microbit2litePin', 'bingles_sensor', 'digital', 'microbit2Led', 'robotis_openCM70_cm', 'NeoSpider', 'xbot_rgb', 'byrobot_petrone_v2_controller_monitor', 'jikkoBuzzer', 'neobot_motor', 'cheese_input', 'RichShield_FND', 'albert_led', 'microbit2Pin', 'info', 'input', 'Ozobot_Sensor', 'MechatroStart', 'ThinkBoard_USONIC', 'practical_course_melody', 'byrobot_petrone_v2_flight_monitor', 'uoalbert_wheel', 'PingpongG4_motor', 'turtle_sound', 'albertai_eye', 'microbit2liteLed', 'posting', 'Dash_head', 'dht', 'cheese_sound', 'ext', 'aibot_aidesk', 'byrobot_dronefighter_flight_controller_light', 'funboardset', 'MicrobitNote', 'chocopi_output', 'z-index', 'audio', 'list', 'syntax', 'condition', 'hamster_s_board', 'rotate_absolute', 'boolean', 'effect', 'ArduinoExtGet', 'dadublockset', 'TONE_Set', 'EduMakerGet', 'generic', 'controller_light', 'message_author', 'jikko_makeModule', 'ultrasonic', 'hamsterlite_buzzer', 'SENSOR', 'neobot_purple_led', 'blacksmithSet', 'chocopi_sensor', 'CodeWiz_OLED', 'message_guild', 'lecoboardTest', 'RichShield_Set', 'albertai_wheel', 'JDKit_Command', 'copy', 'lang', 'byrobot_petrone_v2_controller_boolean_input', 'cheese_led', 'behaviorConductLifeSafety', 'lecoboardAvr_buzzer', 'sciencecubeBlock', 'blacksmithModule', 'roe_set', 'eduinoGet', 'rotate', 'client', 'MicrobitAccelerometer', 'Dash_animation', 'text', 'robotis_sound', 'byrobot_petrone_v2_drive_motor', 'move_relative', 'jikkoModule', 'motor', 'Moving', 'jikko_basicSet', 'variables', 'byrobot_dronefighter_flight_drone_light', 'albert_sensor', 'robotis_humidity', 'byrobot_petrone_v2_controller_controller_display', 'ev3_output', 'weather', 'jikko_basicModule', 'arduino', 'brown_wheel', 'project', 'ThinkBoard_IR', 'eduino', 'neopixel', 'Analog_Get', 'get', 'ssboard_nano_LV5', 'run', 'cheese_pwm_output', 'ThinkBoard_DC', 'setBlock', 'byrobot_petrone_v2_flight_irmessage', 'visibility', 'Krypton0_motor_control', 'bingles_motor', 'cheese_dc_motor', 'neobot_value', 'memaker', 'blacksmithGet', 'PingpongG2_motor', 'jikko_basicLed', 'CODEino_Setmode', 'sensor', 'init', 'hamster_board', 'set', 'Basic', 'Internet', 'shape', 'save', 'login', 'Ozobot_Movement', 'practical_course_diode', 'tts', 'repeat', 'byrobot_dronefighter_flight_control_flight', 'CLCD_Set', 'CodeWiz_Dc', 'control_quad', 'neobot_purple_remote', 'type_struct_literal', 'render', 'Ozobot_Power', 'analysis', 'display', 'PingpongG1_peripheral_LED', 'CODEino_servo', 'Kingcoding3', 'hasseamSet', 'sound', 'list_element', 'lecoboardRobotArm', 'funboardget'])
        block_type.insert(0, 'project_id')
        
        with open(file_name, 'r') as f:
            f_save = open(os.path.join(SAVE_PATH, 'logblocks.csv'), 'w', newline='')
            writer = csv.writer(f_save)
            writer.writerow(block_type)
            block_type.pop(0)
            
            for line_idx, line in enumerate(f):
                line_data = json.loads(line)
                
                if line_data['blocks']:
                    if 'categories' in line_data['blocks']:
                        project_id = line_data['project']['$oid']
                        info = blocktype2info(block_type, line_data['blocks']['categories'])

                        writer.writerow([project_id]+info)

        f_save.close()
                
        print(name, 'preprocessing:\t', round(time.time()-start_time, 3))
        print(name, 'line counts:\t', line_idx+1)

def prepare():
    df_item = pd.read_csv(os.path.join(SAVE_PATH, 'item.csv'))
    df_user = pd.read_csv(os.path.join(SAVE_PATH, 'user.csv'))
    df_session = pd.read_csv(os.path.join(SAVE_PATH, 'interaction.csv'))
    df_logblocks = pd.read_csv(os.path.join(SAVE_PATH, 'logblocks.csv'))
    
    # user_id = set(df_user['id'])
    # item_user_id = set(df_item['user_id'])
    
    # # delete item not in user
    # user_id_intersection = list(user_id.intersection(item_user_id))
    
    # item_prepared = df_item[df_item.user_id.isin(user_id_intersection)].reset_index(drop=True)
    # user_prepared = df_user[df_user.id.isin(user_id_intersection)].reset_index(drop=True)
    # session_prepared = df_session[df_session.user_id.isin(user_id_intersection)].reset_index(drop=True)
    
    # # delete session item_id not in item.csv
    # session_item_id = set(session_prepared['item_id'])
    # item_id = set(item_prepared['id'])
    # item_id_intersection = list(item_id.intersection(session_item_id))
    
    # item_prepared = item_prepared[item_prepared.id.isin(item_id_intersection)].reset_index(drop=True)
    # session_prepared = session_prepared[session_prepared.item_id.isin(item_id_intersection)].reset_index(drop=True)
    
    ##########
    user_id = set(df_user['id'])
    session_user_id = set(df_session['user_id'])
    session_item_id = set(df_session['item_id'])
    item_user_id = set(df_item['user_id'])
    item_item_id = set(df_item['id'])
    
    user_id_intersection = set.intersection(user_id, session_user_id, item_user_id)
    item_id_intersection = set.intersection(session_item_id, item_item_id)
    ##########
    
    user_prepared = df_user[df_user.id.isin(user_id_intersection)].reset_index(drop=True)
    item_prepared = df_item[df_item.id.isin(item_id_intersection) & df_item.user_id.isin(user_id_intersection)].reset_index(drop=True)
    session_prepared = df_session[df_session.user_id.isin(user_id_intersection.intersection(user_prepared['id'])) & df_session.item_id.isin(item_id_intersection.intersection(item_prepared['id']))].reset_index(drop=True)
    logblocks_prepare = df_logblocks[df_logblocks.project_id.isin(item_id_intersection)].reset_index(drop=True)
    
    print('user:', len(df_user), '->', len(user_prepared))
    print('item:', len(df_item), '->', len(item_prepared))
    print('session:', len(df_session), '->', len(session_prepared))
    print('logblocks:', len(df_logblocks), '->', len(logblocks_prepare))
    
    # save
    item_prepared.to_csv(os.path.join(SAVE_PATH, 'prepared', 'item.csv'), index=False)
    item_prepared.to_pickle(os.path.join(SAVE_PATH, 'prepared', 'item.pkl'))
    user_prepared.to_csv(os.path.join(SAVE_PATH, 'prepared', 'user.csv'), index=False)
    user_prepared.to_pickle(os.path.join(SAVE_PATH, 'prepared', 'user.pkl'))
    session_prepared.to_csv(os.path.join(SAVE_PATH, 'prepared', 'session.csv'), index=False)
    session_prepared.to_pickle(os.path.join(SAVE_PATH, 'prepared', 'session.pkl'))
    logblocks_prepare.to_csv(os.path.join(SAVE_PATH, 'prepared', 'logblocks.csv'), index=False)
    logblocks_prepare.to_pickle(os.path.join(SAVE_PATH, 'prepared', 'logblocks.pkl'))

def main():
    # preprocessing('users')
    # preprocessing('projects')
    # preprocessing('likes')
    # preprocessing('logprojectblocks')

    prepare()
    
if __name__=='__main__':
    main()