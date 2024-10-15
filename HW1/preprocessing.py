import os
import re
import numpy as np
import pandas as pd
import kagglehub
import shutil
from sklearn.preprocessing import StandardScaler

def download(data = 1):
    # Download latest version
    if data == 1:
      return kagglehub.dataset_download("ramjasmaurya/top-1000-social-media-channels")
    elif data == 2:
      return kagglehub.dataset_download("yasserh/wine-quality-dataset")
    else:
      return 1
   
def move_data(command, path):
    source_directory = os.path.join(path)
    current_directory = os.getcwd()
    shutil.move(source_directory, current_directory)
    
    if command == 1:
        source_directory = os.path.join(os.getcwd(), '13')
        
        # 定義正則表達式模式
        pattern = re.compile(r'.*youtube.*', re.IGNORECASE)

        # 移動符合條件的檔案到目標資料夾
        for filename in os.listdir(source_directory):
            if pattern.match(filename) and filename != 'social media influencers - youtube.csv':
                source_file = os.path.join(source_directory, filename)
                destination_file = os.path.join(current_directory, filename)
                shutil.move(source_file, destination_file)
        
        shutil.rmtree(source_directory)
    elif command == 2:
        source_directory = os.path.join(os.getcwd(), '1')
        
        # 定義正則表達式模式
        # pattern = re.compile(r'.*youtube.*', re.IGNORECASE)

        # 移動符合條件的檔案到目標資料夾
        for filename in os.listdir(source_directory):
            # if pattern.match(filename) and filename != 'social media influencers - youtube.csv':
                source_file = os.path.join(source_directory, filename)
                destination_file = os.path.join(current_directory, filename)
                shutil.move(source_file, destination_file)
        
        shutil.rmtree(source_directory)
    else:
        print("Please enter the correct number")

def convert_to_numeric(value):
    if isinstance(value, str):
        if 'M' in value:  # 百萬單位
            return float(value.replace('M', '')) * 1e6
        elif 'K' in value:  # 千位單位
            return float(value.replace('K', '')) * 1e3
        elif value.lower() in ['n/a', 'na', '']:
            return np.nan  # 缺失值轉為 NaN
        else:
            try:
                return float(value)
            except ValueError:
                return np.nan
    return value

def file_name_setting(command):
    if command == 1:
        # 定義正則表達式模式
        pattern_jun = re.compile(r'social media influencers-youtube june 2022 - june 2022\.csv', re.IGNORECASE)
        pattern_sep = re.compile(r'social media influencers - Youtube sep-2022\.csv', re.IGNORECASE)
        pattern_nov = re.compile(r'social media influencers-youtube - --nov 2022\.csv', re.IGNORECASE)
        pattern_dec = re.compile(r'social media influencers-YOUTUBE - --DEC 2022\.csv', re.IGNORECASE)

        # 讀取並修正檔案名稱
        for filename in os.listdir('.'):
            if pattern_jun.match(filename):
                new_name = '2022-jun-youtube.csv'
                os.rename(filename, new_name)
            elif pattern_sep.match(filename):
                new_name = '2022-sep-youtube.csv'
                os.rename(filename, new_name)
            elif pattern_nov.match(filename):
                new_name = '2022-nov-youtube.csv'
                os.rename(filename, new_name)
            elif pattern_dec.match(filename):
                new_name = '2022-dec-youtube.csv'
                os.rename(filename, new_name)
    elif command == 2:
        new_name = 'original_WineQT.csv'
        try:
            os.rename('WineQT.csv', new_name)
        except:
            os.remove('WineQT.csv')
    else:
        print("Please enter the correct number")

def readfile(data_set):
    if data_set == 'media influencers':
        jun = pd.read_csv('2022-jun-youtube.csv')
        sep = pd.read_csv('2022-sep-youtube.csv')
        nov = pd.read_csv('2022-nov-youtube.csv')
        dec = pd.read_csv('2022-dec-youtube.csv')
        return jun, sep, nov, dec
    elif data_set == 'Wine Quality':
        WineQT = pd.read_csv('original_WineQT.csv')
        return WineQT
    else:
        print("Please enter the correct number")
    
def removefile(command):
    if command == 1:
        os.remove('2022-jun-youtube.csv')
        os.remove('2022-sep-youtube.csv')
        os.remove('2022-nov-youtube.csv')
        os.remove('2022-dec-youtube.csv')
        os.remove('2022-jun-youtube-cleaned.csv')
        os.remove('2022-sep-youtube-cleaned.csv')
        os.remove('2022-nov-youtube-cleaned.csv')
        os.remove('2022-dec-youtube-cleaned.csv')
    elif command == 2:
        a = 0
    else:
        print("Please enter the correct number")
    
def interface():
    command = -1
    getCommand = False
    while not getCommand:
        print("Please enter the number of the data you want to preprocessing:")
        print("0. Exit")
        print("1. Social Media Influencers in 2022 (Eliminate data with missing values)")
        print("2. Wine Quality Dataset (Standardization)")
        
        try:
            command = int(input('Enter the number: '))
        except:
            print("Please enter the correct number.")
            continue
        
        if command == 0 or command == 1 or command == 2:
            getCommand = True
        else:
            print("Please enter the correct number.")
            
    return command

def Eliminate_missing_values():
    jun, sep, nov, dec = readfile('media influencers') # 讀取檔案
    
    # channel_name, youtuber_name, category, category2, followers, country, view_AVG, like_AVG, comment_AVG
    sep = sep.drop(columns=['S.no'])
    nov = nov.drop(columns=['s.no'])
    dec = dec.drop(columns=['s.no'])
    jun.columns = ['channel_name', 'youtuber_name', 'category', 'category2', 'followers', 'country', 'view_AVG', 'like_AVG', 'comment_AVG']
    sep.columns = ['youtuber_name', 'channel_name', 'country', 'followers', 'category', 'view_AVG', 'like_AVG', 'comment_AVG', 'category2']
    nov.columns = ['channel_name', 'youtuber_name', 'category', 'followers', 'country', 'view_AVG', 'like_AVG', 'comment_AVG', 'category2']
    dec.columns = ['channel_name', 'youtuber_name', 'category', 'followers', 'country', 'view_AVG', 'like_AVG', 'comment_AVG', 'category2']

    # 新增欄位month
    jun['month'] = 'June'
    sep['month'] = 'September'
    nov['month'] = 'November'
    dec['month'] = 'December'

    jun.to_csv('2022-jun-youtube-cleaned.csv', index=False)
    sep.to_csv('2022-sep-youtube-cleaned.csv', index=False)
    nov.to_csv('2022-nov-youtube-cleaned.csv', index=False)
    dec.to_csv('2022-dec-youtube-cleaned.csv', index=False)

    all_data = pd.concat([jun, sep, nov, dec], ignore_index=True)

    all_data.to_csv('original_all_data.csv', index=False)

    all_data = all_data.dropna()

    # 針對相關欄位進行轉換
    all_data['followers'] = all_data['followers'].apply(convert_to_numeric)
    all_data['view_AVG'] = all_data['view_AVG'].apply(convert_to_numeric)
    all_data['like_AVG'] = all_data['like_AVG'].apply(convert_to_numeric)
    all_data['comment_AVG'] = all_data['comment_AVG'].apply(convert_to_numeric)

    all_data = all_data[all_data['comment_AVG'] != 'N/A\'']
    all_data = all_data[all_data['country'] != '-']

    all_data.to_csv('eliminate_all_data.csv', index=False)
    
    removefile(1) # 刪除檔案

def Standardization_data():
    WineQT = readfile('Wine Quality') # 讀取檔案
    
    features = WineQT.drop(columns=['Id', 'quality'])
    target = WineQT['quality']
    
    # 對特徵進行標準化處理
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # 將標準化後的資料轉為 DataFrame 以方便檢視
    features_scaled_df = pd.DataFrame(features_scaled, columns=features.columns)

    # 可選：將標準化的資料與原本的目標變數合併
    final_df = features_scaled_df.copy()
    final_df['quality'] = target

    final_df.to_csv('standardized_WineQT.csv', index=False)

if __name__ == '__main__':    
    finish = False
    while not finish:
        command = interface()           # 使用者介面
        if command == 0:
            print('Thank you for using.')
            finish = True
        else:
            path = download(command)        # 下載kaggle資料
            move_data(command, path)                 # 移動資料到當前資料夾
            file_name_setting(command)             # 修正檔案名稱
            if command == 1:
                Eliminate_missing_values()
            elif command == 2:
                Standardization_data()
            else:
                print("Please enter the correct number")
                finish = False