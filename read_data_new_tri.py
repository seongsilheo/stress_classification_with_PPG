# -*- coding: utf-8 -*-
# +
import os
import pickle
import math
import numpy as np
import pandas as pd

from scipy.interpolate import UnivariateSpline
from preprocessing_tool.feature_extraction import *
# +
WINDOW_IN_SECONDS = 120  # 120 / 180 / 300

NOISE = ['bp_time_ens']
main_path='/home/sheo1/stress_classification_with_PPG/WESAD/'

# +
# E4 (wrist) Sampling Frequencies

fs_dict = {'ACC': 32, 'BVP': 64, 'EDA': 4, 'TEMP': 4, 'label': 700, 'Resp': 700}

label_dict = {'baseline': 1, 'stress': 2, 'amusement': 0}
int_to_label = {1: 'baseline', 2: 'stress', 0: 'amusement'}
    
sec = 12
N = fs_dict['BVP']*sec  # one block : 10 sec
overlap = int(np.round(N * 0.02)) # overlapping length
overlap = overlap if overlap%2 ==0 else overlap+1


# -

class SubjectData:

    def __init__(self, main_path, subject_number):
        self.name = f'S{subject_number}'
        self.subject_keys = ['signal', 'label', 'subject']
        self.signal_keys = ['chest', 'wrist']
        self.chest_keys = ['ACC', 'ECG', 'EMG', 'EDA', 'Temp', 'Resp']
        self.wrist_keys = ['ACC', 'BVP', 'EDA', 'TEMP']
        with open(os.path.join(main_path, self.name) + '/' + self.name + '.pkl', 'rb') as file:
            self.data = pickle.load(file, encoding='latin1')
        self.labels = self.data['label']

    def get_wrist_data(self):
        data = self.data['signal']['wrist']
        data.update({'Resp': self.data['signal']['chest']['Resp']})
        return data

    def get_chest_data(self):
        return self.data['signal']['chest']

    def extract_features(self):  # only wrist
        results = \
            {
                key: get_statistics(self.get_wrist_data()[key].flatten(), self.labels, key)
                for key in self.wrist_keys
            }
        return results


def extract_ppg_data(e4_data_dict, labels, norm_type=None):
    # Dataframes for each sensor type
    df = pd.DataFrame(e4_data_dict['BVP'], columns=['BVP'])
    label_df = pd.DataFrame(labels, columns=['label'])
    

    # Adding indices for combination due to differing sampling frequencies
    df.index = [(1 / fs_dict['BVP']) * i for i in range(len(df))]
    label_df.index = [(1 / fs_dict['label']) * i for i in range(len(label_df))]

    # Change indices to datetime
    df.index = pd.to_datetime(df.index, unit='s')
    label_df.index = pd.to_datetime(label_df.index, unit='s')

    df = df.join(label_df, how='outer')
    
    df['label'] = df['label'].fillna(method='bfill')
    
    df.reset_index(drop=True, inplace=True)
    
    if norm_type is 'std':  # 시그널 자체를 normalization
        # std norm
        df['BVP'] = (df['BVP'] - df['BVP'].mean()) / df['BVP'].std()
    elif norm_type is 'minmax':
        # minmax norm
        df = (df - df.min()) / (df.max() - df.min())

    # Groupby
    df = df.dropna(axis=0) # nan인 행 제거
    
    return df


def seperate_data_by_label(df):
    
    grouped = df.groupby('label')
    baseline = grouped.get_group(1)
    stress = grouped.get_group(2)
    amusement = grouped.get_group(3)   
    
    return grouped, baseline, stress, amusement



def get_samples(data, label, ma_usage):
    global feat_names
    global WINDOW_IN_SECONDS

    samples = []

    window_len = fs_dict['BVP'] * WINDOW_IN_SECONDS  # 64*60 , sliding window: 0.25 sec (60*0.25 = 15)   
    sliding_window_len = int(fs_dict['BVP'] * WINDOW_IN_SECONDS * 0.25)
    
    winNum = 0
    
    i = 0
    while sliding_window_len * i <= len(data) - window_len:
        
         # 한 윈도우에 해당하는 모든 윈도우 담기,
        w = data[sliding_window_len * i: (sliding_window_len * i) + window_len]  
        # Calculate stats for window
        wstats = get_window_stats_27_features(ppg_seg=w['BVP'].tolist(), window_length = window_len, label=label, ensemble = ENSEMBLE, ma_usage=ma_usage)
        winNum += 1
        
        if wstats == []:
            i += 1
            continue;
        # Seperating sample and label
        x = pd.DataFrame(wstats, index = [i])
    
        samples.append(x)
        i += 1

    return pd.concat(samples)


def make_patient_data(subject_id,ma_usage):
    global savePath
    global WINDOW_IN_SECONDS
    
    temp_ths = [1.0,2.0,1.8,1.5] #temp_ths = [1.1,2.2,2.0,1.9] 
    clean_df = pd.read_csv('clean_signal_by_rate.csv',index_col=0)
    cycle = 15
    
    # Make subject data object for Sx
    subject = SubjectData(main_path=main_path', subject_number=subject_id)
    
    # Empatica E4 data
    e4_data_dict = subject.get_wrist_data()

    # norm type
    norm_type = 'std'

    df = extract_ppg_data(e4_data_dict, subject.labels, norm_type)
    df_BVP = df.BVP


    #여기서 signal preprocessing 
    bp_bvp = butter_bandpassfilter(df_BVP, 0.5, 10, fs_dict['BVP'], order=2) # 0.5, 5 -> 0.5,10
    
    if BP:
        df['BVP'] = bp_bvp
        
    if FREQ:
        signal_one_percent = int(len(df_BVP) * 0.01)
        print(signal_one_percent)
        cutoff = get_cutoff(df_BVP[:signal_one_percent], fs_dict['BVP'])
        freq_signal = compute_and_reconstruction_dft(df_BVP, fs_dict['BVP'], sec, overlap, cutoff)
        df['BVP'] = freq_signal

    if TIME:
        fwd = moving_average(bp_bvp, size=3)
        bwd = moving_average(bp_bvp[::-1], size=3)
        bp_bvp = np.mean(np.vstack((fwd,bwd[::-1])), axis=0)
        df['BVP'] = bp_bvp
        
        signal_01_percent = int(len(df_BVP) * 0.001)
        print(signal_01_percent, int(clean_df.loc[subject_id]['index']))
        clean_signal = df_BVP[int(clean_df.loc[subject_id]['index']):int(clean_df.loc[subject_id]['index'])+signal_01_percent]
        ths = statistic_threshold(clean_signal, fs_dict['BVP'], temp_ths)
        len_before, len_after, time_signal_index = eliminate_noise_in_time(df['BVP'].to_numpy(), fs_dict['BVP'], ths, cycle)
    
        df = df.iloc[time_signal_index,:]
        df = df.reset_index(drop=True)
        #plt.figure(figsize=(40,20))
        #plt.plot(df['BVP'][:2000], color = 'b', linewidth=2.5)
    
    
    grouped, baseline, stress, amusement = seperate_data_by_label(df)   
    
    
    
    baseline_samples = get_samples(baseline, 1, ma_usage)
    print("bas: ",len(baseline_samples))
    stress_samples = get_samples(stress, 2, ma_usage)
    print("st: ",len(stress_samples))
    amusement_samples = get_samples(amusement, 0, ma_usage)
    print("Am: ",len(amusement_samples))
    window_len = len(baseline_samples)+len(stress_samples)+len(amusement_samples)

    all_samples = pd.concat([baseline_samples, stress_samples, amusement_samples])
    all_samples = pd.concat([all_samples.drop('label', axis=1), pd.get_dummies(all_samples['label'])], axis=1) # get dummies로 원핫벡터로 라벨값 나타냄
    
    
    all_samples.to_csv(f'{savePath}{subject_feature_path}/S{subject_id}_feats_4.csv')

    # Does this save any space?
    subject = None
    
    return window_len



def combine_files(subjects):
    df_list = []
    for s in subjects:
        df = pd.read_csv(f'{savePath}{subject_feature_path}/S{s}_feats_4.csv', index_col=0)
        df['subject'] = s
        df_list.append(df)

    df = pd.concat(df_list)

    df['label'] = (df['0'].astype(str) + df['1'].astype(str) + df['2'].astype(str)).apply(lambda x: x.index('1'))  # 1인 부분의 인덱스 반환
    df.drop(['0', '1', '2'], axis=1, inplace=True)

    df.reset_index(drop=True, inplace=True)

    df.to_csv(savePath + merged_path)

    counts = df['label'].value_counts()
    print('Number of samples per class:')
    for label, number in zip(counts.index, counts.values):
        print(f'{int_to_label[label]}: {number}')


# +
total_window_len = 0
BP, FREQ, TIME, ENSEMBLE = False, False, False, False
subject_ids = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]

feat_names = None
savePath = '27_features_ppg_test_3/LMM/'

if not os.path.exists(savePath):
    os.makedirs(savePath)


for n in NOISE:
    if 'bp' in n.split('_'):
        BP = True
    if 'time' in n.split('_'):
        TIME = True
    if 'ens' in n.split('_'):
        ENSEMBLE = True
    

    subject_feature_path = '/subject_feature_' + n + str(WINDOW_IN_SECONDS)
    merged_path = '/data_merged_' + n + str(WINDOW_IN_SECONDS) +'.csv'
    
    if not os.path.exists(savePath + subject_feature_path):
        os.makedirs(savePath + subject_feature_path)
    
        
    for patient in subject_ids:
        print(f'Processing data for S{patient}...')
        window_len = make_patient_data(patient, BP)
        total_window_len += window_len

    combine_files(subject_ids)
    print('total_Window_len: ',total_window_len)
    print('Processing complete.', n)
    total_window_len = 0

