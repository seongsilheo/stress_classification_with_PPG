# -*- coding: utf-8 -*-
# +
import math
from scipy import signal
from scipy import fft, ifft
from scipy import fftpack
from scipy.stats import kurtosis, skew
from scipy.signal import butter, lfilter
from scipy import stats
from sklearn.linear_model import LinearRegression



import numpy as np

import matplotlib.pyplot as plt

# +
# filter


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs  
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpassfilter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y



def movingaverage(data, periods=4):
    result = []
    data_set = np.asarray(data)
    weights = np.ones(periods) / periods
    result = np.convolve(data_set, weights, mode='valid')
    return result

# -



def detrend_signals(signals):
    detrended_signals = []
    X = [k for k in range(0, len(signals))]
    X = np.reshape(X, (len(X), 1)) ## Reshapes the array into necessary format
    model = LinearRegression() ## Defines the model
    model.fit(X, signals) ## Fits signals to the model
    trend = model.predict(X) ## Predicts the model's trend
    detrend = [signals[k] - trend[k] for k in range(len(signals))] ## Removes trend from signal
    detrended_signals.append(detrend)
    
    return detrended_signals[0]


# +
'''
Method 1 ) local minima and maxima
'''

def threshold_peakdetection(dataset, fs):
    
    #print("dataset: ",dataset)
    window = []
    peaklist = []
    ybeat = []
    listpos = 0
    mean = np.average(dataset)
    TH_elapsed = np.ceil(0.36 * fs)
    npeaks = 0
    peakarray = []
    
    localaverage = np.average(dataset)
    for datapoint in dataset:

        if (datapoint < localaverage) and (len(window) < 1):
            listpos += 1
        elif (datapoint >= localaverage):
            window.append(datapoint)
            listpos += 1
        else:
            maximum = max(window)
            beatposition = listpos - len(window) + (window.index(max(window)))
            peaklist.append(beatposition)
            window = []
            listpos += 1

            
    ## Ignore if the previous peak was within 360 ms interval becasuse it is T-wave
    for val in peaklist:
        if npeaks > 0:
            prev_peak = peaklist[npeaks - 1]
            elapsed = val - prev_peak
            if elapsed > TH_elapsed:
                peakarray.append(val)
        else:
            peakarray.append(val)
            
        npeaks += 1    


    return peaklist


def RR_interval(peaklist,fs):
    RR_list = []
    cnt = 0
    while (cnt < (len(peaklist)-1)):
        RR_interval = (peaklist[cnt+1] - peaklist[cnt]) # Calculate distance between beats in # of samples
        ms_dist = ((RR_interval / fs) * 1000.0)  # Convert sample distances to ms distances (fs로 나눠서 1초단위로 거리표현 -> 1ms단위로 change) 
        RR_list.append(ms_dist)
        cnt += 1
        
    return RR_list


def calc_heartrate(RR_list):
    HR = []
    heartrate_array=[]
    window_size = 10

    for val in RR_list:
        if val > 400 and val < 1500:
            heart_rate = 60000.0 / val #60000 ms /1 minute. time per beat(한번 beat하는데 걸리는 시간)
            
        # if RR-interval < .1905 seconds, heart-rate > highest recorded value, 315 BPM. Probably an error!
        elif (val > 0 and val < 400) or val > 1500:
            if len(HR) > 0:
                # ... and use the mean heart-rate from the data so far:
                heart_rate = np.mean(HR[-window_size:])

            else:
                heart_rate = 60.0
        else:
            # Get around divide by 0 error
            heart_rate = 0.0

        HR.append(heart_rate)

    return HR


# +
'''
1. Frequency view

Reference:
Sadhukhan, Deboleena, Saurabh Pal, and Madhuchhanda Mitra. 
"PPG Noise Reduction based on Adaptive Frequency Suppression using Discrete Fourier Transform 
for Portable Home Monitoring Applications." 
2018 15th IEEE India Council International Conference (INDICON). IEEE, 2018.

'''

def FFT(block,fs):
    fourierTransform = np.fft.fft(block)/len(block)  # divide by len(block) to normalize
    fourierTransform = fourierTransform[range(int(len(block)/2))] # single side frequency / symmetric

    tpCount = len(block)
    values = np.arange(int(tpCount)/2)

    timePeriod = tpCount / fs
    frequencies = values/timePeriod # frequency components

    '''
    plt.figure(figsize=(40,20))
    plt.plot(frequencies, abs(fourierTransform)) 

    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.xlabel("frequency(Hz)",fontsize=50)
    plt.ylabel("FFT magnitude(mV)",fontsize=50)
    '''
    
    return frequencies, fourierTransform, timePeriod



#def get_cutoff(block,peak,fs):
def get_cutoff(block,fs):
    
    block = np.array([item.real for item in block])
    peak = threshold_peakdetection(block,fs)
    hr_mean = np.mean(calc_heartrate(RR_interval(peak,fs)))
    #low_cutoff = np.round(hr_mean / 60 - 0.2, 1)
    low_cutoff = np.round(hr_mean / 60 - 0.6, 1) # 0.6
    
    
    frequencies, fourierTransform, timePeriod = FFT(block,fs)
    ths = max(abs(fourierTransform)) * 0.10
    
    for i in range(int(5*timePeriod),0, -1):  # check from 5th harmonic
        if abs(fourierTransform[i]) > ths:
            high_cutoff = np.round(i/timePeriod, 1) 
            break
    
    print('low cutoff: ', low_cutoff, 'high_cutoff: ', high_cutoff)
    
    #low_loc = np.where(frequencies == low_cutoff)[0][0]
    #high_loc = np.where(frequencies == high_cutoff)[0][0]
            
    return [low_cutoff, high_cutoff]



def compute_and_reconstruction_dft(data, fs, sec, overlap, cutoff):
    concatenated_sig = []
    
    for i in range(0, len(data), fs*sec-overlap):
        seg_data = data[i:i+fs*sec]
        sig_fft = fftpack.fft(seg_data)
    
        
        #The corresponding frequencies
        sample_freq = (fftpack.fftfreq(len(seg_data)) * fs)

        new_freq_fft = sig_fft.copy()
        new_freq_fft[np.abs(sample_freq) < cutoff[0]] = 0
        new_freq_fft[np.abs(sample_freq) > cutoff[1]] = 0
    
        filtered_sig = fftpack.ifft(new_freq_fft)
        
        # 2% overlapping
        if i == 0:
            concatenated_sig = np.hstack([concatenated_sig, filtered_sig[:fs*sec - overlap//2]])
        elif i == len(data)-1:
            concatenated_sig = np.hstack(concatenated_sig, filtered_sig[overlap//2:])
        else:
            concatenated_sig = np.hstack([concatenated_sig, filtered_sig[overlap//2:fs*sec - overlap//2]])
        
    return concatenated_sig

# +
from sklearn.preprocessing import MinMaxScaler

def ppg_normalization(data, min_range, max_range):

    scaler = MinMaxScaler(feature_range=(min_range,max_range))
    scaler.fit(data)
    norm_data = scaler.transform(data)
    
    return norm_data


# +
'''

Reference:
Hanyu, Shao, and Chen Xiaohui. "Motion artifact detection and reduction in PPG signals based on statistics analysis." 
2017 29th Chinese control and decision conference (CCDC). IEEE, 2017.

'''

# cycle: per period


def statistic_threshold(clean_signal, fs, ths):
    stds, kurtosiss, skews, valley = statistic_detection(clean_signal, fs)
    std_ths = np.mean(stds) + ths[0] # paper's threshold : 3.2  
    kurt_ths = np.mean(kurtosiss) + ths[1]  #3.1  
    skews_ths = [np.mean(skews) - ths[2], np.mean(skews) + ths[3]]  # -0.3 and 0.8   -0.5 , 0.9
    
    return std_ths, kurt_ths, skews_ths


def statistic_detection(signal, fs):
    
    valley = pair_valley(valley_detection(signal, fs))
    stds=[]
    kurtosiss=[]
    skews=[]

    for val in valley: # 사이클 한 번동안의 통계적 평균 리스트 저장
        stds.append(np.std(signal[val[0]:val[1]]))
        kurtosiss.append(kurtosis(signal[val[0]:val[1]]))
        skews.append(skew(signal[val[0]:val[1]])) 

    return stds, kurtosiss, skews, valley

def eliminate_noise_in_time(data, fs, ths,cycle=1):
    stds, kurtosiss,skews, valley = statistic_detection(data, fs)
    
    
    #cycle 수만큼 다시 평균내서 리스트 저장
    stds_, kurtosiss_, skews_ = [], [], []
    stds_ = [np.mean(stds[i:i+cycle]) for i in range(0,len(stds)-cycle+1,cycle)]
    kurtosiss_ = [np.mean(kurtosiss[i:i+cycle]) for i in range(0,len(kurtosiss)-cycle+1,cycle)]
    skews_ = [np.mean(skews[i:i+cycle]) for i in range(0,len(skews)-cycle+1,cycle)]    
   
    # extract clean index, 사이클 인덱스       
    eli_std = [stds_.index(x) for x in stds_ if x < ths[0]]
    eli_kurt = [kurtosiss_.index(x) for x in kurtosiss_ if x < ths[1]]
    eli_skew = [skews_.index(x) for x in skews_ if x > ths[2][0] and x < ths[2][1]]

    total_list = eli_std + eli_kurt + eli_skew
    
    
    # store the number of extracted each index(각 인덱스 extract된 횟수 저장)
    dic = dict()
    for i in total_list:
        if i in dic.keys():
            dic[i] += 1
        else:
            dic[i] = 1
            
    new_list = []
    for key, value in dic.items():
        if value >= 3:
            new_list.append(key)
    new_list.sort()
    
    eliminated_data = []
    index = []
    for x in new_list:
        index.extend([x for x in range(valley[x*cycle][0],valley[x*cycle+cycle-1][1],1)])

    print(len(data), len(index))
    return len(data), len(index), index


# +
def valley_detection(dataset, fs):
    window = []
    valleylist = []
    ybeat = []
    listpos = 0
    TH_elapsed = np.ceil(0.36 * fs)
    nvalleys = 0
    valleyarray = []
    
    localaverage = np.average(dataset)
    for datapoint in dataset:

        if (datapoint > localaverage) and (len(window) < 1):
            listpos += 1
        elif (datapoint <= localaverage):
            window.append(datapoint)
            listpos += 1
        else:
            minimum = min(window)
            beatposition = listpos - len(window) + (window.index(min(window)))
            valleylist.append(beatposition)
            window = []
            listpos += 1

    ## Ignore if the previous peak was within 360 ms interval becasuse it is T-wave
    for val in valleylist:
        if nvalleys > 0:
            prev_valley = valleylist[nvalleys - 1]
            elapsed = val - prev_valley
            if elapsed > TH_elapsed:
                valleyarray.append(val)
        else:
            valleyarray.append(val)
            
        nvalleys += 1    

    return valleyarray


def pair_valley(valley):
    pair_valley=[]
    for i in range(len(valley)-1):
        pair_valley.append([valley[i], valley[i+1]])
    return pair_valley


# -

def auto_correlation(filtered_3sec):
    
    a = plt.acorr(filtered_3sec, usevlines=True, normed=True, maxlags=(len(filtered_3sec)-1), lw=2)
    valley_indexes = signal.argrelextrema(a[1], np.less)  # find valley index
    plt.scatter(x=valley_indexes[0]-(len(filtered_3sec)-1), y=a[1][valley_indexes[0]], color='red', s=20)
    
    diff = []
    for i in range(len(valley_indexes[0]) -1):
        diff.append(valley_indexes[0][i+1] - valley_indexes[0][i])
    
    return np.average(diff)


# +
# it doesn't work well when they are noisy signals, a and b gotta be same

'''
def improved_moving_window_method(data, fs, mean_period):
    
    # find first valley
    valley = valley_detection(data[:100],fs)
    final_valley = [valley[0]]
    
    i=0
    while True:
        if i == 0:
            meuw = mean_period
            a = math.ceil(valley[0] + 2/3*meuw)
            b = math.ceil(valley[0] + 4/3*meuw)
        else:
            meuw = final_valley[-1] - final_valley[-2]
            a = math.ceil(final_valley[-1] + 2/3*meuw)
            b = math.ceil(final_valley[-1] + 4/3*meuw)

        final_valley.append(a + np.argmin(data[a:b]))      
        i += 1   
                        
        if b > len(data):
            break
    
    return final_valley
# + {}
'''

