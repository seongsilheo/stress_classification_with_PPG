# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import scipy.signal

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

# +
'''
Method 2 ) first derivative with adaptive threshold

Reference:

1. Li, Bing Nan, Ming Chui Dong, and Mang I. Vai. "On an automatic delineator for 
arterial blood pressure waveforms." Biomedical Signal Processing and Control 5.1 (2010): 76-81.

2. Elgendi, Mohamed, et al. "Systolic peak detection in acceleration photoplethysmograms 
measured from emergency responders in tropical conditions." PLoS One 8.10 (2013).

'''

def seperate_division(data,fs):
    divisionSet = []
    for divisionUnit in range(0,len(data)-1,5*fs):  # index of groups (per 5sec) 
        eachDivision = data[divisionUnit: (divisionUnit+1) * 5 * fs]
        divisionSet.append(eachDivision)
    return divisionSet

def first_derivative_with_adaptive_ths(data, fs):
    
    peak = []
    divisionSet = seperate_division(data, fs)
    selectiveWindow = 2 * fs
    block_size = 5 * fs
    bef_idx = -300
    
    for divInd in range(len(divisionSet)):
        block = divisionSet[divInd]
        ths = np.mean(block[:selectiveWindow]) # ths: 2 seconds mean in block
        
        firstDeriv = block[1:] - block[:-1]
        for i in range(1,len(firstDeriv)):
            if  firstDeriv[i] <= 0 and firstDeriv[i-1] > 0:
                if block[i] > ths:
                    idx = block_size*divInd + i
                    if idx - bef_idx > (300*fs/1000):
                        peak.append(idx)
                        bef_idx = idx
                                                
    return peak
        



# +
'''
Method 3: Slope sum function with an adaptive threshold

Reference
1. Jang, Dae-Geun, et al. "A robust method for pulse peak determination 
in a digital volume pulse waveform with a wandering baseline." 
IEEE transactions on biomedical circuits and systems 8.5 (2014): 729-737.

2. Jang, Dae-Geun, et al. "A real-time pulse peak detection algorithm for 
the photoplethysmogram." International Journal of Electronics and Electrical Engineering 2.1 (2014): 45-49.
'''

def determine_peak_or_not(prevAmp, curAmp, nextAmp):
    if prevAmp < curAmp and curAmp >= nextAmp:
        return True
    else:
        return False
    
def onoff_set(peak, sig):     # move peak from dy signal to original signal   
    onoffset = []
    for p in peak:
        for i in range(p, 0,-1):
            if sig[i] == 0:
                onset = i
                break
        for j in range(p, len(sig)):
            if sig[j] == 0:
                offset = j
                break
        if onset < offset:
            onoffset.append([onset,offset])
    return onoffset
    

def slope_sum_function(data,fs):
    dy = [0]
    
    dy.extend(np.diff(data))
    #dy[dy < 0 ] = 0
    
    w = fs // 8
    dy_ = [0] * w
    for i in range(len(data)-w):
        sum_ = np.sum(dy[i:i+w])
        if sum_ > 0:
            dy_.append(sum_)
        else:
            dy_.append(0)
    
    init_ths = 0.6 * np.max(dy[:3*fs])
    ths = init_ths
    recent_5_peakAmp = []
    peak_ind = []
    bef_idx = -300
    
    for idx in range(1,len(dy_)-1):
        prevAmp = dy_[idx-1]
        curAmp = dy_[idx]
        nextAmp = dy_[idx+1]
        if determine_peak_or_not(prevAmp, curAmp, nextAmp) == True:
            if (idx - bef_idx) > (300 * fs /1000):  # Ignore if the previous peak was within 300 ms interval
                if len(recent_5_peakAmp) < 100:  
                    if curAmp > ths:
                        peak_ind.append(idx)
                        bef_idx = idx
                        recent_5_peakAmp.append(curAmp)
                elif len(recent_5_peakAmp) == 100:
                    ths = 0.7*np.median(recent_5_peakAmp)
                    if curAmp > ths:
                        peak_ind.append(idx)
                        bef_idx = idx
                        recent_5_peakAmp.pop(0)
                        recent_5_peakAmp.append(curAmp)
                        
    onoffset = onoff_set(peak_ind, dy_)
    corrected_peak_ind = []
    for onoff in onoffset:
        segment = data[onoff[0]:onoff[1]]
        corrected_peak_ind.append(np.argmax(segment) + onoff[0])
                    
    return corrected_peak_ind


# +
'''
Method 4

Event-Related Moving Averages with Dynamic Threshold

Reference

1. Elgendi, Mohamed, et al. "Systolic peak detection in acceleration photoplethysmograms 
measured from emergency responders in tropical conditions." PLoS One 8.10 (2013).

2. https://github.com/neuropsychology/NeuroKit/blob/8a2148fe477f20328d18b6da7bbb1c8438e60f18/neurokit2/signal/signal_formatpeaks.py

'''

def moving_average(signal, kernel='boxcar', size=5):
    size = int(size)
    window = scipy.signal.get_window(kernel, size)
    w = window / window.sum()
    
    # Extend signal edges to avoid boundary effects
    x = np.concatenate((signal[0] * np.ones(size), signal, signal[-1] * np.ones(size)))
    
    # Compute moving average
    smoothed = np.convolve(w, x, mode='same')
    smoothed = smoothed[size:-size]
    return smoothed


def moving_averages_with_dynamic_ths(signals,sampling_rate=64, peakwindow=.111, 
                                     beatwindow=.667, beatoffset=.02, mindelay=.3,show=False):
    if show:
        fig, (ax0, ax1) = plt.subplots(nrow=2, ncols=1, sharex=True)
        ax0.plot(data, label='filtered')
    
    signal = signals.copy()
    # ignore the samples with n
    signal[signal < 0] = 0
    sqrd = signal**2
    
    # Compute the thresholds for peak detection. Call with show=True in order
    # to visualize thresholds.
    ma_peak_kernel = int(np.rint(peakwindow * sampling_rate))
    ma_peak = moving_average(sqrd, size=ma_peak_kernel)
    
    ma_beat_kernel = int(np.rint(beatwindow * sampling_rate))
    ma_beat = moving_average(sqrd, size=ma_beat_kernel)

    
    thr1 = ma_beat + beatoffset * np.mean(sqrd)    # threshold 1

    if show:
        ax1.plot(sqrd, label="squared")
        ax1.plot(thr1, label="threshold")
        ax1.legend(loc="upper right")

    # Identify start and end of PPG waves.
    waves = ma_peak > thr1
    
    beg_waves = np.where(np.logical_and(np.logical_not(waves[0:-1]),
                                        waves[1:]))[0]
    end_waves = np.where(np.logical_and(waves[0:-1],
                                        np.logical_not(waves[1:])))[0]
    # Throw out wave-ends that precede first wave-start.
    end_waves = end_waves[end_waves > beg_waves[0]]

    # Identify systolic peaks within waves (ignore waves that are too short).
    num_waves = min(beg_waves.size, end_waves.size)
    min_len = int(np.rint(peakwindow * sampling_rate))    # threshold 2
    min_delay = int(np.rint(mindelay * sampling_rate))
    peaks = [0]

    for i in range(num_waves):

        beg = beg_waves[i]
        end = end_waves[i]
        len_wave = end - beg

        if len_wave < min_len: # threshold 2
            continue

        # Visualize wave span.
        if show:
            ax1.axvspan(beg, end, facecolor="m", alpha=0.5)

        # Find local maxima and their prominence within wave span.
        data = signal[beg:end]
        locmax, props = scipy.signal.find_peaks(data, prominence=(None, None))

        if locmax.size > 0:
            # Identify most prominent local maximum.
            peak = beg + locmax[np.argmax(props["prominences"])]
            # Enforce minimum delay between peaks(300ms)
            if peak - peaks[-1] > min_delay:
                peaks.append(peak)

    peaks.pop(0)

    if show:
        ax0.scatter(peaks, signal[peaks], c="r")

    peaks = np.asarray(peaks).astype(int)
    return peaks


# +
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def lmm_peakdetection(data,fs):
    
    peak_final = []
    peaks, _ = find_peaks(data,height=0)
    
    for peak in peaks:
        if data[peak] > 0:
            peak_final.append(peak)
        
    return peak_final



def ensemble_peak(preprocessed_data, fs, ensemble_ths=4):
    
    peak1 = threshold_peakdetection(preprocessed_data,fs)
    peak2 = slope_sum_function(preprocessed_data, fs)
    peak3 = first_derivative_with_adaptive_ths(preprocessed_data, fs)
    peak4 = moving_averages_with_dynamic_ths(preprocessed_data)
    peak5 = lmm_peakdetection(preprocessed_data,fs)
    
    peak_dic = dict()

    for key in peak1:
        peak_dic[key] = 1

    for key in peak2:
        if key in peak_dic.keys():
            peak_dic[key] += 1
        else:
            peak_dic[key] = 1
    
    for key in peak3:
        if key in peak_dic.keys():
            peak_dic[key] += 1
        else:
            peak_dic[key] = 1
        
    for key in peak4:
        if key in peak_dic.keys():
            peak_dic[key] += 1
        else:
            peak_dic[key] = 1
        
    for key in peak5:
        if key in peak_dic.keys():
            peak_dic[key] += 1
        else:
            peak_dic[key] = 1
        
    peak_dic = dict(sorted(peak_dic.items()))

    count = 0
    cnt = 0
    bef_key = 0
    margin = 1

    new_peak_dic = dict()

    for key in peak_dic.keys():
        if cnt == 0:
            new_peak_dic[key] = peak_dic[key]
        else:
            if bef_key + margin >= key:  # 마진 1안에 다음 피크가 존재하면
                if peak_dic[bef_key] > peak_dic[key]: # 이전 피크 기준으로 개수 카운트
                    new_peak_dic[bef_key] += peak_dic[key]
                else:
                    #print("new peak dic: ",new_peak_dic)
                    new_peak_dic[key] = peak_dic[key] + peak_dic[bef_key] # 현재 피크 기준으로 개수 카운트
                    del(new_peak_dic[bef_key])
                    bef_key = key
            else:
                new_peak_dic[key] = peak_dic[key]
                bef_key = key
        cnt += 1
    
    ensemble_dic = dict()
    for (key, value) in new_peak_dic.items():
        if value >= ensemble_ths:
            ensemble_dic[key] = value
            
    final_peak = list(ensemble_dic.keys())
    
    return final_peak


