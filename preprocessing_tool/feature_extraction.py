# -*- coding: utf-8 -*-
# !pip install nolds

# +
import sys
sys.path.append("/home/sheo1/stress_classification_with_PPG/preprocessing_tool/") 

from noise_reduction import *
from peak_detection import *

import math
import numpy as np
import pandas as pd
import nolds

from scipy.interpolate import UnivariateSpline
from scipy import stats


# +
def calc_RRI(peaklist, fs):
    RR_list = []
    RR_list_e = []
    cnt = 0
    while (cnt < (len(peaklist)-1)):
        RR_interval = (peaklist[cnt+1] - peaklist[cnt]) #Calculate distance between beats in # of samples
        ms_dist = ((RR_interval / fs) * 1000.0)  #fs로 나눠서 1초단위로 거리표현 -> 1ms단위로 change /  Convert sample distances to ms distances
        cnt += 1
        RR_list.append(ms_dist)
    mean_RR = np.mean(RR_list)

    for ind, rr in enumerate(RR_list):
        if rr >  mean_RR - 300 and rr < mean_RR + 300:
            RR_list_e.append(rr)
            
    RR_diff = []
    RR_sqdiff = []
    cnt = 0
    while (cnt < (len(RR_list_e)-1)):
        RR_diff.append(abs(RR_list_e[cnt] - RR_list_e[cnt+1]))
        RR_sqdiff.append(math.pow(RR_list_e[cnt] - RR_list_e[cnt+1], 2))
        cnt += 1
        
    return RR_list_e, RR_diff, RR_sqdiff

def calc_heartrate(RR_list):
    HR = []
    heartrate_array=[]
    window_size = 10

    for val in RR_list:
        if val > 400 and val < 1500:
            heart_rate = 60000.0 / val #60000 ms (1 minute) / 한번 beat하는데 걸리는 시간
        # if RR-interval < .1905 seconds, heart-rate > highest recorded value, 315 BPM. Probably an error!
        elif (val > 0 and val < 400) or val > 1500:
            if len(HR) > 0:
                # ... and use the mean heart-rate from the data so far:
                heart_rate = np.mean(HR[-window_size:])

            else:
                heart_rate = 60.0
        else:
            # Get around divide by 0 error
            print("err")
            heart_rate = 0.0

        HR.append(heart_rate)

    return HR


# -

def get_window_stats_original(ppg_seg, window_length, label=-1):  # Nan을 제외하고 평균 냄 
    
    fs = 64   
    
    peak = threshold_peakdetection(ppg_seg, fs)
    RR_list, RR_diff, RR_sqdiff = calc_RRI(peak, fs)
    
    # Time
    HR = calc_heartrate(RR_list)
    HR_mean, HR_std = np.mean(HR), np.std(HR)
    SD_mean, SD_std = np.mean(RR_diff) , np.std(RR_diff)
    NN50 = [x for x in RR_diff if x > 50]
    pNN50 = len(NN50) / window_length
    bar_y, bar_x = np.histogram(RR_list)
    TINN = np.max(bar_x) - np.min(bar_x)
    RMSSD = np.sqrt(np.mean(RR_sqdiff))
    
    # Frequency
    rr_x = []
    pointer = 0
    for x in RR_list:
        pointer += x
        rr_x.append(pointer)
    RR_x_new = np.linspace(rr_x[0], rr_x[-1], int(rr_x[-1]))
    
    if len(rr_x) <= 5 or len(RR_list) <= 5:
        print("rr_x or RR_list less than 5")   
    
   
    interpolated_func = UnivariateSpline(rr_x, RR_list, k=3)
    
    datalen = len(RR_x_new)
    frq = np.fft.fftfreq(datalen, d=((1/1000.0)))
    frq = frq[range(int(datalen/2))]
    Y = np.fft.fft(interpolated_func(RR_x_new))/datalen
    Y = Y[range(int(datalen/2))]
    psd = np.power(Y, 2)  # power spectral density

    lf = np.trapz(abs(psd[(frq >= 0.04) & (frq <= 0.15)])) #Slice frequency spectrum where x is between 0.04 and 0.15Hz (LF), and use NumPy's trapezoidal integration function to find the are
    hf = np.trapz(abs(psd[(frq > 0.15) & (frq <= 0.5)])) #Do the same for 0.16-0.5Hz (HF)
    ulf = np.trapz(abs(psd[frq < 0.003]))
    vlf = np.trapz(abs(psd[(frq >= 0.003) & (frq < 0.04)]))
    
    if hf != 0:
        lfhf = lf/hf
    else:
        lfhf = 0
        
    total_power = lf + hf + vlf

    features = {'HR_mean': HR_mean, 'HR_std': HR_std, 'SD_mean': SD_mean, 'SD_std': SD_std, 'pNN50': pNN50, 'TINN': TINN, 'RMSSD': RMSSD,
                'LF': lf, 'HF': hf, 'ULF' : ulf, 'VLF': vlf, 'LFHF': lfhf, 'Total_power': total_power, 'label': label}

    return features


# +

def approximate_entropy(U, m=2, r=3):

    def _maxdist(x_i, x_j):
        return max([abs(ua - va) for ua, va in zip(x_i, x_j)])

    def _phi(m):
        x = [[U[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
        C = [len([1 for x_j in x if _maxdist(x_i, x_j) <= r]) / (N - m + 1.0) for x_i in x]
        return (N - m + 1.0)**(-1) * sum(np.log(C))

    N = len(U)

    return abs(_phi(m+1) - _phi(m))

def shannon_entropy(signal):
    #signal = list(signal)
    
    data_set = list(set(signal))
    freq_list = []
    for entry in data_set:
        counter = 0.
        for i in signal:
            if i == entry:
                counter += 1
        freq_list.append(float(counter) / len(signal))
        
    ent = 0.0
    for freq in freq_list:
        ent += freq * np.log2(freq)
    
    ent = -ent
    
    return ent



# https://horizon.kias.re.kr/12415/

def sample_entropy(sig,ordr,tor):
    # sig: the input signal or series, it should be numpy array with type float
    # ordr: order, the length of template,embedding dimension
    # tor: percent of standard deviation
    
    sig = np.array(sig)
    n = len(sig)
    #tor = np.std(sig)*tor
    
    matchnum = 0.0
    for i in range(n-ordr):
        tmpl = sig[i:i+ordr] # generate samples length ordr
        for j in range (i+1,n-ordr+1): 
            ltmp = sig[j:j+ordr]
            diff = tmpl-ltmp  # measure mean similarity
            if all(diff<tor):
                matchnum+=1
    
    allnum = (n-ordr+1)*(n-ordr)/2
    if matchnum<0.1:
        sen = 1000.0
    else:
        sen = -math.log(matchnum/allnum)
    return sen


# -

def calc_td_hrv(RR_list, RR_diff, RR_sqdiff, window_length): 
    
    # Time
    HR = calc_heartrate(RR_list)
    HR_mean, HR_std = np.mean(HR), np.std(HR)
    meanNN, SDNN, medianNN = np.mean(RR_list), np.std(RR_list), np.median(np.abs(RR_list))
    meanSD, SDSD = np.mean(RR_diff) , np.std(RR_diff)
    RMSSD = np.sqrt(np.mean(RR_sqdiff))
    
    NN20 = [x for x in RR_diff if x > 20]
    NN50 = [x for x in RR_diff if x > 50]
    pNN20 = len(NN20) / window_length
    pNN50 = len(NN50) / window_length
    
    
    bar_y, bar_x = np.histogram(RR_list)
    TINN = np.max(bar_x) - np.min(bar_x)
    
    RMSSD = np.sqrt(np.mean(RR_sqdiff))
    

    features = {'HR_mean': HR_mean, 'HR_std': HR_std, 'meanNN': meanNN, 'SDNN': SDNN, 'medianNN': medianNN,
                'meanSD': meanSD, 'SDSD': SDSD, 'RMSSD': RMSSD, 'pNN20': pNN20, 'pNN50': pNN50, 'TINN': TINN}

    return features


def calc_fd_hrv(RR_list):  
    
    rr_x = []
    pointer = 0
    for x in RR_list:
        pointer += x
        rr_x.append(pointer)
        
    if len(rr_x) <= 3 or len(RR_list) <= 3:
        print("rr_x or RR_list less than 5")   
        return 0
    
    RR_x_new = np.linspace(rr_x[0], rr_x[-1], int(rr_x[-1]))
    
   
    interpolated_func = UnivariateSpline(rr_x, RR_list, k=3)
    
    datalen = len(RR_x_new)
    frq = np.fft.fftfreq(datalen, d=((1/1000.0)))
    frq = frq[range(int(datalen/2))]
    Y = np.fft.fft(interpolated_func(RR_x_new))/datalen
    Y = Y[range(int(datalen/2))]
    psd = np.power(Y, 2)  # power spectral density

    lf = np.trapz(abs(psd[(frq >= 0.04) & (frq <= 0.15)])) #Slice frequency spectrum where x is between 0.04 and 0.15Hz (LF), and use NumPy's trapezoidal integration function to find the are
    hf = np.trapz(abs(psd[(frq > 0.15) & (frq <= 0.5)])) #Do the same for 0.16-0.5Hz (HF)
    ulf = np.trapz(abs(psd[frq < 0.003]))
    vlf = np.trapz(abs(psd[(frq >= 0.003) & (frq < 0.04)]))
    
    if hf != 0:
        lfhf = lf/hf
    else:
        lfhf = 0
        
    total_power = lf + hf + vlf
    lfp = lf / total_power
    hfp = hf / total_power

    features = {'LF': lf, 'HF': hf, 'ULF' : ulf, 'VLF': vlf, 'LFHF': lfhf, 'total_power': total_power, 'lfp': lfp, 'hfp': hfp}
    bef_features = features
    
    return features


def calc_nonli_hrv(RR_list,label): 
    
    diff_RR = np.diff(RR_list)
    sd_heart_period = np.std(diff_RR, ddof=1) ** 2
    SD1 = np.sqrt(sd_heart_period * 0.5)
    SD2 = 2 * sd_heart_period - 0.5 * sd_heart_period
    pA = SD1*SD2
    
    if SD2 != 0:
        pQ = SD1 / SD2
    else:
        print("SD2 is zero")
        pQ = 0
    
    ApEn = approximate_entropy(RR_list,2,3)  
    shanEn = shannon_entropy(RR_list)
    #sampEn = nolds.sampen(RR_list,emb_dim=2)
    D2 = nolds.corr_dim(RR_list, emb_dim=2)
    #dfa1 = nolds.dfa(RR_list, range(4,17))
    # dfa2 = nolds.dfa(RR_list, range(16,min(len(RR_list)-1, 66)))
    #dimension, delay, threshold, norm, minimum_diagonal_line_length = 3, 2, 0.7, "manhattan", 2
    #rec_mat = recurrence_matrix(RR_list, dimension, delay, threshold, norm)
    #REC, RPImean, RPImax, RPadet = recurrence_quantification_analysis(rec_mat, minimum_diagonal_line_length)
    # recurrence_rate, average_diagonal_line_length, longest_diagonal_line_length, determinism

    features = {'SD1': SD1, 'SD2': SD2, 'pA': pA, 'pQ': pQ, 'ApEn' : ApEn, 'shanEn': shanEn, 'D2': D2, 
                'label': label}
    # 'dfa1': dfa1, 'dfa2': dfa2, 'REC': REC, 'RPImean': RPImean, 'RPImax': RPImax, 'RPadet': RPadet,
    return features


def get_window_stats_27_features(ppg_seg, window_length, label, ensemble, ma_usage):  
    
    fs = 64  
    
    if ma_usage:
        fwd = moving_average(ppg_seg, size=3)
        bwd = moving_average(ppg_seg[::-1], size=3)
        ppg_seg = np.mean(np.vstack((fwd,bwd[::-1])), axis=0)
    ppg_seg = np.array([item.real for item in ppg_seg])
    
    #peak = threshold_peakdetection(ppg_seg, fs)
    #peak = first_derivative_with_adaptive_ths(ppg_seg, fs)
    #peak = slope_sum_function(ppg_seg, fs)
    #peak = moving_averages_with_dynamic_ths(ppg_seg)
    peak = lmm_peakdetection(ppg_seg,fs)

        
    if ensemble:
        ensemble_ths = 3
        #print("one algorithm peak length: ", len(peak))
        peak = ensemble_peak(ppg_seg, fs, ensemble_ths)
        #print("after ensemble peak length: ", len(peak))
        
        if(len(peak) < 100):
            print("skip")
            return []

        
    RR_list, RR_diff, RR_sqdiff = calc_RRI(peak, fs)
    #print(RR_list)
    
    if len(RR_list) <= 3:
        return []
    
    td_features = calc_td_hrv(RR_list, RR_diff, RR_sqdiff, window_length)
    fd_features = calc_fd_hrv(RR_list)
    
    if fd_features == 0:
        return []
    nonli_features = calc_nonli_hrv(RR_list,label)
    
    total_features = {**td_features, **fd_features, **nonli_features}
    
    
    return total_features


