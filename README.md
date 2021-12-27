# Stress Detection with Single PPG sensor by Orchestrating Multiple Denoising and Peak Detection Method (IEEE ACCESS 2021)

Code implementing the OMDP method introduced in "Stress Detection with Single PPG sensor by Orchestrating Multiple Denoising and Peak Detection Method" by Seongsil Heo, Sunyoung Kwon, and Jaekoo Lee*.

![block-diagram](https://user-images.githubusercontent.com/44438752/147434917-dc11f55e-3ee3-4f97-84fa-f74aaf9f0f51.png)

# Abstract
Stress is one of the major causes of diseases in modern society. Therefore, measuring and
managing the degree of stress is crucial to maintain a healthy life. The goal of this paper is to improve
stress-detection performance using precise signal processing based on photoplethysmogram (PPG) data.
PPG signals can be collected through wearable devices, but are affected by many internal and external
noises. To solve this problem, we propose a two-step denoising method, to filter the noise in terms of
frequency and remove the remaining noise in terms of time. We also propose an ensemble-based multiple
peak-detecting method to extract accurate features through refined signals. We used a typical public dataset,
namely, wearable stress and affect detection dataset (WESAD) and measured the performance of the
proposed PPG denoising and peak-detecting methods by lightweight multiple classifiers. By measuring
the stress-detection performance using the proposed method, we demonstrate an improved result compared
with the existing methods: accuracy is 96.50 and the F1 score is 93.36%

## Run
### 1. Download the WESAD dataset

https://archive.ics.uci.edu/ml/datasets/WESAD+%28Wearable+Stress+and+Affect+Detection%29

### 2. Data processing
If you want to apply noise filtering noise elimination, and ensemble, include 'bp', 'time', 'ens' each in variable *NOISE*.
Change the *main_path* where your WESAD dataset is located.

    python read_data_new_binary.py
    python read_data_new_tri.py
    python read_data_new_quad.py
    
### 3. Train & Test

    python ML_binary.py
    python ML_tri.py
    python ML_quad.py
    

## Cite
If you use this code in your own work, please cite our paper.

    @article{heo2021stress,
      title={Stress Detection With Single PPG Sensor by Orchestrating Multiple Denoising and Peak-Detecting Methods},
      author={Heo, Seongsil and Kwon, Sunyoung and Lee, Jaekoo},
      journal={IEEE Access},
      volume={9},
      pages={47777--47785},
      year={2021},
      publisher={IEEE}
    }
