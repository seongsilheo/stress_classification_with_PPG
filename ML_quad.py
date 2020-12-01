
import pandas as pd
import numpy as np
import csv

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import tree
from sklearn import svm
from sklearn.preprocessing import StandardScaler

from sklearn import metrics  
from sklearn.metrics import roc_curve, auc, roc_auc_score, f1_score
from sklearn.metrics import recall_score, precision_score, accuracy_score
from sklearn.metrics import confusion_matrix  


feats = ['HR_mean','HR_std','meanNN','SDNN','medianNN','meanSD','SDSD','RMSSD','pNN20','pNN50','TINN','LF','HF','ULF','VLF','LFHF',
         'total_power','lfp','hfp','SD1','SD2','pA','pQ','ApEn','shanEn','D2','subject','label']
WINDOW_SIZE = '120'

NOISE = ['ens_1','bp_ens_1','bp_time_ens_1']
NOISE = ['bp_1', 'bp_time_1']

subjects = [2,3,4,5,6,7,8,9,10,11,13,14,15,16,17]

# +
from collections import Counter



def read_csv(path, feats, testset_num):
    print("testset num: ",testset_num)
    df = pd.read_csv(path, index_col = 0)
    
    df = df[feats]

    train_df = df.loc[df['subject'] != testset_num]
    test_df =  df.loc[df['subject'] == testset_num]

    del train_df['subject']
    del test_df['subject']
    del df['subject']

    
    X_train = train_df.drop('label', axis=1).values
    y_train = train_df['label'].values   
    X_test = test_df.drop('label', axis=1).values
    y_test = test_df['label'].values    
    
    return df, X_train, y_train, X_test, y_test
# -



# # Machine learning



# +
def DT_model(X_train, y_train, X_test, y_test):
    
    model = tree.DecisionTreeClassifier(random_state=0)
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    
    ACC = accuracy_score(y_test, y_pred)

    
    fpr,tpr, roc_auc = dict(), dict(), dict()
    n_classes = 4
    
    
    y_pred = np.eye(n_classes)[y_pred]
    y_test = np.eye(n_classes)[y_test]  # one-hot-vector
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    AUC = np.array(list(roc_auc.values())).mean()
    F1 = f1_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1), average='macro')  
    
    return AUC, F1, ACC


def RF_model(X_train, y_train, X_test, y_test):
    
    model = RandomForestClassifier(max_depth=4, random_state=0)
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    
    ACC = accuracy_score(y_test, y_pred)
    
    fpr,tpr, roc_auc = dict(), dict(), dict()
    n_classes = 4
    
    y_pred = np.eye(n_classes)[y_pred]
    y_test = np.eye(n_classes)[y_test]  # one-hot-vector
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    AUC = np.array(list(roc_auc.values())).mean()
    F1 = f1_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1), average='macro')  
    
    return AUC, F1, ACC

def AB_model(X_train, y_train, X_test, y_test):
    
    model = AdaBoostClassifier(random_state=0)
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    
    ACC = accuracy_score(y_test, y_pred)
    
    fpr,tpr, roc_auc = dict(), dict(), dict()
    n_classes = 4
    
    y_pred = np.eye(n_classes)[y_pred]
    y_test = np.eye(n_classes)[y_test]  # one-hot-vector
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    AUC = np.array(list(roc_auc.values())).mean()
    F1 = f1_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1), average='macro')  
    
    return AUC, F1, ACC

def KN_model(X_train, y_train, X_test, y_test):
    
    model = KNeighborsClassifier(n_neighbors=9)
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    
    ACC = accuracy_score(y_test, y_pred)
    
    fpr,tpr, roc_auc = dict(), dict(), dict()
    n_classes = 4

    y_pred = np.eye(n_classes)[y_pred]
    y_test = np.eye(n_classes)[y_test]  # one-hot-vector
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    AUC = np.array(list(roc_auc.values())).mean()
    F1 = f1_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1), average='macro')  
    
    return AUC, F1, ACC

def LDA_model(X_train, y_train, X_test, y_test):
    
    model = LinearDiscriminantAnalysis()
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    
    ACC = accuracy_score(y_test, y_pred)
    
    fpr,tpr, roc_auc = dict(), dict(), dict()
    n_classes = 4
    
    y_pred = np.eye(n_classes)[y_pred]
    y_test = np.eye(n_classes)[y_test]  # one-hot-vector
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    AUC = np.array(list(roc_auc.values())).mean()
    F1 = f1_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1), average='macro')  
    
    
    return AUC, F1, ACC

def SVM_model(X_train, y_train, X_test, y_test):
    
    model = svm.SVC()
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    
    ACC = accuracy_score(y_test, y_pred)
    
    fpr,tpr, roc_auc = dict(), dict(), dict()
    n_classes = 4
    
    y_pred = np.eye(n_classes)[y_pred]
    y_test = np.eye(n_classes)[y_test]  # one-hot-vector
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    AUC = np.array(list(roc_auc.values())).mean()
    F1 = f1_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1), average='macro')  
    
    
    return AUC, F1, ACC


def GB_model(X_train, y_train, X_test, y_test):
    
    model = GradientBoostingClassifier(random_state=0)
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    
    ACC = accuracy_score(y_test, y_pred)
    
    fpr,tpr, roc_auc = dict(), dict(), dict()
    n_classes = 4
    
    y_pred = np.eye(n_classes)[y_pred]
    y_test = np.eye(n_classes)[y_test]  # one-hot-vector
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    AUC = np.array(list(roc_auc.values())).mean()
    F1 = f1_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1), average='macro')  
    
    
    return AUC, F1, ACC

# +
feats = ['HR_mean','HR_std','meanNN','SDNN','medianNN','meanSD','SDSD','RMSSD','pNN20','pNN50','TINN','LF','HF','ULF','VLF','LFHF',
         'total_power','SD1','SD2','pA','pQ','ApEn','shanEn','D2','subject','label']       

for n in NOISE:
    
    #path = '27_features_ppg_9/data_merged_' + n + WINDOW_SIZE + '.csv'
    #result_path_all = 'result/BGM2/all_features_' + n + WINDOW_SIZE + '.csv'

    path = '27_features_ppg_test_4/LMM/data_merged_' + n + WINDOW_SIZE + '.csv'
    result_path_all = 'result_4/LMM/all_features_' + n + WINDOW_SIZE + '.csv'

    DT_AUC, DT_F1, DT_ACC = [], [], []
    RF_AUC, RF_F1, RF_ACC = [], [], []
    AB_AUC, AB_F1, AB_ACC = [], [], []
    KN_AUC, KN_F1, KN_ACC = [], [], []
    LDA_AUC, LDA_F1, LDA_ACC = [], [], []
    SVM_AUC, SVM_F1, SVM_ACC = [], [], []
    GB_AUC, GB_F1, GB_ACC = [], [], []

    for sub in subjects:
    
        df, X_train, y_train, X_test, y_test = read_csv(path, feats, sub)
        df.fillna(0)
        # Normalization
        sc = StandardScaler()  
        X_train = sc.fit_transform(X_train)  
        X_test = sc.transform(X_test)  

    
        auc_dt, f1_dt, acc_dt = DT_model(X_train, y_train, X_test, y_test)
        auc_rf, f1_rf, acc_rf = RF_model(X_train, y_train, X_test, y_test)
        auc_ab, f1_ab, acc_ab = AB_model(X_train, y_train, X_test, y_test)
        auc_kn, f1_kn, acc_kn = KN_model(X_train, y_train, X_test, y_test)
        auc_lda, f1_lda, acc_lda = LDA_model(X_train, y_train, X_test, y_test)
        auc_svm, f1_svm, acc_svm = SVM_model(X_train, y_train, X_test, y_test)
        auc_gb, f1_gb, acc_gb = GB_model(X_train, y_train, X_test, y_test)

        DT_AUC.append(auc_dt)
        DT_F1.append(f1_dt)
        DT_ACC.append(acc_dt)
        RF_AUC.append(auc_rf)
        RF_F1.append(f1_rf)
        RF_ACC.append(acc_rf)
        AB_AUC.append(auc_ab)
        AB_F1.append(f1_ab)
        AB_ACC.append(acc_ab)
        KN_AUC.append(auc_kn)
        KN_F1.append(f1_kn)
        KN_ACC.append(f1_kn)
        LDA_AUC.append(auc_lda)
        LDA_F1.append(f1_lda)
        LDA_ACC.append(acc_lda)
        SVM_AUC.append(auc_svm)
        SVM_F1.append(f1_svm)
        SVM_ACC.append(acc_svm)
        GB_AUC.append(auc_gb)
        GB_F1.append(f1_gb)
        GB_ACC.append(acc_gb)
    

    with open(result_path_all, 'w', newline='') as file:
        writer = csv.writer(file)

        writer.writerow(['subject','S2','S3','S4','S5','S6','S7','S8','S9','S10','S11','S13','S14','S15','S16','S17','total'])
        writer.writerow(['DT_AUC'] + DT_AUC + [np.mean(DT_AUC)])
        writer.writerow(['RF_AUC'] + RF_AUC + [np.mean(RF_AUC)])
        writer.writerow(['AB_AUC'] + AB_AUC + [np.mean(AB_AUC)])
        writer.writerow(['KN_AUC'] + KN_AUC + [np.mean(KN_AUC)])
        writer.writerow(['LDA_AUC'] + LDA_AUC + [np.mean(LDA_AUC)])
        writer.writerow(['SVM_AUC'] + SVM_AUC + [np.mean(SVM_AUC)])
        writer.writerow(['GB_AUC'] + GB_AUC + [np.mean(GB_AUC)])
        writer.writerow(['DT_F1'] + DT_F1 + [np.mean(DT_F1)])
        writer.writerow(['RF_F1'] + RF_F1 + [np.mean(RF_F1)])
        writer.writerow(['AB_F1'] + AB_F1 + [np.mean(AB_F1)])
        writer.writerow(['KN_F1'] + KN_F1 + [np.mean(KN_F1)])
        writer.writerow(['LDA_F1'] + LDA_F1 + [np.mean(LDA_F1)])
        writer.writerow(['SVM_F1'] + SVM_F1 + [np.mean(SVM_F1)])
        writer.writerow(['GB_F1'] + GB_F1 + [np.mean(GB_F1)])
        writer.writerow(['DT_ACC'] + DT_ACC + [np.mean(DT_ACC)])
        writer.writerow(['RF_ACC'] + RF_ACC + [np.mean(RF_ACC)])
        writer.writerow(['AB_ACC'] + AB_ACC + [np.mean(AB_ACC)])
        writer.writerow(['KN_ACC'] + KN_ACC + [np.mean(KN_ACC)])
        writer.writerow(['LDA_ACC'] + LDA_ACC + [np.mean(LDA_ACC)])
        writer.writerow(['SVM_ACC'] + SVM_ACC + [np.mean(SVM_ACC)])
        writer.writerow(['GB_ACC'] + GB_ACC + [np.mean(GB_ACC)])


        file.close()
 
    print("DONE: ",n)
# -
# #### 

