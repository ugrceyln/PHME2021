#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd


import os

# os.system("pip install -q sacred")
# os.system("pip install -q openpyxl")

from sacred import Experiment
from sacred.commands import print_config
from sacred.observers import FileStorageObserver



ex = Experiment('PHME21', interactive=True)
ex.observers.append(FileStorageObserver('./lda_logs/'))

@ex.config
def configuration_settings():
    fill_missing_ = True
    scale_type = "standard"
    window_size_ = 5
    train_model = True
    eval_model = True
    subsample = 10 # for sampling
    stride = 5 # for stride
    
    num_leaves_ = -1
    learning_rate_ = 0.1
    n_estimators_ = 100
    


### import data
data_df_1 = pd.read_csv("../../../data/imputed_training_validation_1.csv")
data_df_2 = pd.read_csv("../../../data/imputed_training_validation_2.csv")
data_df_3 = pd.read_csv("../../../data/imputed_model_refinement.csv")

# data_df_1 = pd.read_csv("/kaggle/input/phme21-imputed-dataset/imputed_training_validation_1.csv")
# data_df_2 = pd.read_csv("/kaggle/input/phme21-imputed-dataset/imputed_training_validation_2.csv")
# data_df_3 = pd.read_csv("/kaggle/input/phme21-imputed-dataset/imputed_model_refinement.csv")

merged_df = pd.concat([data_df_1, data_df_2, data_df_3], axis=0) # Merge data frames

# train_df = merged_df[~ (merged_df['class'] == 0)]

train_df = merged_df.copy()

train_df['runId'] = 1000 * train_df['class'] + train_df['run']

labels = train_df['class']
runs = train_df['runId']

run_df = train_df[['class', 'runId']].copy()
run_df.drop_duplicates(inplace=True)
run_df.reset_index(inplace=True)
del run_df['index']

del train_df['run']

full_sensor_list = list(train_df.columns)
full_sensor_list.remove('class')
full_sensor_list.remove('runId')
len(full_sensor_list)


sensor_list_lgbm = [
    'Temperature_value', 'Humidity_value', 'LightBarrierActiveTaskDuration1_vMax', 'LightBarrierActiveTaskDuration1_vFreq',
    'SmartMotorSpeed_vTrend', 'DurationPickToPick_value', 'Pressure_vStd',  'VacuumFusePicked_vStd', 'EPOSVelocity_vStd', 
    'FusePicked_vMin', 'VacuumFusePicked_vTrend', 'TotalMemoryConsumption_vStd', 'IntensityTotalThermoImage_vCnt', 
    'ProcessCpuLoadNormalized_vMax', 'SmartMotorPositionError_vMax', 'TotalMemoryConsumption_vMin', 
    'TemperatureThermoCam_vFreq', 'ValidFrame_vFreq', 'LightBarrierPassiveTaskDuration1_vFreq', 'FuseHeatSlopeOK_vFreq', 
    'DurationRobotFromFeederToTestBench_value', 'FuseOutsideOperationalSpace_vMax', 'FuseHeatSlopeNOK_vMax', 'Vacuum_vStd', 
    'VacuumValveClosed_vTrend', 'IntensityTotalImage_vCnt', 'VacuumValveClosed_vMin', 'FeederAction1_vCnt', 
    'VacuumValveClosed_vCnt', 'FuseHeatSlopeNOK_value', 
          'IntensityTotalThermoImage_vFreq', 
          'ProcessMemoryConsumption_vMin', 
          'SharpnessImage_vFreq', 
          'FusePicked_vTrend', 
          'EPOSVelocity_vMin', 
          'LightBarrierPassiveTaskDuration1_value', 
          'CpuTemperature_vStd', 
          'EPOSCurrent_vFreq',
          'EPOSPosition_vCnt',
          'FuseHeatSlope_vFreq',
          'EPOSCurrent_vTrend', 
          'IntensityTotalThermoImage_vMin', 
          'DurationRobotFromTestBenchToFeeder_vCnt', 
          'FusePicked_vMax', 
          'DurationTestBenchClosed_value', 
          'IntensityTotalThermoImage_vStd', 
          'DurationRobotFromFeederToTestBench_vStd', 
          'LightBarrierActiveTaskDuration1_vMin',
          'LightBarrierPassiveTaskDuration1_vMax', 
          'TotalCpuLoadNormalized_vMin', 
          'Vacuum_vMin', 
          'LightBarrierActiveTaskDuration1_value', 
          'Vacuum_vMax', 
          'NumberFuseDetected_vCnt', 
          'DurationRobotFromFeederToTestBench_vMin', 
          'IntensityTotalThermoImage_vTrend', 
          'FuseHeatSlope_value', 
          'SmartMotorPositionError_vMin', 
          'Vacuum_vTrend', 
          'TemperatureThermoCam_vMax', 
          'IntensityTotalThermoImage_vMax', 
          'LightBarrierActiveTaskDuration1_vCnt', 
          'ProcessMemoryConsumption_vMax', 
          'VacuumFusePicked_value', 
          'CpuTemperature_value', 
          'DurationTestBenchClosed_vTrend', 
          'TemperatureThermoCam_vMin', 
          'FuseOutsideOperationalSpace_value', 
          'FuseHeatSlope_vCnt', 
          'SmartMotorSpeed_vFreq', 
          'TemperatureThermoCam_value', 
          'LightBarrierPassiveTaskDuration1_vMin', 
          'DurationRobotFromFeederToTestBench_vTrend', 
          'FuseCycleDuration_vMax', 
          'NumberFuseEstimated_vCnt', 
          'TemperatureThermoCam_vCnt', 
          'EPOSCurrent_vStd', 
          'FeederBackgroundIlluminationIntensity_vFreq', 
          'FeederAction3_vCnt',
          'LightBarrierPassiveTaskDuration1_vTrend', 
          'SmartMotorPositionError_vTrend', 
          'EPOSVelocity_value', 
          'FuseHeatSlopeNOK_vFreq', 
          'EPOSPosition_vTrend', 
          'Pressure_vFreq', 
          'TotalCpuLoadNormalized_value', 
          'FuseCycleDuration_value', 
          'SharpnessImage_vCnt',
          'DurationTestBenchClosed_vCnt', 
          'DurationRobotFromTestBenchToFeeder_vFreq',
          'FuseHeatSlope_vMin', 
          'DurationPickToPick_vStd', 
          'DurationTestBenchClosed_vFreq', 
          'DurationRobotFromFeederToTestBench_vCnt', 
          'LightBarrierActiveTaskDuration1_vTrend',
          'IntensityTotalImage_vFreq',
          'FusePicked_vStd', 
          'SmartMotorSpeed_vCnt', 
          'SmartMotorPositionError_value', 
          'ProcessCpuLoadNormalized_vMin', 
          'IntensityTotalThermoImage_value', 
          'FuseOutsideOperationalSpace_vFreq', 
          'TemperatureThermoCam_vTrend', 
          'DurationPickToPick_vTrend', 
          'FuseHeatSlope_vTrend', 
          'FuseTestResult_vTrend', 
          'DurationRobotFromTestBenchToFeeder_vMin', 
          'FuseTestResult_value', 
          'Vacuum_vFreq', 
          'ProcessMemoryConsumption_vStd', 
          'DurationRobotFromFeederToTestBench_vMax', 
          'VacuumValveClosed_vStd', 
          'EPOSPosition_vMax', 
          'EPOSPosition_vStd', 
          'DurationTestBenchClosed_vMin',
          'Vacuum_vCnt', 
          'EPOSPosition_vMin', 
          'DurationRobotFromTestBenchToFeeder_vMax',
          'FuseTestResult_vStd', 
          'FeederBackgroundIlluminationIntensity_vCnt',
          'ErrorFrame_vCnt', 
          'SmartMotorPositionError_vStd', 
          'SmartMotorSpeed_value', 
          'FuseOutsideOperationalSpace_vStd', 
          'LightBarrierActiveTaskDuration1_vStd', 
          'DurationPickToPick_vCnt', 
          'VacuumValveClosed_vMax', 
          'FuseTestResult_vMin', 
          'EPOSVelocity_vCnt', 
          'EPOSCurrent_vMax', 
          'TotalCpuLoadNormalized_vStd',
          'EPOSCurrent_vCnt', 
          'NumberEmptyFeeder_vCnt', 
          'ProcessCpuLoadNormalized_value',
          'FuseCycleDuration_vStd', 
          'EPOSCurrent_value', 
          'FuseCycleDuration_vMin',
          'FusePicked_vCnt', 
          'DurationRobotFromTestBenchToFeeder_vTrend', 
          'FusePicked_value',
          'VacuumFusePicked_vCnt',
          'Pressure_vCnt', 
          'Pressure_vTrend',
          'FeederAction2_vCnt', 
          'VacuumValveClosed_value', 
          'FuseHeatSlopeNOK_vCnt', 
          'NumberFuseEstimated_vFreq', 
          'FuseCycleDuration_vFreq', 
          'FuseIntoFeeder_vCnt',
          'ValidFrame_vCnt',
          'VacuumFusePicked_vMin', 
          'Vacuum_value', 
          'FeederAction4_vCnt',
          'DurationTestBenchClosed_vMax', 
          'LightBarrierPassiveTaskDuration1_vStd',
          'SmartMotorPositionError_vCnt', 
          'DurationRobotFromTestBenchToFeeder_value',
          'ErrorFrame_vFreq', 
          'LightBarrierPassiveTaskDuration1_vCnt', 
           'ProcessMemoryConsumption_value', 
          'FuseCycleDuration_vTrend', 
          'TemperatureThermoCam_vStd', 
          'EPOSCurrent_vMin',
          'FuseTestResult_vCnt',
          'EPOSVelocity_vTrend',
          'ProcessCpuLoadNormalized_vStd',
          'SmartMotorPositionError_vFreq',
          'FuseHeatSlopeOK_vCnt']

sensor_list = sensor_list_lgbm

# from catboost import CatBoostClassifier
# from ngboost import NGBClassifier
# from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.svm import SVC

from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import Matern, RBF, RationalQuadratic, DotProduct, WhiteKernel

# from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA

from sklearn.metrics import f1_score, accuracy_score, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold


from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline
from imblearn.metrics import classification_report_imbalanced

# Globals

# Main sensor list. 
# List shall include other features such as operating conditions. 
_subsample = 10
_stride = 5

# scale_type = "standard"
# window_size = 50
# cv_fold = 3

# train_model = True
# eval_model = True
verbose = False

# split a sequence into samples
def create_sequence(sequence, n_steps):
    X = list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence):
            break
        # gather input and output parts of the pattern
        seq_x = sequence[i:end_ix]
        X.append(seq_x)
    return np.array(X)

use_subsample = False

def create_dataset_for_run(df, ws):
#     data_data = np.empty((0, ws * len(sensor_list))) # for 1D
#     data_data = np.empty((0, ws, len(sensor_list))) # for 2D
#     data_data = np.empty((0, len(sensor_list), ws)) # for 2D
#     label_data = np.empty((0, 1))

    sensors_df = df.filter(sensor_list)

    # Calculate seq of windows_size len
    if use_subsample:    
        seq = create_sequence(sensors_df.values, n_steps=_subsample)
        seq = seq.mean(axis=1)
        seq = seq[::_stride]
        seq = create_sequence(seq, n_steps=ws)
    else:
        seq = create_sequence(sensors_df.values, n_steps=ws)

#     seq = np.transpose(seq, axes=(0, 2, 1))
    seq_count = seq.shape[0]
    seq = seq.reshape((seq_count, -1)) # for 1D

    # add new seq to data_data array
#     data_data = np.vstack((data_data, seq))

    # Calculate RULS
    labels = df['class'].values[:seq_count]

    # add rul to rul_data array
#     rul_data = np.vstack((rul_data, ruls))

# TODO: What is RUL_Max in this context?

#     print ("Shape:", seq.shape, labels.shape)
    return seq, labels


 
# TODO: X_t, X_tp1, y_t, y_tp1 should be calculated per run.  
# TODO: Then should be merged into one X_t, X_tp1, y_t, y_tp1.
def create_datasets(df, ws):
    
    run_list = df['runId'].unique()

    X_df_list = []
    y_df_list = []
    
    for r in run_list:
        r_df = df[df['runId'] == r]
#         print ("--> r: ", r, r_df.shape)
        sensor_data, label_data = create_dataset_for_run(r_df, ws)

        # Post Processing for the model

        # Padding for model input 
        padded_sensor_data = sensor_data.copy() #np.hstack((sensor_data, np.zeros((sensor_data.shape[0], 2)))) # for AE     

        # Calculate X(t) and X(t+1) for model input/output 
        X_t = padded_sensor_data[:]

        # Calculate y(t) and y(t+1) for model input/output 
        y_t = label_data[:]

        X_df_list.append(pd.DataFrame(X_t))
        y_df_list.append(pd.DataFrame(y_t))
    
    X_t = pd.concat(X_df_list, axis=0) # Merge data frames
    y_t = pd.concat(y_df_list, axis=0) # Merge data frames

    return X_t.values, y_t.values.flatten()

fillna_list = [True]
ws_list = [25, 50, 75, 100, 150, 200]

params_num_leaves = [5, 10, 15, 20, 30, 40, 50]
param_learning_rate = [0.1, 0.01, 0.001]
params_n_estimators = [100, 200, 300, 400, 500, 1000]

total_runs = len(fillna_list) * len(ws_list) * len(params_num_leaves) * len(param_learning_rate)  * len(params_n_estimators)


kernel1 = RBF(length_scale=1.0, length_scale_bounds=(1e-1, 1e2)) # + WhiteKernel()
kernel2 = Matern(length_scale=1.0, length_scale_bounds=(1e-1, 1e2), nu=1.5) #+ WhiteKernel()


# some_train_df contains only Class 0 and Class C instances
def train_model_for_class(X_train, y_train):

    model = make_pipeline(
        StandardScaler(),
        # SMOTE(),
        LinearDiscriminantAnalysis(), 
        SVC(class_weight='balanced', gamma='auto', probability=True)
    )

    model.fit(X_train, y_train)
    
    return model


from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

import pickle
import joblib


run_counter = 0
results_df = pd.DataFrame()

@ex.main
def ex_main(_run, fill_missing_, scale_type, window_size_, train_model, eval_model, subsample, stride, num_leaves_, learning_rate_, n_estimators_):

    global total_runs
    global run_counter
    global results_df
    
    print ('===========================================================================')
    print ('Run:', run_counter+1, '/', total_runs)
    print ("Fill missing:", fill_missing_, "Window Size:", window_size_, "LGBM Params:", num_leaves_, learning_rate_, n_estimators_)

    run_counter += 1
    
    acc_sum = 0
    f1_sum = 0
    mcc_sum = 0
    
    cv = StratifiedKFold(n_splits=3, shuffle=True)

    for fold, (training_indices, validation_indices) in enumerate(cv.split(run_df['runId'], run_df['class'])):
        print ("--> Fold: ", fold)

        training_runIds = run_df.loc[training_indices]['runId']
        validation_runIds = run_df.loc[validation_indices]['runId']

        X_train_df = train_df[train_df['runId'].isin(training_runIds)].copy()
        X_val_df = train_df[train_df['runId'].isin(validation_runIds)].copy()

        X_train, y_train = create_datasets(X_train_df, window_size_)
        X_val, y_val = create_datasets(X_val_df, window_size_)

    #     print ("Data shape", X_train_df.shape, X_val_df.shape)
        print ("Train data shape:", X_train.shape, y_train.shape)
        print ("Val data shape:", X_val.shape, y_val.shape)

        one_vs_one_models = []
        class_list = sorted(train_df['class'].unique())
        
        X_train_predictions = np.zeros((X_train.shape[0], 2 * len(class_list[1:])))
        X_val_predictions = np.zeros((X_val.shape[0], 2 * len(class_list[1:])))
        
        for i, c in enumerate(class_list[1:]): # For all classes except 0
            c_X_train = X_train[(y_train==0) | (y_train==c)]
            c_y_train = y_train[(y_train==0) | (y_train==c)]
            print ("Training for class", c, c_X_train.shape)
            
            model = train_model_for_class(c_X_train, c_y_train)
            model_pred = model.predict_proba(X_train)
            
            X_train_predictions[:, 2*i:2*(i+1)] = model_pred
            
            one_vs_one_models.append(model)

        fusion_model = make_pipeline(
            # SMOTE(),
#             GaussianProcessClassifier(kernel=kernel1+kernel2)
            LinearDiscriminantAnalysis(), 
            SVC(class_weight='balanced', gamma='auto')
        )

        print ("Fusing...")
        fusion_model.fit(X_train_predictions, y_train)

        for i, c in enumerate(class_list[1:]): # For all classes except 0
            model = one_vs_one_models[i]
            val_pred = model.predict_proba(X_val)
            X_val_predictions[:, 2*i:2*(i+1)] = val_pred
            
        pred = fusion_model.predict(X_val_predictions)

            
        acc_val = accuracy_score(y_val, pred)
        f1_val = f1_score(y_val, pred, average='weighted')
        mcc_val = matthews_corrcoef(y_val, pred)
        
        acc_sum += acc_val
        f1_sum += f1_val
        mcc_sum += mcc_val
        
        print ("Fold:", fold, "ACC:", acc_val, "F1:", f1_val, "MCC:", mcc_val)

        cm = confusion_matrix(y_val, pred, labels=sorted(train_df['class'].unique()))

        print (cm)

#         for c in sorted(train_df['class'].unique()):
#             print (c, 
#                    f1_score(y_val[y_val == c], model.predict(X_val[y_val == c]), average='weighted'), 
#                    matthews_corrcoef(y_val[y_val == c], model.predict(X_val[y_val == c])))
        
    print ()
    print ("Avg ACC:", acc_sum / 3.0, "Avg F1:", f1_sum / 3.0, "Avg MCC:", mcc_sum / 3.0)

    result = {
        'ML_Algorithm': 'SVM_one_vs_one',
        'Window Size': window_size_, 
        'ACC': acc_sum / 3.0,
        'F1': f1_sum / 3.0,
        'MCC': mcc_sum / 3.0,
    }

    results_df = results_df.append(result, ignore_index=True)
    results_df.to_excel("LDA_2_step_smote.xlsx")    


ws_list = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
ws_list = [45, 50]

total_runs = len(ws_list)
run_counter = 0

for ws in ws_list:
    ex.run(config_updates={
        'window_size_': ws,
    })

