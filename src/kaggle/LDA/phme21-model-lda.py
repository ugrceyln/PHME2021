#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd

import os
# os.system("pip install sacred")
# os.system("pip install openpyxl")


from sacred import Experiment
from sacred.commands import print_config
from sacred.observers import FileStorageObserver

ex = Experiment('PHME21', interactive=True)
ex.observers.append(FileStorageObserver('./lda_logs/'))

from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline

import warnings
warnings.filterwarnings("ignore")


# import data
# data_df_1 = pd.read_csv("/kaggle/input/phme21-imputed-dataset/interpolated_training_validation_1.csv")
# data_df_2 = pd.read_csv("/kaggle/input/phme21-imputed-dataset/interpolated_training_validation_2.csv")
# data_df_3 = pd.read_csv("/kaggle/input/phme21-imputed-dataset/interpolated_model_refinement.csv")

data_df_1 = pd.read_csv("../../../data/interpolated_training_validation_1.csv")
data_df_2 = pd.read_csv("../../../data/interpolated_training_validation_2.csv")
data_df_3 = pd.read_csv("../../../data/interpolated_model_refinement.csv")

merged_df = pd.concat([data_df_1, data_df_2, data_df_3], axis=0) # Merge data frames
train_df = merged_df.copy()

train_df['runId'] = 1000 * train_df['class'] + train_df['run']

labels = train_df['class']
runs = train_df['runId']

run_df = train_df[['class', 'runId']].copy()
run_df.drop_duplicates(inplace=True)
run_df.reset_index(inplace=True)
del run_df['index']

del train_df['run']

sensor = ["Temperature_value",
          'Humidity_value',
          'LightBarrierActiveTaskDuration1_vMax',
          'LightBarrierActiveTaskDuration1_vFreq',
          'SmartMotorSpeed_vTrend',
          'DurationPickToPick_value',
          'Pressure_vStd', 
          'VacuumFusePicked_vStd', 
          'EPOSVelocity_vStd',
          'FusePicked_vMin', 
          'VacuumFusePicked_vTrend',
          'TotalMemoryConsumption_vStd',
          'IntensityTotalThermoImage_vCnt', 
          'ProcessCpuLoadNormalized_vMax', 
          'SmartMotorPositionError_vMax', 
          'TotalMemoryConsumption_vMin',
          'TemperatureThermoCam_vFreq', 
          'ValidFrame_vFreq', 
          'LightBarrierPassiveTaskDuration1_vFreq', 
          'FuseHeatSlopeOK_vFreq', 
          'DurationRobotFromFeederToTestBench_value', 
          'FuseOutsideOperationalSpace_vMax', 
          'FuseHeatSlopeNOK_vMax', 
          'Vacuum_vStd', 
          'VacuumValveClosed_vTrend', 
          'IntensityTotalImage_vCnt', 
          'VacuumValveClosed_vMin', 
          'FeederAction1_vCnt', 
          'VacuumValveClosed_vCnt', 
          'FuseHeatSlopeNOK_value', 
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


sensor_list = sensor

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import f1_score, accuracy_score, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold


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


def create_dataset_for_run(df, ws):
#     data_data = np.empty((0, ws * len(sensor_list))) # for 1D
#     data_data = np.empty((0, ws, len(sensor_list))) # for 2D
#     data_data = np.empty((0, len(sensor_list), ws)) # for 2D
#     label_data = np.empty((0, 1))

    sensors_df = df.filter(sensor_list)

    # Calculate seq of windows_size len
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

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

import pickle
import joblib

run_counter = 0
results_df = pd.DataFrame()

@ex.main
def ex_main(_run, window_size_):    

    global total_runs
    global run_counter
    global results_df
    
    print ('===========================================================================')
    print ('Run:', run_counter+1, '/', total_runs)
    
    print ("Window Size:", window_size_, "Model LDA")
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
        
        print ("Train data shape:", X_train.shape, y_train.shape)
        print ("Val data shape:", X_val.shape, y_val.shape)

        model = make_pipeline(
            RobustScaler(), 
            LinearDiscriminantAnalysis(), # For dimension reduction
            LinearDiscriminantAnalysis() # For classification
        )

        model.fit(X_train, y_train)
        
        pred = model.predict(X_val)

        acc_val = accuracy_score(y_val,pred)
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
    print ("Avg ACC:", acc_sum / 3.0, "Avg F1:", f1_sum / 3.0, "Avg MCC", mcc_sum / 3.0)

    result = {
        'ML_Algorithm': 'LDA', 
        'Window Size': window_size_,         
        'ACC': acc_sum / 3.0,
        'F1': f1_sum / 3.0,
        "MCC" : mcc_sum / 3.0
    }

    results_df = results_df.append(result, ignore_index=True)
    results_df.to_excel("model_lda_result_ws_50.xlsx")

# ws_list = [5, 10, 15, 20, 25, 30, 35, 40, 45]
ws_list = [45] # [30, 35, 40, 45, 50]

total_runs = len(ws_list)

for wsl in ws_list:
    ex.run(config_updates={                        
                        'window_size_': wsl,
                           })

