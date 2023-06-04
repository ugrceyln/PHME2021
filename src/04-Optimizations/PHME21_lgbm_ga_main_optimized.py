import pandas as pd
import numpy as np

running_on = 'local'
# running_on = 'kaggle'

import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if (running_on == 'kaggle'):
    os.system("conda install -q sacred")
    os.system("conda install -q openpyxl")

import pickle
import joblib

sensor_list_lgbm = [
    'Temperature_value', 'Humidity_value', 'LightBarrierActiveTaskDuration1_vMax', 'LightBarrierActiveTaskDuration1_vFreq',
    'SmartMotorSpeed_vTrend', 'DurationPickToPick_value', 'Pressure_vStd',  'VacuumFusePicked_vStd', 'EPOSVelocity_vStd', 
    'FusePicked_vMin', 'VacuumFusePicked_vTrend', 'TotalMemoryConsumption_vStd', 'IntensityTotalThermoImage_vCnt', 
    'ProcessCpuLoadNormalized_vMax', 'SmartMotorPositionError_vMax', 'TotalMemoryConsumption_vMin', 
    'TemperatureThermoCam_vFreq', 'ValidFrame_vFreq', 'LightBarrierPassiveTaskDuration1_vFreq', 'FuseHeatSlopeOK_vFreq', 
    'DurationRobotFromFeederToTestBench_value', 'FuseOutsideOperationalSpace_vMax', 'FuseHeatSlopeNOK_vMax', 'Vacuum_vStd', 
    'VacuumValveClosed_vTrend', 'IntensityTotalImage_vCnt', 'VacuumValveClosed_vMin', 'FeederAction1_vCnt', 
    'VacuumValveClosed_vCnt', 'FuseHeatSlopeNOK_value', 'IntensityTotalThermoImage_vFreq', 'ProcessMemoryConsumption_vMin', 
    'SharpnessImage_vFreq', 'FusePicked_vTrend', 'EPOSVelocity_vMin', 'LightBarrierPassiveTaskDuration1_value', 
    'CpuTemperature_vStd', 'EPOSCurrent_vFreq', 'EPOSPosition_vCnt', 'FuseHeatSlope_vFreq', 'EPOSCurrent_vTrend', 
    'IntensityTotalThermoImage_vMin', 'DurationRobotFromTestBenchToFeeder_vCnt', 'FusePicked_vMax', 
    'DurationTestBenchClosed_value', 'IntensityTotalThermoImage_vStd', 'DurationRobotFromFeederToTestBench_vStd', 
    'LightBarrierActiveTaskDuration1_vMin','LightBarrierPassiveTaskDuration1_vMax', 'TotalCpuLoadNormalized_vMin', 
    'Vacuum_vMin', 'LightBarrierActiveTaskDuration1_value', 'Vacuum_vMax', 'NumberFuseDetected_vCnt', 
    'DurationRobotFromFeederToTestBench_vMin', 'IntensityTotalThermoImage_vTrend', 'FuseHeatSlope_value', 
    'SmartMotorPositionError_vMin', 'Vacuum_vTrend', 'TemperatureThermoCam_vMax', 'IntensityTotalThermoImage_vMax', 
    'LightBarrierActiveTaskDuration1_vCnt', 'ProcessMemoryConsumption_vMax', 'VacuumFusePicked_value', 
    'CpuTemperature_value', 'DurationTestBenchClosed_vTrend', 'TemperatureThermoCam_vMin', 
    'FuseOutsideOperationalSpace_value', 'FuseHeatSlope_vCnt', 'SmartMotorSpeed_vFreq', 'TemperatureThermoCam_value', 
    'LightBarrierPassiveTaskDuration1_vMin', 'DurationRobotFromFeederToTestBench_vTrend', 'FuseCycleDuration_vMax', 
    'NumberFuseEstimated_vCnt', 'TemperatureThermoCam_vCnt', 'EPOSCurrent_vStd', 
    'FeederBackgroundIlluminationIntensity_vFreq', 'FeederAction3_vCnt','LightBarrierPassiveTaskDuration1_vTrend', 
    'SmartMotorPositionError_vTrend', 'EPOSVelocity_value', 'FuseHeatSlopeNOK_vFreq', 'EPOSPosition_vTrend', 
    'Pressure_vFreq', 'TotalCpuLoadNormalized_value', 'FuseCycleDuration_value', 'SharpnessImage_vCnt',
    'DurationTestBenchClosed_vCnt', 'DurationRobotFromTestBenchToFeeder_vFreq','FuseHeatSlope_vMin', 
    'DurationPickToPick_vStd', 'DurationTestBenchClosed_vFreq', 'DurationRobotFromFeederToTestBench_vCnt', 
    'LightBarrierActiveTaskDuration1_vTrend','IntensityTotalImage_vFreq','FusePicked_vStd', 'SmartMotorSpeed_vCnt', 
    'SmartMotorPositionError_value', 'ProcessCpuLoadNormalized_vMin', 'IntensityTotalThermoImage_value', 
    'FuseOutsideOperationalSpace_vFreq', 'TemperatureThermoCam_vTrend', 'DurationPickToPick_vTrend', 
    'FuseHeatSlope_vTrend', 'FuseTestResult_vTrend', 'DurationRobotFromTestBenchToFeeder_vMin', 'FuseTestResult_value', 
    'Vacuum_vFreq', 'ProcessMemoryConsumption_vStd', 'DurationRobotFromFeederToTestBench_vMax', 'VacuumValveClosed_vStd', 
    'EPOSPosition_vMax', 'EPOSPosition_vStd', 'DurationTestBenchClosed_vMin','Vacuum_vCnt', 'EPOSPosition_vMin', 
    'DurationRobotFromTestBenchToFeeder_vMax','FuseTestResult_vStd', 'FeederBackgroundIlluminationIntensity_vCnt',
    'ErrorFrame_vCnt', 'SmartMotorPositionError_vStd', 'SmartMotorSpeed_value', 'FuseOutsideOperationalSpace_vStd', 
    'LightBarrierActiveTaskDuration1_vStd', 'DurationPickToPick_vCnt', 'VacuumValveClosed_vMax', 'FuseTestResult_vMin', 
    'EPOSVelocity_vCnt', 'EPOSCurrent_vMax', 'TotalCpuLoadNormalized_vStd','EPOSCurrent_vCnt', 'NumberEmptyFeeder_vCnt', 
    'ProcessCpuLoadNormalized_value','FuseCycleDuration_vStd', 'EPOSCurrent_value', 'FuseCycleDuration_vMin',
    'FusePicked_vCnt', 'DurationRobotFromTestBenchToFeeder_vTrend', 'FusePicked_value','VacuumFusePicked_vCnt',
    'Pressure_vCnt', 'Pressure_vTrend','FeederAction2_vCnt', 'VacuumValveClosed_value', 'FuseHeatSlopeNOK_vCnt', 
    'NumberFuseEstimated_vFreq', 'FuseCycleDuration_vFreq', 'FuseIntoFeeder_vCnt','ValidFrame_vCnt',
    'VacuumFusePicked_vMin', 'Vacuum_value', 'FeederAction4_vCnt','DurationTestBenchClosed_vMax', 
    'LightBarrierPassiveTaskDuration1_vStd','SmartMotorPositionError_vCnt', 'DurationRobotFromTestBenchToFeeder_value',
    'ErrorFrame_vFreq', 'LightBarrierPassiveTaskDuration1_vCnt',  'ProcessMemoryConsumption_value', 
    'FuseCycleDuration_vTrend', 'TemperatureThermoCam_vStd', 'EPOSCurrent_vMin','FuseTestResult_vCnt',
    'EPOSVelocity_vTrend','ProcessCpuLoadNormalized_vStd','SmartMotorPositionError_vFreq','FuseHeatSlopeOK_vCnt'
    ]

sensor_list_lgbm_2 = ['Temperature_value', 'Humidity_value', 'DurationTestBenchClosed_vFreq', 'FuseHeatSlopeNOK_vMin', 'Pressure_vMax', 'FeederAction4_vCnt', 'ProcessCpuLoadNormalized_value', 'Vacuum_vFreq', 'TotalMemoryConsumption_vMin', 'CpuTemperature_vStd', 'SmartMotorPositionError_vMax', 'FuseCycleDuration_vMax', 'LightBarrierActiveTaskDuration1_vTrend', 'DurationRobotFromFeederToTestBench_vMin', 'SmartMotorSpeed_vMax', 'FusePicked_vFreq', 'DurationPickToPick_vFreq', 'DurationTestBenchClosed_value', 'DurationPickToPick_vTrend', 'VacuumValveClosed_vStd', 'LightBarrierPassiveTaskDuration1_vCnt', 'DurationRobotFromTestBenchToFeeder_vMin', 'NumberFuseEstimated_vFreq', 'VacuumFusePicked_vFreq', 'FuseTestResult_vStd', 'EPOSCurrent_vStd', 'DurationTestBenchClosed_vMin', 'VacuumFusePicked_vMin', 'TemperatureThermoCam_value', 'SmartMotorPositionError_vFreq', 'LightBarrierActiveTaskDuration1_vMax', 'IntensityTotalThermoImage_vMax', 'DurationRobotFromFeederToTestBench_vStd', 'TemperatureThermoCam_vTrend', 'EPOSVelocity_vStd', 'ProcessMemoryConsumption_vStd', 'VacuumFusePicked_vCnt', 'FuseHeatSlopeNOK_vFreq', 'FuseCycleDuration_vCnt', 'Vacuum_vTrend', 'LightBarrierActiveTaskDuration1_vStd', 'FuseCycleDuration_vMin', 'DurationRobotFromTestBenchToFeeder_vTrend', 'FuseHeatSlope_vMax', 'FusePicked_vMin', 'DurationTestBenchClosed_vCnt', 'SmartMotorPositionError_vTrend', 'SmartMotorSpeed_vFreq', 'Vacuum_vMax', 'Pressure_vMin', 'Vacuum_value', 'VacuumFusePicked_value', 'Pressure_vFreq', 'FuseOutsideOperationalSpace_vCnt', 'FuseHeatSlopeNOK_vMax', 'EPOSVelocity_vTrend', 'NumberFuseEstimated_vCnt', 'Vacuum_vStd', 'EPOSCurrent_vTrend', 'FuseOutsideOperationalSpace_vFreq', 'EPOSVelocity_vFreq', 'VacuumValveClosed_value', 'LightBarrierActiveTaskDuration1_vCnt', 'FuseHeatSlope_value', 'ValidFrame_vCnt', 'Vacuum_vCnt', 'TotalMemoryConsumption_vMax', 'Vacuum_vMin', 'TotalMemoryConsumption_vStd', 'ProcessCpuLoadNormalized_vMax', 'EPOSCurrent_vMax', 'TotalCpuLoadNormalized_vStd', 'ValidFrameOptrisPIIRCamera_vFreq', 'FusePicked_vTrend', 'DurationRobotFromTestBenchToFeeder_vFreq', 'FuseCycleDuration_value', 'EPOSCurrent_vMin', 'VacuumValveClosed_vMin', 'FeederAction2_vCnt', 'DurationPickToPick_vMax', 'FuseCycleDuration_vFreq', 'LightBarrierPassiveTaskDuration1_vStd', 'NumberFuseDetected_vFreq', 'DurationTestBenchClosed_vTrend', 'LightBarrierActiveTaskDuration1_vMin', 'DurationRobotFromFeederToTestBench_vMax', 'FusePicked_vCnt', 'EPOSVelocity_vCnt', 'DurationPickToPick_vCnt', 'FuseHeatSlope_vMin', 'SmartMotorPositionError_vStd', 'SmartMotorPositionError_vMin', 'EPOSVelocity_vMax', 'EPOSPosition_vMin', 'TemperatureThermoCam_vStd', 'DurationRobotFromFeederToTestBench_vFreq', 'FuseHeatSlope_vTrend', 'EPOSVelocity_vMin', 'SmartMotorSpeed_value', 'LightBarrierActiveTaskDuration1_value', 'FuseTestResult_value', 'VacuumFusePicked_vTrend', 'FusePicked_vStd', 'EPOSPosition_vTrend', 'VacuumValveClosed_vFreq', 'LightBarrierPassiveTaskDuration1_vTrend', 'ValidFrameOptrisPIIRCamera_vCnt', 'DurationRobotFromTestBenchToFeeder_vCnt', 'LightBarrierActiveTaskDuration1_vFreq', 'FuseTestResult_vFreq', 'EPOSPosition_vFreq', 'IntensityTotalImage_vFreq', 'EPOSCurrent_vFreq', 'FeederBackgroundIlluminationIntensity_vCnt', 'DurationRobotFromTestBenchToFeeder_value', 'EPOSCurrent_value', 'DurationPickToPick_vStd', 'SmartMotorSpeed_vTrend', 'FuseCycleDuration_vStd', 'DurationRobotFromFeederToTestBench_vCnt', 'FuseHeatSlope_vStd', 'Pressure_vCnt', 'FuseIntoFeeder_vCnt', 'ValidFrame_vFreq', 'ProcessMemoryConsumption_vMin', 'TotalCpuLoadNormalized_vMax', 'DurationTestBenchClosed_vStd', 'DurationPickToPick_value', 'CpuTemperature_vMin', 'LightBarrierPassiveTaskDuration1_vFreq', 'DurationPickToPick_vMin', 'CpuTemperature_vMax', 'VacuumValveClosed_vMax', 'CpuTemperature_value', 'TotalCpuLoadNormalized_vMin', 'DurationTestBenchClosed_vMax', 'VacuumFusePicked_vStd', 'ProcessMemoryConsumption_vMax', 'EPOSCurrent_vCnt', 'Pressure_value', 'ProcessMemoryConsumption_value']

sensor_list = sensor_list_lgbm_2

print (len(sensor_list))

# activations = ['relu', 'tanh', 'sigmoid', 'linear']

cv_fold = 3

# Globals
train_df = None # Load data files
run_df = None # Load data files

Filenames = None # 

Scale_type = None # ex.main
Score_df = None # create_and_evaluate_model
Score_df_cols = None # create_and_evaluate_model, ga_optimizer_setup

toolbox = None # ga_optimizer_setup
genom_list =  None # ga_optimizer_setup

#********************************************** Utils
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

def squared_error(y_true, y_pred):
    return ((y_true - y_pred)**2).sum()
                  
def absolute_error(y_true, y_pred):
    return np.abs(y_true - y_pred).sum()
    
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

    sensors_df = df.filter(sensor_list)

    # Calculate seq of windows_size len
    seq = create_sequence(sensors_df.values, n_steps=ws)
    seq_count = seq.shape[0]
    seq = seq.reshape((seq_count, -1))  # for 1D

    labels = df['class'].values[:seq_count]

    return seq, labels


def create_datasets(df, ws):
    run_list = df['runId'].unique()

    X_df_list = []
    y_df_list = []

    for r in run_list:
        r_df = df[df['runId'] == r]
        # print ("--> r: ", r, r_df.shape)
        sensor_data, label_data = create_dataset_for_run(r_df, ws)

        # Post Processing for the model

        # Padding for model input
        padded_sensor_data = sensor_data.copy()  # np.hstack((sensor_data, np.zeros((sensor_data.shape[0], 2)))) # for AE

        # Calculate X(t) and X(t+1) for model input/output
        X_t = padded_sensor_data[:]

        # Calculate y(t) and y(t+1) for model input/output
        y_t = label_data[:]

        X_df_list.append(pd.DataFrame(X_t))
        y_df_list.append(pd.DataFrame(y_t))

    X_t = pd.concat(X_df_list, axis=0)  # Merge data frames
    y_t = pd.concat(y_df_list, axis=0)  # Merge data frames

    return X_t.values, y_t.values.flatten()


from sklearn.preprocessing import LabelBinarizer

class Custom_Scaler(object):
    
    def get_scaler(self, scaler_name=None):
        ret = None
        if scaler_name == 'standard':
            ret = StandardScaler()
        elif scaler_name == 'minmax':
            ret = MinMaxScaler()
        elif scaler_name == 'minmax01':
            ret = MinMaxScaler()
        elif scaler_name == 'minmax-11':
            ret = MinMaxScaler(feature_range=(-1,1))
        elif scaler_name == 'robust':
            ret = RobustScaler()
        return ret

    def __init__(self, scaler_name):
        self.profiles = []
        self.scalers = []
        self.scaler_name = scaler_name
    
    def fit(self, X_df, y=0):
        self.scalers = {}
        self.profiles = X_df.Profile.unique()
        # Full dataset fit
        for profile in self.profiles:
            sensors_readings = X_df[(X_df['Profile']==profile)].filter(sensor_list)          # Get sensor readings
            state_scaler = self.get_scaler(self.scaler_name).fit(sensors_readings)              # Fit scaler
            self.scalers[profile] = state_scaler                                                   # Add to sclaer_list for further reference
        
        return self
     
    def transform(self, X_df, y=0):
        # Full dataset transform
        for profile in self.profiles:
            sensors_readings = X_df[(X_df['Profile']==profile)].filter(sensor_list)          # Get sensor readings
#             if sensors_readings.shape[0] == 0: continue # Should it be? 
            cols = sensors_readings.columns
            normalized_sensor_readings = self.scalers[profile].transform(sensors_readings)      # transform sensor readings
            X_df.loc[(X_df['Profile']==profile), cols] = normalized_sensor_readings                  # record transformed values

        return X_df
   
# def write_model_summary(model, fname):
#     with open(fname,'w') as fh:
#         # Pass the file handle in as a lambda function to make it callable
#         model.summary(line_length=120, print_fn=lambda x: fh.write(x + '\n'))

#********************************************** Model  
from sklearn.metrics import f1_score, accuracy_score, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline, make_pipeline

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

from lightgbm import LGBMClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def create_model(config):
    
    _, attr_lr, \
    attr_boostingtype, attr_noofleaves, \
    attr_maxdepth, attr_nofestimators, \
    attr_minsplitgain, attr_subsample, \
    attr_subsamplefreq, attr_colsample = tuple(config)
               
    model = make_pipeline(
                RobustScaler(), 
                LinearDiscriminantAnalysis(), # For dimension reduction
                LGBMClassifier( # For classification
                    learning_rate=attr_lr,
                    boosting_type=attr_boostingtype,
                    num_leaves=attr_noofleaves,
                    max_depth=attr_maxdepth,
                    n_estimators=attr_nofestimators,
                    min_split_gain=attr_minsplitgain,
                    subsample=attr_subsample,
                    subsample_freq=attr_subsamplefreq,
                    colsample_bytree=attr_colsample,
                    metric='multi_logloss', 
                    objective='multiclass') 
            )

    return model    

run_counter = 0

def run_model(config): # config --> GA individual
    global run_counter
    print ('***************************************************************')
    print ("--> Individual ID: ", run_counter)
    print (config)

    run_counter += 1
    
    window_size = config[0]
    # learning_rate = config[1]
    
    ind_score_columns = ['fold', 'ACC_Score', 'F1_Score', 'MCC_Score']
    ind_score_df = pd.DataFrame(columns=ind_score_columns)

    # acc_sum = 0
    # f1_sum = 0
    # mcc_sum = 0

    cv = StratifiedKFold(n_splits=cv_fold, shuffle=True)

    for fold, (training_indices, validation_indices) in enumerate(cv.split(run_df['runId'], run_df['class'])):
        print ('===========================')
        print ("--> Fold: ", fold)

# Training        
        print ("Training...")
        training_runIds = run_df.loc[training_indices]['runId']
        X_train_df = train_df[train_df['runId'].isin(training_runIds)].copy()

        X_train, y_train = create_datasets(X_train_df, window_size)
        print ("Train data shape:", X_train.shape, y_train.shape)

        model = create_model(config)
        model.fit(X_train, y_train)

        joblib.dump(model, Filenames['pickle'].format(window_size, fold))

# Evaluation
        print ("Evaluating...")
        validation_runIds = run_df.loc[validation_indices]['runId']
        X_val_df = train_df[train_df['runId'].isin(validation_runIds)].copy()

        X_val, y_val = create_datasets(X_val_df, window_size)
        print ("Val data shape:", X_val.shape, y_val.shape)

        pred = model.predict(X_val)

        acc_val = accuracy_score(y_val,pred)
        f1_val = f1_score(y_val, pred, average='weighted')
        mcc_val = matthews_corrcoef(y_val, pred)
        
        # acc_sum += acc_val
        # f1_sum += f1_val
        # mcc_sum += mcc_val

        print ("Fold:", fold, "ACC:", acc_val, "F1:", f1_val, "MCC:", mcc_val)

        cm = confusion_matrix(y_val, pred, labels=sorted(train_df['class'].unique()))
        print (cm)

        new_scores = {
            'fold':         fold,
                     
            'ACC_Score':    acc_val,
            'F1_Score':     f1_val,
            'MCC_Score':    mcc_val,
        } 

        print (new_scores)
        ind_score_df = ind_score_df.append(new_scores, ignore_index=True)

        max_ = 0.95
        min_ = 0.75

        threshold = (cv_fold*min_ - (cv_fold - 1 - fold)*max_) / (fold+1)

        if (ind_score_df.MCC_Score.mean() < threshold):
            print ("Terminating evaluation. Avg Score <", threshold)
            break

    print()
    print ("Score means:", ind_score_df.mean())
    print()

    # score_df = score_df.append(score_df.mean(), ignore_index=True)
    ind_score_df.to_excel('fold_scores.xlsx', header=True, index=False)

    return ind_score_df.MCC_Score.mean(), ind_score_df.F1_Score.mean(), ind_score_df.ACC_Score.mean(), 

#********************************************** GA Utils  
import random
import math
from deap import base, creator, tools
from sklearn.metrics import mean_absolute_error

def random_loguniform(low, high, base=10):
#     r = random.uniform(math.log(low, base), math.log(np.nextafter(high, high + 1.)))
    r = random.uniform(math.log(low, base), math.log(high, base))
    return base**(r)

def create_and_evaluate_model(individual):
    mcc_score, f1_score, acc_score = run_model(individual)
    print ("-->", individual)
    print ("ACC:", acc_score, "F1:", f1_score, "MCC:", mcc_score)
    d = individual + [acc_score, f1_score, mcc_score]
    
    global Score_df
    global Score_df_cols # Need not be global.
    
    data = {Score_df_cols[i]: d[i] for i in range(len(Score_df_cols))}
    
    Score_df = Score_df.append(data, ignore_index=True)
    Score_df.to_excel("General_Scores_LGBM.xlsx", header=True)
    
    return mcc_score, f1_score, acc_score

def custom_mutate(individual, prob=0.5):
    size = len(individual)
    for i in range(size):
        if (random.random() < prob):
            fun = getattr(toolbox, genom_list[i][0])
            individual[i] = fun()
    return individual,

def custom_mutate_(individual):
    size = len(individual)
    for i in range(size):
        fun = getattr(toolbox, genom_list[i][0])
        individual[i] = fun()
    return individual,

custom_individual_list = [
    ]

def create_custom_individuals(creator, n):
    assert (n <= len(custom_individual_list)), "n should be <=" + str(len(custom_individual_list))
    
    individuals = []
    for i in range(n):
        individual = custom_individual_list[i]
        individual = creator(individual)
        individuals.append(individual)
        
    return individuals


def create_custom_population(creator, toolbox):
    toolbox.register("population", create_custom_individuals,  creator.Individual)


boosting_type_list = ['gbdt', 'dart'] #, 'goss']

# lgb_param_grid = {
#     'learning_rate': [0.1, 0.01, 0.001, 0.0001],

#     'boosting_type'  : ['gbdt', 'rf', 'goss'],
#     'num_leaves': [5, 10, 20, 30, 40, 50],
#     'max_depth': [-1, 5, 10, 15, 20, 30],
#     'n_estimators': [100, 200, 300, 400, 500, 750, 1000],  
#     'min_split_gain': [0.0, 0.2, 0.5, 0.7, 0.9],
#     'subsample': [0.5, 0.2, 0.8],
#     'subsample_freq': [0, 3, 5, 7],
#     'colsample_bytree': [0.2, 0.5, 0.8, 0.9],

#     'metric' : ['multi_logloss'],
#     'objective' : ['multiclass'],
# }
 


def ga_optimizer_setup(custom_population=False):
    creator.create('FitnessMax', base.Fitness, weights=(1.0, 1.0))
    creator.create('Individual', list, fitness=creator.FitnessMax)

    global genom_list
    
    genom_list = {
    # Preprocessing    
       0 :  ['attr_ws', random.randint, 5, 60],
    # Learing Rate
    # LGBM Model   
       1 :  ['attr_lr', random.choice, [0.1, 0.01, 0.001, 0.0001]],
       2 :  ['attr_boostingtype', random.choice, boosting_type_list],
       3 :  ['attr_noofleaves', random.choice, [2, 5, 10, 20, 30, 40, 50]],
       4 :  ['attr_maxdepth', random.choice, [-1] + list(range(5, 50))], 
       5 :  ['attr_nofestimators', random.choice, list(range(100,1001,50))],
       6 :  ['attr_minsplitgain', random.choice, [0.0, 0.2, 0.5, 0.7, 0.9]],
       7 :  ['attr_subsample', random.choice, [0.2, 0.5, 0.8]], 
       8 :  ['attr_subsamplefreq', random.choice, [0, 3, 5, 7]],
       9 :  ['attr_colsample', random.choice, [0.2, 0.5, 0.8, 0.9]], 
    }

#     attr2id = { 
#         genom_list[i][0]: i for i in range(len(genom_list))
#         }

    global Score_df_cols
    Score_df_cols = [genom_list[i][0] for i in range(len(genom_list))]

    global toolbox 
    toolbox = base.Toolbox()

    # register Genoms
    for i in range(len(genom_list)):
        toolbox.register(genom_list[i][0], *genom_list[i][1:]) 
    
    # Structure initializers
    toolbox.register("individual", tools.initCycle, creator.Individual, 
                     (toolbox.attr_ws, 
                      toolbox.attr_lr, 
                      toolbox.attr_boostingtype,
                      toolbox.attr_noofleaves,
                      toolbox.attr_maxdepth,
                      toolbox.attr_nofestimators,
                      toolbox.attr_minsplitgain,
                      toolbox.attr_subsample,
                      toolbox.attr_subsamplefreq,
                      toolbox.attr_colsample), 1)
    
    if (custom_population):
        toolbox.register("population", create_custom_individuals,  creator.Individual)
    else:
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
#     create_custom_population(creator, toolbox)
    
    toolbox.register("evaluate", create_and_evaluate_model)
    toolbox.register("mate", tools.cxOnePoint)
    # toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("mutate", custom_mutate, prob=0.3)
    # toolbox.register("mutate", custom_mutate_)
    toolbox.register("select", tools.selTournament, tournsize=3)
    # toolbox.register("select", tools.selBest)

from deap import algorithms
from deap.algorithms import varOr

def eaMuPlusLambda(population, toolbox, mu, lambda_, cxpb, mutpb, ngen,
                   stats=None, halloffame=None, verbose=__debug__):
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats is not None else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Vary the population
        offspring = varOr(population, toolbox, lambda_, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Select the next generation population
        population[:] = toolbox.select(population + offspring, mu)

        # Update the statistics with the new population
        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

    return population, logbook

initial_population = [
    [49, 0.01, 'gbdt', 10, -1, 300, 0, 0.5, 0, 0.8],
    [49, 0.01, 'gbdt', 50, 25, 300, 0.5, 0.5, 0, 0.8]    
]

def ga_optimizer_main():
    
    global sensor_list

    pop = toolbox.population(n=30)

    print (pop)

    sensor_list = sensor_list_lgbm
    create_and_evaluate_model(initial_population[0])
    create_and_evaluate_model(initial_population[0])
    create_and_evaluate_model(initial_population[0])
    create_and_evaluate_model(initial_population[1])
    create_and_evaluate_model(initial_population[1])
    create_and_evaluate_model(initial_population[1])

    sensor_list = sensor_list_lgbm_2
    create_and_evaluate_model(initial_population[0])
    create_and_evaluate_model(initial_population[0])
    create_and_evaluate_model(initial_population[0])
    create_and_evaluate_model(initial_population[1])
    create_and_evaluate_model(initial_population[1])
    create_and_evaluate_model(initial_population[1])
    
    exit(0)

    hof = tools.HallOfFame(3, similar=np.array_equal)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
#     stats.register("std", np.std)
    stats.register("min", np.min)
#     stats.register("max", np.max)

    eaMuPlusLambda(pop, toolbox, 3, 10, 0.5, 0.5, 4, halloffame=hof, stats=stats)
    
    print (hof)
    
#********************************************** Sacred main functions
import os

from sacred import Experiment
from sacred.commands import print_config

ex = Experiment()

@ex.config
def ex_config():
    scale_type = "standard"
    modelname = "LGBM"
    outputpath = os.path.join(".", modelname)
    datapath = "D:/0_PhD_Work/PHME21/data/"
    # datapath = "/kaggle/input/phme21-imputed-dataset/"

    filenames = {
        'train_1': os.path.join(datapath, "interpolated_training_validation_1.csv"),
        'train_2': os.path.join(datapath, "interpolated_training_validation_2.csv"),
        'train_3': os.path.join(datapath, "interpolated_model_refinement.csv"),
        'pickle': os.path.join(outputpath, modelname+"_ws_{}_fold_{}.pickle")
    }
    

# Load data files into global variables.

def load_dataset(filenames):
    global train_df
    global run_df

    data_df_1 = pd.read_csv(filenames['train_1'])
    data_df_2 = pd.read_csv(filenames['train_2'])
    data_df_3 = pd.read_csv(filenames['train_3'])

# Merge data files
    merged_df = pd.concat([data_df_1, data_df_2, data_df_3], axis=0) # Merge data frames

# Filter train_df as needed. Usually a copy() will be Ok.
    train_df = merged_df[~ merged_df['run'].isin([56, 74, 49, 23, 35, 6, 83, 54])].copy()
    train_df['runId'] = 1000 * train_df['class'] + train_df['run']

# Prepate run list for cross-validation
    run_df = train_df[['class', 'runId']].copy()
    run_df.drop_duplicates(inplace=True)
    run_df.reset_index(inplace=True)

# Clean 
    del run_df['index']
    del train_df['run']

    
@ex.main
def ex_main(_run, seed, scale_type, modelname, outputpath, filenames):
    global Scale_type
    global Filenames

    print_config(_run)

    try:
        os.mkdir(outputpath)
    except FileExistsError:
        pass

    Scale_type = scale_type
    Filenames = filenames

    np.random.seed(42)
    random.seed(42)

    load_dataset(filenames)
    
    ga_optimizer_setup(custom_population=False)
    
    global Score_df
    global Score_df_cols
    
    Score_df_cols += ['acc_score', 'f1_score', 'mcc_score']
    Score_df = pd.DataFrame(columns=Score_df_cols)
    
    ga_optimizer_main()
    
    Score_df.to_excel("Final_Scores_LGBM.xlsx", header=True)
    
if __name__ == "__main__":

    datapath = "D:/0_PhD_Work/PHME21/data/"
    if (running_on == 'kaggle'):
        datapath = "/kaggle/input/phme21-imputed-dataset/"

    ex.run(config_updates={
        'modelname': 'LGBM',
        'datapath': datapath
    })
