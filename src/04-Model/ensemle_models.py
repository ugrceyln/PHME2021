import numpy as np
import pandas as pd
from numpy import nan
import matplotlib.pyplot as plt
import seaborn as sns
import random
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import KFold
from lofo import LOFOImportance, Dataset, plot_importance
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import ExtraTreeClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings("ignore")
data_df = pd.read_csv("D:/Datasets/PHME21/df_cleaned.csv")
df = data_df.copy()

refinement_folds = {0: [100, 103, 10, 13, 26, 28, 42, 44, 49, 51, 58, 61, 64, 69, 73, 82, 89, 92, 94, 99],
                    4: [1, 2, 4],
                    11: [0, 1, 3],
                    12: [0, 1, 3]
                    }

# for key, value in refinement_folds.items():
#     df = df.loc[~((df["class"] == key) & df["run"].isin(value))]

print(df.shape)
scaler_cols = list(set(df.columns).difference(["class", "run"]))

scaler = RobustScaler()
scaler_data = scaler.fit_transform(df[scaler_cols])
scaler_data = pd.DataFrame(scaler_data, index=df.index, columns=scaler_cols)


df = pd.concat([df[["class", "run"]], scaler_data], axis=1)
df['runId'] = 1000 * df['class'] + df['run']

run_df = df[['class', 'runId']].copy()
run_df.drop_duplicates(inplace=True)
run_df.reset_index(inplace=True)
# run_df = run_df.sample(frac=1, random_state=14).reset_index(drop=True)

del run_df['index']


# del train_df['class']
del df['run']


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


df_ = df.copy(deep=True)
del df_['runId']

ws = 30
fold_num = 3
cv = StratifiedKFold(n_splits=fold_num, shuffle=True, random_state=53)

# # df_ = df_.sample(frac=1, random_state=0)
# dataset = Dataset(df=df_, target="class", features=[col for col in df_.columns if col != "class"])
#
# model = LGBMClassifier(random_state=41)
# lofo_imp = LOFOImportance(dataset, cv=cv, model=model, scoring="balanced_accuracy", n_jobs=-3)
#
# # get the mean and standard deviation of the importances in pandas format
# importance_df = lofo_imp.get_importance()
# importance_df.to_pickle("importance_df.pkl")
#
# # plot the means and standard deviations of the importances
# plot_importance(importance_df, figsize=(12, 20))
# plt.savefig("importance_df_f1.png")
# plt.show()

importance_df = pd.read_pickle("importance_df.pkl")
# importance_df = importance_df.loc[(importance_df["importance_mean"] > 0.0001) & (importance_df["importance_std"] < 0.001)]
importance_df = importance_df.loc[(importance_df["importance_mean"] > 0)]

sorted_features_imp = list(importance_df["feature"].values)
value_features_imp = list(importance_df["importance_mean"].values)

if len(sorted_features_imp) > 50:
    sorted_features_imp = sorted_features_imp[:130]

sensor_list = sorted_features_imp # list(df.filter(regex="vCnt|value").columns)
print(len(sensor_list), sensor_list)

acc_sum_1 = 0
f1_sum_1 = 0

acc_sum_4 = 0
f1_sum_4 = 0

for fold, (training_indices, validation_indices) in enumerate(cv.split(run_df['runId'], run_df['class'])):
    print("--> Fold: ", fold)

    training_runIds = run_df.loc[training_indices]['runId']
    validation_runIds = run_df.loc[validation_indices]['runId']

    X_train_df = df[df['runId'].isin(training_runIds)].copy()
    X_val_df = df[df['runId'].isin(validation_runIds)].copy()

    X_train_df = X_train_df[sensor_list + ["class", "runId"]]
    X_val_df = X_val_df[sensor_list + ["class", "runId"]]

    X_train, y_train = create_datasets(X_train_df, ws)
    X_val, y_val = create_datasets(X_val_df, ws)

    pca = PCA(n_components=0.99)
    X_train = pca.fit_transform(X_train)
    X_val = pca.transform(X_val)

    print("Train data shape:", X_train.shape, y_train.shape)
    print("Val data shape:", X_val.shape, y_val.shape)

    model1 = LGBMClassifier(random_state=41)
    model1.fit(X_train, y_train)

    pred = model1.predict(X_val)

    acc_val1 = accuracy_score(y_val, pred)
    f1_val1 = f1_score(y_val, pred, average='weighted')

    df_result = pd.DataFrame()
    df_result["actual"] = y_val
    df_result["pred"] = pred
    df_result.to_excel("lgb_fold_{}.xlsx".format(fold))

    acc_sum_1 += acc_val1
    f1_sum_1 += f1_val1
    print("LightGBM Fold:", fold, "ACC:", acc_val1, "F1:", f1_val1)
    acc_val_false = accuracy_score(pred, y_val)
    f1_val1_acc_val_false = f1_score(pred, y_val, average='weighted')
    print("LightGBM-false Fold:", fold, "ACC:", acc_val_false, "F1:", f1_val1_acc_val_false)

    # param = {
    #     'objective': 'multi:softprob',  # error evaluation for multiclass training
    #     'num_class': len(pd.unique(y_train))}  # the number of classes that exist in this datset
    # model4 = XGBClassifier(param, random_state=41, verbosity=0)
    # model4.fit(X_train, y_train)
    #
    # pred = model4.predict(X_val)
    #
    # acc_val4 = accuracy_score(y_val, pred)
    # f1_val4 = f1_score(y_val, pred, average='weighted')
    # df_result4 = pd.DataFrame()
    # df_result4["actual"] = y_val
    # df_result4["pred"] = pred
    # df_result4.to_excel("xgb_fold_{}.xlsx".format(fold))
    #
    # acc_sum_4 += acc_val4
    # f1_sum_4 += f1_val4
    # print("XGBClassifier Fold:", fold, "ACC:", acc_val4, "F1:", f1_val4)
    #
    # acc_val_false = accuracy_score(pred, y_val)
    # f1_val1_acc_val_false = f1_score(pred, y_val, average='weighted')
    # print("XGBClassifier-false Fold:", fold, "ACC:", acc_val_false, "F1:", f1_val1_acc_val_false)

print("LightGBM Avg ACC:", acc_sum_1 / fold_num, "Avg F1:", f1_sum_1 / fold_num)
# print("XGBClassifier Avg ACC:", acc_sum_4 / fold_num, "Avg F1:", f1_sum_4 / fold_num)

"""
FALSE RESULTS
score(y_pred, y_true) (but correct should be score(y_true, y_false))
----------------------------------------------------------
(57971, 207)
feature_num=70
window_size=30
pca(0.99)
--------
Train data shape: (32791, 1101) (32791,)
Val data shape: (22309, 1101) (22309,)
LightGBM Fold: 0 ACC: 0.7053207225783316 F1: 0.7707671737191404
--> Fold:  1
Train data shape: (36907, 1102) (36907,)
Val data shape: (18193, 1102) (18193,)
LightGBM Fold: 1 ACC: 0.761446710273182 F1: 0.782663744722909
--> Fold:  2
Train data shape: (40502, 1106) (40502,)
Val data shape: (14598, 1106) (14598,)
LightGBM Fold: 2 ACC: 0.7759967118783395 F1: 0.8362151092971636
LightGBM Avg ACC: 0.7475880482432844 Avg F1: 0.796548675913071
----------------------------------------------------------
(57971, 207)
feature_num=80
window_size=30
pca(0.95)
--> Fold:  0
Train data shape: (30650, 795) (30650,)
Val data shape: (24450, 795) (24450,)
LightGBM Fold: 0 ACC: 0.685521472392638 F1: 0.7683328601331642
--> Fold:  1
Train data shape: (39768, 821) (39768,)
Val data shape: (15332, 821) (15332,)
LightGBM Fold: 1 ACC: 0.8529872162796764 F1: 0.8666625642485422
--> Fold:  2
Train data shape: (39782, 818) (39782,)
Val data shape: (15318, 818) (15318,)
LightGBM Fold: 2 ACC: 0.7746442094268181 F1: 0.790077588125481
LightGBM Avg ACC: 0.7710509660330441 Avg F1: 0.8083576708357291
----------------------------------------------------------
(57971, 207)
feature_num=80
window_size=30
pca(0.99)
--------
--> Fold:  0
Train data shape: (34926, 1171) (34926,)
Val data shape: (20174, 1171) (20174,)
LightGBM Fold: 0 ACC: 0.7417963715673639 F1: 0.7802430929464826
XGBClassifier Fold: 0 ACC: 0.7548329533062358 F1: 0.7922649207645889
--> Fold:  1
Train data shape: (37658, 1181) (37658,)
Val data shape: (17442, 1181) (17442,)
LightGBM Fold: 1 ACC: 0.7622405687421168 F1: 0.7873281428340084
XGBClassifier Fold: 1 ACC: 0.768948515078546 F1: 0.7933916537595165
--> Fold:  2
Train data shape: (37616, 1180) (37616,)
Val data shape: (17484, 1180) (17484,)
LightGBM Fold: 2 ACC: 0.753317318691375 F1: 0.7983628038344416
XGBClassifier Fold: 2 ACC: 0.7507435369480668 F1: 0.7929036390384969
LightGBM Avg ACC: 0.752451419666952 Avg F1: 0.7886446798716443
XGBClassifier Avg ACC: 0.7581750017776162 Avg F1: 0.7928534045208675
--------------------------------------------------------------------------
ws=30
fn=100
pca=0.97
(57971, 207)
100 ['Temperature_value', 'Humidity_value', 'LightBarrierActiveTaskDuration1_vMax', 'LightBarrierActiveTaskDuration1_vFreq', 'SmartMotorSpeed_vTrend', 'DurationPickToPick_value', 'Pressure_vStd', 'VacuumFusePicked_vStd', 'EPOSVelocity_vStd', 'FusePicked_vMin', 'VacuumFusePicked_vTrend', 'TotalMemoryConsumption_vStd', 'IntensityTotalThermoImage_vCnt', 'ProcessCpuLoadNormalized_vMax', 'SmartMotorPositionError_vMax', 'TotalMemoryConsumption_vMin', 'TemperatureThermoCam_vFreq', 'ValidFrame_vFreq', 'LightBarrierPassiveTaskDuration1_vFreq', 'FuseHeatSlopeOK_vFreq', 'DurationRobotFromFeederToTestBench_value', 'FuseOutsideOperationalSpace_vMax', 'FuseHeatSlopeNOK_vMax', 'Vacuum_vStd', 'VacuumValveClosed_vTrend', 'IntensityTotalImage_vCnt', 'VacuumValveClosed_vMin', 'FeederAction1_vCnt', 'VacuumValveClosed_vCnt', 'FuseHeatSlopeNOK_value', 'IntensityTotalThermoImage_vFreq', 'ProcessMemoryConsumption_vMin', 'SharpnessImage_vFreq', 'FusePicked_vTrend', 'EPOSVelocity_vMin', 'LightBarrierPassiveTaskDuration1_value', 'CpuTemperature_vStd', 'EPOSCurrent_vFreq', 'EPOSPosition_vCnt', 'FuseHeatSlope_vFreq', 'EPOSCurrent_vTrend', 'IntensityTotalThermoImage_vMin', 'DurationRobotFromTestBenchToFeeder_vCnt', 'FusePicked_vMax', 'DurationTestBenchClosed_value', 'IntensityTotalThermoImage_vStd', 'DurationRobotFromFeederToTestBench_vStd', 'LightBarrierActiveTaskDuration1_vMin', 'LightBarrierPassiveTaskDuration1_vMax', 'TotalCpuLoadNormalized_vMin', 'Vacuum_vMin', 'LightBarrierActiveTaskDuration1_value', 'Vacuum_vMax', 'NumberFuseDetected_vCnt', 'DurationRobotFromFeederToTestBench_vMin', 'IntensityTotalThermoImage_vTrend', 'FuseHeatSlope_value', 'SmartMotorPositionError_vMin', 'Vacuum_vTrend', 'TemperatureThermoCam_vMax', 'IntensityTotalThermoImage_vMax', 'LightBarrierActiveTaskDuration1_vCnt', 'ProcessMemoryConsumption_vMax', 'VacuumFusePicked_value', 'CpuTemperature_value', 'DurationTestBenchClosed_vTrend', 'TemperatureThermoCam_vMin', 'FuseOutsideOperationalSpace_value', 'FuseHeatSlope_vCnt', 'SmartMotorSpeed_vFreq', 'TemperatureThermoCam_value', 'LightBarrierPassiveTaskDuration1_vMin', 'DurationRobotFromFeederToTestBench_vTrend', 'FuseCycleDuration_vMax', 'NumberFuseEstimated_vCnt', 'TemperatureThermoCam_vCnt', 'EPOSCurrent_vStd', 'FeederBackgroundIlluminationIntensity_vFreq', 'FeederAction3_vCnt', 'LightBarrierPassiveTaskDuration1_vTrend', 'SmartMotorPositionError_vTrend', 'EPOSVelocity_value', 'FuseHeatSlopeNOK_vFreq', 'EPOSPosition_vTrend', 'Pressure_vFreq', 'TotalCpuLoadNormalized_value', 'FuseCycleDuration_value', 'SharpnessImage_vCnt', 'DurationTestBenchClosed_vCnt', 'DurationRobotFromTestBenchToFeeder_vFreq', 'FuseHeatSlope_vMin', 'DurationPickToPick_vStd', 'DurationTestBenchClosed_vFreq', 'DurationRobotFromFeederToTestBench_vCnt', 'LightBarrierActiveTaskDuration1_vTrend', 'IntensityTotalImage_vFreq', 'FusePicked_vStd', 'SmartMotorSpeed_vCnt', 'SmartMotorPositionError_value', 'ProcessCpuLoadNormalized_vMin']
--> Fold:  0
Train data shape: (36348, 83) (36348,)
Val data shape: (18752, 83) (18752,)
LightGBM Fold: 0 ACC: 0.7303754266211604 F1: 0.7913115225440857
--> Fold:  1
Train data shape: (41205, 97) (41205,)
Val data shape: (13895, 97) (13895,)
LightGBM Fold: 1 ACC: 0.8214465635120547 F1: 0.8652902672751808
--> Fold:  2
Train data shape: (32647, 80) (32647,)
Val data shape: (22453, 80) (22453,)
LightGBM Fold: 2 ACC: 0.8076871687525052 F1: 0.8245889777849851
LightGBM Avg ACC: 0.7865030529619067 Avg F1: 0.8270635892014172
---------------------------------------------------
ws=30
fn=100
pca=0.95
(57971, 207)
(57971, 207)
100 ['Temperature_value', 'Humidity_value', 'LightBarrierActiveTaskDuration1_vMax', 'LightBarrierActiveTaskDuration1_vFreq', 'SmartMotorSpeed_vTrend', 'DurationPickToPick_value', 'Pressure_vStd', 'VacuumFusePicked_vStd', 'EPOSVelocity_vStd', 'FusePicked_vMin', 'VacuumFusePicked_vTrend', 'TotalMemoryConsumption_vStd', 'IntensityTotalThermoImage_vCnt', 'ProcessCpuLoadNormalized_vMax', 'SmartMotorPositionError_vMax', 'TotalMemoryConsumption_vMin', 'TemperatureThermoCam_vFreq', 'ValidFrame_vFreq', 'LightBarrierPassiveTaskDuration1_vFreq', 'FuseHeatSlopeOK_vFreq', 'DurationRobotFromFeederToTestBench_value', 'FuseOutsideOperationalSpace_vMax', 'FuseHeatSlopeNOK_vMax', 'Vacuum_vStd', 'VacuumValveClosed_vTrend', 'IntensityTotalImage_vCnt', 'VacuumValveClosed_vMin', 'FeederAction1_vCnt', 'VacuumValveClosed_vCnt', 'FuseHeatSlopeNOK_value', 'IntensityTotalThermoImage_vFreq', 'ProcessMemoryConsumption_vMin', 'SharpnessImage_vFreq', 'FusePicked_vTrend', 'EPOSVelocity_vMin', 'LightBarrierPassiveTaskDuration1_value', 'CpuTemperature_vStd', 'EPOSCurrent_vFreq', 'EPOSPosition_vCnt', 'FuseHeatSlope_vFreq', 'EPOSCurrent_vTrend', 'IntensityTotalThermoImage_vMin', 'DurationRobotFromTestBenchToFeeder_vCnt', 'FusePicked_vMax', 'DurationTestBenchClosed_value', 'IntensityTotalThermoImage_vStd', 'DurationRobotFromFeederToTestBench_vStd', 'LightBarrierActiveTaskDuration1_vMin', 'LightBarrierPassiveTaskDuration1_vMax', 'TotalCpuLoadNormalized_vMin', 'Vacuum_vMin', 'LightBarrierActiveTaskDuration1_value', 'Vacuum_vMax', 'NumberFuseDetected_vCnt', 'DurationRobotFromFeederToTestBench_vMin', 'IntensityTotalThermoImage_vTrend', 'FuseHeatSlope_value', 'SmartMotorPositionError_vMin', 'Vacuum_vTrend', 'TemperatureThermoCam_vMax', 'IntensityTotalThermoImage_vMax', 'LightBarrierActiveTaskDuration1_vCnt', 'ProcessMemoryConsumption_vMax', 'VacuumFusePicked_value', 'CpuTemperature_value', 'DurationTestBenchClosed_vTrend', 'TemperatureThermoCam_vMin', 'FuseOutsideOperationalSpace_value', 'FuseHeatSlope_vCnt', 'SmartMotorSpeed_vFreq', 'TemperatureThermoCam_value', 'LightBarrierPassiveTaskDuration1_vMin', 'DurationRobotFromFeederToTestBench_vTrend', 'FuseCycleDuration_vMax', 'NumberFuseEstimated_vCnt', 'TemperatureThermoCam_vCnt', 'EPOSCurrent_vStd', 'FeederBackgroundIlluminationIntensity_vFreq', 'FeederAction3_vCnt', 'LightBarrierPassiveTaskDuration1_vTrend', 'SmartMotorPositionError_vTrend', 'EPOSVelocity_value', 'FuseHeatSlopeNOK_vFreq', 'EPOSPosition_vTrend', 'Pressure_vFreq', 'TotalCpuLoadNormalized_value', 'FuseCycleDuration_value', 'SharpnessImage_vCnt', 'DurationTestBenchClosed_vCnt', 'DurationRobotFromTestBenchToFeeder_vFreq', 'FuseHeatSlope_vMin', 'DurationPickToPick_vStd', 'DurationTestBenchClosed_vFreq', 'DurationRobotFromFeederToTestBench_vCnt', 'LightBarrierActiveTaskDuration1_vTrend', 'IntensityTotalImage_vFreq', 'FusePicked_vStd', 'SmartMotorSpeed_vCnt', 'SmartMotorPositionError_value', 'ProcessCpuLoadNormalized_vMin']
--> Fold:  0
Train data shape: (36348, 83) (36348,)
Val data shape: (18752, 83) (18752,)
LightGBM Fold: 0 ACC: 0.7303754266211604 F1: 0.7913115225440857
XGBClassifier Fold: 0 ACC: 0.7301621160409556 F1: 0.7921230704690742
--> Fold:  1
Train data shape: (41205, 97) (41205,)
Val data shape: (13895, 97) (13895,)
LightGBM Fold: 1 ACC: 0.8214465635120547 F1: 0.8652902672751808
XGBClassifier Fold: 1 ACC: 0.8205829435048578 F1: 0.8684574053291902
--> Fold:  2
Train data shape: (32647, 80) (32647,)
Val data shape: (22453, 80) (22453,)
LightGBM Fold: 2 ACC: 0.8076871687525052 F1: 0.8245889777849851
XGBClassifier Fold: 2 ACC: 0.8058165946644101 F1: 0.8215861127892368
LightGBM Avg ACC: 0.7865030529619067 Avg F1: 0.8270635892014172
XGBClassifier Avg ACC: 0.7855205514 Avg F1: 0.82738886286
---------------------------------------------------
ws=30
fn=120
pca=0.95
(57971, 207)
120 ['Temperature_value', 'Humidity_value', 'LightBarrierActiveTaskDuration1_vMax', 'LightBarrierActiveTaskDuration1_vFreq', 'SmartMotorSpeed_vTrend', 'DurationPickToPick_value', 'Pressure_vStd', 'VacuumFusePicked_vStd', 'EPOSVelocity_vStd', 'FusePicked_vMin', 'VacuumFusePicked_vTrend', 'TotalMemoryConsumption_vStd', 'IntensityTotalThermoImage_vCnt', 'ProcessCpuLoadNormalized_vMax', 'SmartMotorPositionError_vMax', 'TotalMemoryConsumption_vMin', 'TemperatureThermoCam_vFreq', 'ValidFrame_vFreq', 'LightBarrierPassiveTaskDuration1_vFreq', 'FuseHeatSlopeOK_vFreq', 'DurationRobotFromFeederToTestBench_value', 'FuseOutsideOperationalSpace_vMax', 'FuseHeatSlopeNOK_vMax', 'Vacuum_vStd', 'VacuumValveClosed_vTrend', 'IntensityTotalImage_vCnt', 'VacuumValveClosed_vMin', 'FeederAction1_vCnt', 'VacuumValveClosed_vCnt', 'FuseHeatSlopeNOK_value', 'IntensityTotalThermoImage_vFreq', 'ProcessMemoryConsumption_vMin', 'SharpnessImage_vFreq', 'FusePicked_vTrend', 'EPOSVelocity_vMin', 'LightBarrierPassiveTaskDuration1_value', 'CpuTemperature_vStd', 'EPOSCurrent_vFreq', 'EPOSPosition_vCnt', 'FuseHeatSlope_vFreq', 'EPOSCurrent_vTrend', 'IntensityTotalThermoImage_vMin', 'DurationRobotFromTestBenchToFeeder_vCnt', 'FusePicked_vMax', 'DurationTestBenchClosed_value', 'IntensityTotalThermoImage_vStd', 'DurationRobotFromFeederToTestBench_vStd', 'LightBarrierActiveTaskDuration1_vMin', 'LightBarrierPassiveTaskDuration1_vMax', 'TotalCpuLoadNormalized_vMin', 'Vacuum_vMin', 'LightBarrierActiveTaskDuration1_value', 'Vacuum_vMax', 'NumberFuseDetected_vCnt', 'DurationRobotFromFeederToTestBench_vMin', 'IntensityTotalThermoImage_vTrend', 'FuseHeatSlope_value', 'SmartMotorPositionError_vMin', 'Vacuum_vTrend', 'TemperatureThermoCam_vMax', 'IntensityTotalThermoImage_vMax', 'LightBarrierActiveTaskDuration1_vCnt', 'ProcessMemoryConsumption_vMax', 'VacuumFusePicked_value', 'CpuTemperature_value', 'DurationTestBenchClosed_vTrend', 'TemperatureThermoCam_vMin', 'FuseOutsideOperationalSpace_value', 'FuseHeatSlope_vCnt', 'SmartMotorSpeed_vFreq', 'TemperatureThermoCam_value', 'LightBarrierPassiveTaskDuration1_vMin', 'DurationRobotFromFeederToTestBench_vTrend', 'FuseCycleDuration_vMax', 'NumberFuseEstimated_vCnt', 'TemperatureThermoCam_vCnt', 'EPOSCurrent_vStd', 'FeederBackgroundIlluminationIntensity_vFreq', 'FeederAction3_vCnt', 'LightBarrierPassiveTaskDuration1_vTrend', 'SmartMotorPositionError_vTrend', 'EPOSVelocity_value', 'FuseHeatSlopeNOK_vFreq', 'EPOSPosition_vTrend', 'Pressure_vFreq', 'TotalCpuLoadNormalized_value', 'FuseCycleDuration_value', 'SharpnessImage_vCnt', 'DurationTestBenchClosed_vCnt', 'DurationRobotFromTestBenchToFeeder_vFreq', 'FuseHeatSlope_vMin', 'DurationPickToPick_vStd', 'DurationTestBenchClosed_vFreq', 'DurationRobotFromFeederToTestBench_vCnt', 'LightBarrierActiveTaskDuration1_vTrend', 'IntensityTotalImage_vFreq', 'FusePicked_vStd', 'SmartMotorSpeed_vCnt', 'SmartMotorPositionError_value', 'ProcessCpuLoadNormalized_vMin', 'IntensityTotalThermoImage_value', 'FuseOutsideOperationalSpace_vFreq', 'TemperatureThermoCam_vTrend', 'DurationPickToPick_vTrend', 'FuseHeatSlope_vTrend', 'FuseTestResult_vTrend', 'DurationRobotFromTestBenchToFeeder_vMin', 'FuseTestResult_value', 'Vacuum_vFreq', 'ProcessMemoryConsumption_vStd', 'DurationRobotFromFeederToTestBench_vMax', 'VacuumValveClosed_vStd', 'EPOSPosition_vMax', 'EPOSPosition_vStd', 'DurationTestBenchClosed_vMin', 'Vacuum_vCnt', 'EPOSPosition_vMin', 'DurationRobotFromTestBenchToFeeder_vMax', 'FuseTestResult_vStd', 'FeederBackgroundIlluminationIntensity_vCnt']
--> Fold:  0
Train data shape: (36348, 127) (36348,)
Val data shape: (18752, 127) (18752,)
LightGBM Fold: 0 ACC: 0.7372546928327645 F1: 0.799684437557634
XGBClassifier Fold: 0 ACC: 0.7398677474402731 F1: 0.8030204674143362
--> Fold:  1
Train data shape: (41205, 135) (41205,)
Val data shape: (13895, 135) (13895,)
LightGBM Fold: 1 ACC: 0.8539762504498021 F1: 0.8860453933046025
XGBClassifier Fold: 1 ACC: 0.8568549838071249 F1: 0.8882405204255149
--> Fold:  2
Train data shape: (32647, 122) (32647,)
Val data shape: (22453, 122) (22453,)
LightGBM Fold: 2 ACC: 0.8171291141495568 F1: 0.8358482647342405
XGBClassifier Fold: 2 ACC: 0.8175299514541486 F1: 0.8359143462070258
LightGBM Avg ACC: 0.8027866858107079 Avg F1: 0.8405260318654925
XGBClassifier Avg ACC: 0.8047508942338489 Avg F1: 0.8423917780156257
----------------------------------------------------------------------
ws=30
fn=140
pca=0.98
(57971, 207)
140 ['Temperature_value', 'Humidity_value', 'LightBarrierActiveTaskDuration1_vMax', 'LightBarrierActiveTaskDuration1_vFreq', 'SmartMotorSpeed_vTrend', 'DurationPickToPick_value', 'Pressure_vStd', 'VacuumFusePicked_vStd', 'EPOSVelocity_vStd', 'FusePicked_vMin', 'VacuumFusePicked_vTrend', 'TotalMemoryConsumption_vStd', 'IntensityTotalThermoImage_vCnt', 'ProcessCpuLoadNormalized_vMax', 'SmartMotorPositionError_vMax', 'TotalMemoryConsumption_vMin', 'TemperatureThermoCam_vFreq', 'ValidFrame_vFreq', 'LightBarrierPassiveTaskDuration1_vFreq', 'FuseHeatSlopeOK_vFreq', 'DurationRobotFromFeederToTestBench_value', 'FuseOutsideOperationalSpace_vMax', 'FuseHeatSlopeNOK_vMax', 'Vacuum_vStd', 'VacuumValveClosed_vTrend', 'IntensityTotalImage_vCnt', 'VacuumValveClosed_vMin', 'FeederAction1_vCnt', 'VacuumValveClosed_vCnt', 'FuseHeatSlopeNOK_value', 'IntensityTotalThermoImage_vFreq', 'ProcessMemoryConsumption_vMin', 'SharpnessImage_vFreq', 'FusePicked_vTrend', 'EPOSVelocity_vMin', 'LightBarrierPassiveTaskDuration1_value', 'CpuTemperature_vStd', 'EPOSCurrent_vFreq', 'EPOSPosition_vCnt', 'FuseHeatSlope_vFreq', 'EPOSCurrent_vTrend', 'IntensityTotalThermoImage_vMin', 'DurationRobotFromTestBenchToFeeder_vCnt', 'FusePicked_vMax', 'DurationTestBenchClosed_value', 'IntensityTotalThermoImage_vStd', 'DurationRobotFromFeederToTestBench_vStd', 'LightBarrierActiveTaskDuration1_vMin', 'LightBarrierPassiveTaskDuration1_vMax', 'TotalCpuLoadNormalized_vMin', 'Vacuum_vMin', 'LightBarrierActiveTaskDuration1_value', 'Vacuum_vMax', 'NumberFuseDetected_vCnt', 'DurationRobotFromFeederToTestBench_vMin', 'IntensityTotalThermoImage_vTrend', 'FuseHeatSlope_value', 'SmartMotorPositionError_vMin', 'Vacuum_vTrend', 'TemperatureThermoCam_vMax', 'IntensityTotalThermoImage_vMax', 'LightBarrierActiveTaskDuration1_vCnt', 'ProcessMemoryConsumption_vMax', 'VacuumFusePicked_value', 'CpuTemperature_value', 'DurationTestBenchClosed_vTrend', 'TemperatureThermoCam_vMin', 'FuseOutsideOperationalSpace_value', 'FuseHeatSlope_vCnt', 'SmartMotorSpeed_vFreq', 'TemperatureThermoCam_value', 'LightBarrierPassiveTaskDuration1_vMin', 'DurationRobotFromFeederToTestBench_vTrend', 'FuseCycleDuration_vMax', 'NumberFuseEstimated_vCnt', 'TemperatureThermoCam_vCnt', 'EPOSCurrent_vStd', 'FeederBackgroundIlluminationIntensity_vFreq', 'FeederAction3_vCnt', 'LightBarrierPassiveTaskDuration1_vTrend', 'SmartMotorPositionError_vTrend', 'EPOSVelocity_value', 'FuseHeatSlopeNOK_vFreq', 'EPOSPosition_vTrend', 'Pressure_vFreq', 'TotalCpuLoadNormalized_value', 'FuseCycleDuration_value', 'SharpnessImage_vCnt', 'DurationTestBenchClosed_vCnt', 'DurationRobotFromTestBenchToFeeder_vFreq', 'FuseHeatSlope_vMin', 'DurationPickToPick_vStd', 'DurationTestBenchClosed_vFreq', 'DurationRobotFromFeederToTestBench_vCnt', 'LightBarrierActiveTaskDuration1_vTrend', 'IntensityTotalImage_vFreq', 'FusePicked_vStd', 'SmartMotorSpeed_vCnt', 'SmartMotorPositionError_value', 'ProcessCpuLoadNormalized_vMin', 'IntensityTotalThermoImage_value', 'FuseOutsideOperationalSpace_vFreq', 'TemperatureThermoCam_vTrend', 'DurationPickToPick_vTrend', 'FuseHeatSlope_vTrend', 'FuseTestResult_vTrend', 'DurationRobotFromTestBenchToFeeder_vMin', 'FuseTestResult_value', 'Vacuum_vFreq', 'ProcessMemoryConsumption_vStd', 'DurationRobotFromFeederToTestBench_vMax', 'VacuumValveClosed_vStd', 'EPOSPosition_vMax', 'EPOSPosition_vStd', 'DurationTestBenchClosed_vMin', 'Vacuum_vCnt', 'EPOSPosition_vMin', 'DurationRobotFromTestBenchToFeeder_vMax', 'FuseTestResult_vStd', 'FeederBackgroundIlluminationIntensity_vCnt', 'ErrorFrame_vCnt', 'SmartMotorPositionError_vStd', 'SmartMotorSpeed_value', 'FuseOutsideOperationalSpace_vStd', 'LightBarrierActiveTaskDuration1_vStd', 'DurationPickToPick_vCnt', 'VacuumValveClosed_vMax', 'FuseTestResult_vMin', 'EPOSVelocity_vCnt', 'EPOSCurrent_vMax', 'TotalCpuLoadNormalized_vStd', 'EPOSCurrent_vCnt', 'NumberEmptyFeeder_vCnt', 'ProcessCpuLoadNormalized_value', 'FuseCycleDuration_vStd', 'EPOSCurrent_value', 'FuseCycleDuration_vMin', 'FusePicked_vCnt', 'DurationRobotFromTestBenchToFeeder_vTrend', 'FusePicked_value']
--> Fold:  0
Train data shape: (36348, 144) (36348,)
Val data shape: (18752, 144) (18752,)
LightGBM Fold: 0 ACC: 0.7429074232081911 F1: 0.8054663455263565
XGBClassifier Fold: 0 ACC: 0.742320819112628 F1: 0.8069480819214347
--> Fold:  1
Train data shape: (41205, 150) (41205,)
Val data shape: (13895, 150) (13895,)
LightGBM Fold: 1 ACC: 0.8473551637279597 F1: 0.8806227442483875
XGBClassifier Fold: 1 ACC: 0.8474991003958259 F1: 0.8816524709431058
--> Fold:  2
Train data shape: (32647, 139) (32647,)
Val data shape: (22453, 139) (22453,)
LightGBM Fold: 2 ACC: 0.8169509642364049 F1: 0.8362140906197298
XGBClassifier Fold: 2 ACC: 0.8176635638890126 F1: 0.8369286430586402
LightGBM Avg ACC: 0.8024045170575186 Avg F1: 0.840767726798158
XGBClassifier Avg ACC: 0.8024944944658222 Avg F1: 0.8418430653077268
----------------------------------------------------------------------
ws=30
fn=130
pca=0.97
(57971, 207)
130 ['Temperature_value', 'Humidity_value', 'LightBarrierActiveTaskDuration1_vMax', 'LightBarrierActiveTaskDuration1_vFreq', 'SmartMotorSpeed_vTrend', 'DurationPickToPick_value', 'Pressure_vStd', 'VacuumFusePicked_vStd', 'EPOSVelocity_vStd', 'FusePicked_vMin', 'VacuumFusePicked_vTrend', 'TotalMemoryConsumption_vStd', 'IntensityTotalThermoImage_vCnt', 'ProcessCpuLoadNormalized_vMax', 'SmartMotorPositionError_vMax', 'TotalMemoryConsumption_vMin', 'TemperatureThermoCam_vFreq', 'ValidFrame_vFreq', 'LightBarrierPassiveTaskDuration1_vFreq', 'FuseHeatSlopeOK_vFreq', 'DurationRobotFromFeederToTestBench_value', 'FuseOutsideOperationalSpace_vMax', 'FuseHeatSlopeNOK_vMax', 'Vacuum_vStd', 'VacuumValveClosed_vTrend', 'IntensityTotalImage_vCnt', 'VacuumValveClosed_vMin', 'FeederAction1_vCnt', 'VacuumValveClosed_vCnt', 'FuseHeatSlopeNOK_value', 'IntensityTotalThermoImage_vFreq', 'ProcessMemoryConsumption_vMin', 'SharpnessImage_vFreq', 'FusePicked_vTrend', 'EPOSVelocity_vMin', 'LightBarrierPassiveTaskDuration1_value', 'CpuTemperature_vStd', 'EPOSCurrent_vFreq', 'EPOSPosition_vCnt', 'FuseHeatSlope_vFreq', 'EPOSCurrent_vTrend', 'IntensityTotalThermoImage_vMin', 'DurationRobotFromTestBenchToFeeder_vCnt', 'FusePicked_vMax', 'DurationTestBenchClosed_value', 'IntensityTotalThermoImage_vStd', 'DurationRobotFromFeederToTestBench_vStd', 'LightBarrierActiveTaskDuration1_vMin', 'LightBarrierPassiveTaskDuration1_vMax', 'TotalCpuLoadNormalized_vMin', 'Vacuum_vMin', 'LightBarrierActiveTaskDuration1_value', 'Vacuum_vMax', 'NumberFuseDetected_vCnt', 'DurationRobotFromFeederToTestBench_vMin', 'IntensityTotalThermoImage_vTrend', 'FuseHeatSlope_value', 'SmartMotorPositionError_vMin', 'Vacuum_vTrend', 'TemperatureThermoCam_vMax', 'IntensityTotalThermoImage_vMax', 'LightBarrierActiveTaskDuration1_vCnt', 'ProcessMemoryConsumption_vMax', 'VacuumFusePicked_value', 'CpuTemperature_value', 'DurationTestBenchClosed_vTrend', 'TemperatureThermoCam_vMin', 'FuseOutsideOperationalSpace_value', 'FuseHeatSlope_vCnt', 'SmartMotorSpeed_vFreq', 'TemperatureThermoCam_value', 'LightBarrierPassiveTaskDuration1_vMin', 'DurationRobotFromFeederToTestBench_vTrend', 'FuseCycleDuration_vMax', 'NumberFuseEstimated_vCnt', 'TemperatureThermoCam_vCnt', 'EPOSCurrent_vStd', 'FeederBackgroundIlluminationIntensity_vFreq', 'FeederAction3_vCnt', 'LightBarrierPassiveTaskDuration1_vTrend', 'SmartMotorPositionError_vTrend', 'EPOSVelocity_value', 'FuseHeatSlopeNOK_vFreq', 'EPOSPosition_vTrend', 'Pressure_vFreq', 'TotalCpuLoadNormalized_value', 'FuseCycleDuration_value', 'SharpnessImage_vCnt', 'DurationTestBenchClosed_vCnt', 'DurationRobotFromTestBenchToFeeder_vFreq', 'FuseHeatSlope_vMin', 'DurationPickToPick_vStd', 'DurationTestBenchClosed_vFreq', 'DurationRobotFromFeederToTestBench_vCnt', 'LightBarrierActiveTaskDuration1_vTrend', 'IntensityTotalImage_vFreq', 'FusePicked_vStd', 'SmartMotorSpeed_vCnt', 'SmartMotorPositionError_value', 'ProcessCpuLoadNormalized_vMin', 'IntensityTotalThermoImage_value', 'FuseOutsideOperationalSpace_vFreq', 'TemperatureThermoCam_vTrend', 'DurationPickToPick_vTrend', 'FuseHeatSlope_vTrend', 'FuseTestResult_vTrend', 'DurationRobotFromTestBenchToFeeder_vMin', 'FuseTestResult_value', 'Vacuum_vFreq', 'ProcessMemoryConsumption_vStd', 'DurationRobotFromFeederToTestBench_vMax', 'VacuumValveClosed_vStd', 'EPOSPosition_vMax', 'EPOSPosition_vStd', 'DurationTestBenchClosed_vMin', 'Vacuum_vCnt', 'EPOSPosition_vMin', 'DurationRobotFromTestBenchToFeeder_vMax', 'FuseTestResult_vStd', 'FeederBackgroundIlluminationIntensity_vCnt', 'ErrorFrame_vCnt', 'SmartMotorPositionError_vStd', 'SmartMotorSpeed_value', 'FuseOutsideOperationalSpace_vStd', 'LightBarrierActiveTaskDuration1_vStd', 'DurationPickToPick_vCnt', 'VacuumValveClosed_vMax', 'FuseTestResult_vMin', 'EPOSVelocity_vCnt', 'EPOSCurrent_vMax']
--> Fold:  0
Train data shape: (36348, 128) (36348,)
Val data shape: (18752, 128) (18752,)
LightGBM Fold: 0 ACC: 0.7444539249146758 F1: 0.8066436424143295
XGBClassifier Fold: 0 ACC: 0.742054180887372 F1: 0.8065337398251646
--> Fold:  1
Train data shape: (41205, 136) (41205,)
Val data shape: (13895, 136) (13895,)
LightGBM Fold: 1 ACC: 0.8451241453760345 F1: 0.8797365875488538
XGBClassifier Fold: 1 ACC: 0.8454120187117669 F1: 0.8801453705175158
--> Fold:  2
Train data shape: (32647, 123) (32647,)
Val data shape: (22453, 123) (22453,)
LightGBM Fold: 2 ACC: 0.8165055894535251 F1: 0.8362041138876951
XGBClassifier Fold: 2 ACC: 0.8177526388455886 F1: 0.8371496123441562
LightGBM Avg ACC: 0.8020278865814118 Avg F1: 0.8408614479502928
XGBClassifier Avg ACC: 0.8017396128149091 Avg F1: 0.8412762408956121

"""