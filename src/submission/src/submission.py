import warnings

warnings.filterwarnings("ignore")
import numpy as np
from numpy import nan
import pandas as pd

import pickle
import collections
from collections import defaultdict

from matplotlib import pyplot as plt
from os import listdir
from os.path import isfile, join
import seaborn as sns
import re

import random
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import KFold
from lofo import LOFOImportance, Dataset, plot_importance
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, recall_score, precision_score, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import ExtraTreeClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.svm import SVC
import pickle


data_df = pd.read_csv("../data/df_cleaned.csv")
df = data_df.copy()

df['runId'] = 1000 * df['class'] + df['run']
run_df = df[['class', 'runId']].copy()
run_df.drop_duplicates(inplace=True)
run_df.reset_index(inplace=True)
run_df_ = df['run'].copy()
# run_df = run_df.sample(frac=1, random_state=14).reset_index(drop=True)
del run_df['index'], df['run']

# fold_num = 3
# cv = StratifiedKFold(n_splits=fold_num, shuffle=True, random_state=41)
#
# df_ = df.copy(deep=True)
# del df_['runId']
# # df_ = df_.sample(frac=1, random_state=0)
# dataset = Dataset(df=df_, target="class", features=[col for col in df_.columns if col != "class"])
#
# model = LGBMClassifier(random_state=41)
# lofo_imp = LOFOImportance(dataset, cv=cv, model=model, scoring="balanced_accuracy")
#
# # get the mean and standard deviation of the importances in pandas format
# importance_df = lofo_imp.get_importance()
# importance_df.to_pickle("../data/importance_df_ba.pkl")
#
# plot_importance(importance_df, figsize=(12, 20))
# plt.savefig("importance_df_f1.png")
# plt.show()

importance_df = pd.read_pickle("../data/importance_df_ba.pkl")
importance_df = importance_df.loc[(importance_df["importance_mean"] > 0)]
sorted_features_imp = list(importance_df["feature"].values)
value_features_imp = list(importance_df["importance_mean"].values)

f_imp = [(name, value) for name, value in zip(sorted_features_imp, value_features_imp)]

print("fold_num:", len(f_imp), f_imp)
print(sorted_features_imp)

# region create dataset
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
    seq = seq.reshape((seq_count, -1))  # for 1D

    # add new seq to data_data array
    # data_data = np.vstack((data_data, seq))

    # Calculate RULS
    labels = df['class'].values[:seq_count]

    # add rul to rul_data array
    #     rul_data = np.vstack((rul_data, ruls))

    # TODO: What is RUL_Max in this context?

    # print ("Shape:", seq.shape, labels.shape)

    return seq, labels

# TODO: X_t, X_tp1, y_t, y_tp1 should be calculated per run.
# TODO: Then should be merged into one X_t, X_tp1, y_t, y_tp1.

def create_datasets(df, ws):
    run_list = df['runId'].unique()
    l_len_runs = []

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
        l_len_runs.append(len(X_t))

    X_t = pd.concat(X_df_list, axis=0)  # Merge data frames
    y_t = pd.concat(y_df_list, axis=0)  # Merge data frames

    return X_t.values, y_t.values.flatten(), run_list, l_len_runs

# endregion

fold_num = 3
cv = StratifiedKFold(n_splits=fold_num, shuffle=True, random_state=41)

df_report = pd.DataFrame()
sensor_list = sorted_features_imp[:len(sorted_features_imp)].copy()

df_result_ave = pd.DataFrame()

for ws in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]:

    acc_sum_1 = 0
    f1_sum_1 = 0
    mcc_sum_1 = 0

    acc_sum_4 = 0
    f1_sum_4 = 0
    mcc_sum_4 = 0

    print("_______________________________________________________________")

    for fold, (training_indices, validation_indices) in enumerate(cv.split(run_df['runId'], run_df['class'])):
        print("----------------------------------------------------------")

        report_index = "ws_{}_fold_{}".format(ws, fold + 1)
        print("Fold: ", report_index)

        training_runIds = run_df.loc[training_indices]['runId']
        validation_runIds = run_df.loc[validation_indices]['runId']

        # df_ = df.loc[~(df["runId"].isin([56, 74, 49, 23, 35, 6, 83, 54]))]
        df_ = df.loc[~(df["runId"].isin([56, 74, 49, 23, 35, 54]))]
        # df_ = df.copy()

        X_train_df = df_[df_['runId'].isin(training_runIds)].copy()
        X_val_df = df_[df_['runId'].isin(validation_runIds)].copy()

        X_train_df = X_train_df[sensor_list + ["class", "runId"]].copy()
        X_val_df = X_val_df[sensor_list + ["class", "runId"]].copy()

        scaler_cols = sensor_list.copy()  # list(set(sensor_list).difference(["class", "runId"]))

        scaler = RobustScaler()
        scaler_data_tr = scaler.fit_transform(X_train_df[scaler_cols])
        scaler_data_tr = pd.DataFrame(scaler_data_tr, index=X_train_df.index, columns=scaler_cols)
        X_train_df = pd.concat([X_train_df[["class", "runId"]], scaler_data_tr], axis=1)

        scaler_data_ts = scaler.transform(X_val_df[scaler_cols])
        scaler_data_ts = pd.DataFrame(scaler_data_ts, index=X_val_df.index, columns=scaler_cols)
        X_val_df = pd.concat([X_val_df[["class", "runId"]], scaler_data_ts], axis=1)

        X_train, y_train, runList_tr, l_len_runs_tr = create_datasets(X_train_df, ws)
        X_val, y_val, runList_val, l_len_runs_val = create_datasets(X_val_df, ws)

        lda = LinearDiscriminantAnalysis()
        X_train = lda.fit_transform(X_train, y_train)
        X_val = lda.transform(X_val)

        # model1 = OneVsOneClassifier(LGBMClassifier(random_state=41))
        model1 = LGBMClassifier(random_state=41)
        model1.fit(X_train, y_train)
        pred = model1.predict(X_val)

        acc_val1 = round(accuracy_score(y_val, pred), 3)
        f1_val1 = round(f1_score(y_val, pred, average='weighted'), 3)
        mcc_val1 = round(matthews_corrcoef(y_val, pred), 3)
        cm1 = confusion_matrix(y_val, pred)

        l_index = []
        for run_, num_run in zip(runList_val, l_len_runs_val):
            l_index.extend([run_] * num_run)

        # # 56(4), 74(4), 49(4); 7000
        # # 23(4), 35(4), 6(7 ama az), 83(iyi); 9001
        # # 54(9)
        # print(len(l_index), len(y_val))
        df_result = pd.DataFrame()
        df_result["actual"] = y_val
        df_result["pred"] = pred
        df_result.loc[df_result.index[0], "cm"] = str(cm1)
        df_result.index = l_index

        df_result.to_excel("../data/results/excels/lgb_{}_mcc.xlsx".format(report_index))
        with open('../data/results/models/lgb_{}_mcc.pickle'.format(report_index), 'wb') as handle:
            pickle.dump(model1, handle, protocol=pickle.HIGHEST_PROTOCOL)

        acc_sum_1 += acc_val1
        f1_sum_1 += f1_val1
        mcc_sum_1 += mcc_val1

        print(cm1)
        print("LightGBM Fold:", fold, "ACC:", acc_val1, "F1:", f1_val1, "MCC:", mcc_val1)
        df_report.loc[report_index, "LGBM_ACC"] = acc_val1
        df_report.loc[report_index, "LGBM_F1"] = f1_val1
        df_report.loc[report_index, "LGBM_MCC"] = mcc_val1

        param = {
            'objective': 'multi:softprob',  # error evaluation for multiclass training
            'num_class': len(pd.unique(y_train))}  # the number of classes that exist in this datset
        # model4 = OneVsOneClassifier(XGBClassifier(param, random_state=41, verbosity=0))
        model4 = XGBClassifier(param, random_state=41, verbosity=0)
        model4.fit(X_train, y_train)
        pred = model4.predict(X_val)

        acc_val4 = round(accuracy_score(y_val, pred), 3)
        f1_val4 = round(f1_score(y_val, pred, average='weighted'), 3)
        mcc_val4 = round(matthews_corrcoef(y_val, pred), 3)
        cm4 = confusion_matrix(y_val, pred)

        acc_sum_4 += acc_val4
        f1_sum_4 += f1_val4
        mcc_sum_4 += mcc_val4

        print(cm4)
        print("XGBClassifier Fold:", fold, "ACC:", acc_val4, "F1:", f1_val4, "MCC:", mcc_val4)
        df_report.loc[report_index, "XGB_ACC"] = acc_val4
        df_report.loc[report_index, "XGB_F1"] = f1_val4
        df_report.loc[report_index, "XGB_MCC"] = mcc_val4

        df_result = pd.DataFrame()
        df_result["actual"] = y_val
        df_result["pred"] = pred
        df_result.loc[df_result.index[0], "cm"] = str(cm4)
        df_result.index = l_index
        df_result.to_excel("../data/results/excels/xgb_{}_mcc.xlsx".format(report_index))
        with open('../data/results/models/xgb_{}_mcc.pickle'.format(report_index), 'wb') as handle:
            pickle.dump(model4, handle, protocol=pickle.HIGHEST_PROTOCOL)

    ave_acc1 = round(acc_sum_1 / fold_num, 3)
    ave_f1_score1 = round(f1_sum_1 / fold_num, 3)
    ave_mcc1 = round(mcc_sum_1 / fold_num, 3)

    ave_acc4 = round(acc_sum_4 / fold_num, 3)
    ave_f1_score4 = round(f1_sum_4 / fold_num, 3)
    ave_mcc4 = round(mcc_sum_4 / fold_num, 3)

    print("\nLightGBM Avg ACC:", ave_acc1, "Avg F1:", ave_f1_score1, "Avg MCC:", ave_mcc1)
    print("XGBM Avg ACC:", ave_acc4, "Avg F1:", ave_f1_score4, "Avg MCC:", ave_mcc4)

    df_result_ave.loc[ws, "LGBM_ACC"] = ave_acc1
    df_result_ave.loc[ws, "LGBM_F1"] = ave_f1_score1
    df_result_ave.loc[ws, "LGBM_MCC"] = ave_mcc1

    df_result_ave.loc[ws, "XGB_ACC"] = ave_acc4
    df_result_ave.loc[ws, "XGB_F1"] = ave_f1_score4
    df_result_ave.loc[ws, "XGB_MCC"] = ave_mcc4

df_result_ave.to_excel("../data/results/reports/_ws_all_mcc.xlsx".format(ws))
df_report.to_excel("../data/results/reports/_report_all_mcc.xlsx")

"""
output
fold_num: 141 [('Temperature_value', 0.0019902063851064966), ('Humidity_value', 0.0018532015234764554), ('DurationTestBenchClosed_vFreq', 0.0003369399383144067), ('FuseHeatSlopeNOK_vMin', 0.0003246559842096097), ('Pressure_vMax', 0.00031569883180659897), ('FeederAction4_vCnt', 0.0003038113676794296), ('ProcessCpuLoadNormalized_value', 0.0002998769695838899), ('Vacuum_vFreq', 0.00026840750607130354), ('TotalMemoryConsumption_vMin', 0.00025607796982448267), ('CpuTemperature_vStd', 0.0002551601247079797), ('SmartMotorPositionError_vMax', 0.00025388009129471517), ('FuseCycleDuration_vMax', 0.0002537349646733282), ('LightBarrierActiveTaskDuration1_vTrend', 0.0002526432451493976), ('DurationRobotFromFeederToTestBench_vMin', 0.0002514328930613985), ('SmartMotorSpeed_vMax', 0.00025006732933141596), ('FusePicked_vFreq', 0.0002468930035120683), ('DurationPickToPick_vFreq', 0.0002429966063576261), ('DurationTestBenchClosed_value', 0.00023162248143426822), ('DurationPickToPick_vTrend', 0.00022991688075553926), ('VacuumValveClosed_vStd', 0.0002278364646743173), ('LightBarrierPassiveTaskDuration1_vCnt', 0.0002113077447663351), ('DurationRobotFromTestBenchToFeeder_vMin', 0.00021089521429070354), ('NumberFuseEstimated_vFreq', 0.00021000310987943452), ('VacuumFusePicked_vFreq', 0.00020638381623442706), ('FuseTestResult_vStd', 0.0002025920312100915), ('EPOSCurrent_vStd', 0.00019987989352643343), ('DurationTestBenchClosed_vMin', 0.00019751114576549433), ('VacuumFusePicked_vMin', 0.0001955902296841098), ('TemperatureThermoCam_value', 0.00019347355722561707), ('SmartMotorPositionError_vFreq', 0.00019337526256639018), ('LightBarrierActiveTaskDuration1_vMax', 0.0001906389318714711), ('IntensityTotalThermoImage_vMax', 0.00018459395529764797), ('DurationRobotFromFeederToTestBench_vStd', 0.0001828924869330765), ('TemperatureThermoCam_vTrend', 0.00018033475374904887), ('EPOSVelocity_vStd', 0.00017564308494588676), ('ProcessMemoryConsumption_vStd', 0.000175562709344231), ('VacuumFusePicked_vCnt', 0.00016936208946483067), ('FuseHeatSlopeNOK_vFreq', 0.00016924089298290626), ('FuseCycleDuration_vCnt', 0.00016897874488963627), ('Vacuum_vTrend', 0.0001668628205809449), ('LightBarrierActiveTaskDuration1_vStd', 0.00016344881532448507), ('FuseCycleDuration_vMin', 0.00015641334995023884), ('DurationRobotFromTestBenchToFeeder_vTrend', 0.0001558708639812408), ('FuseHeatSlope_vMax', 0.00015522872264450704), ('FusePicked_vMin', 0.00015412100135118548), ('DurationTestBenchClosed_vCnt', 0.00015400032040705872), ('SmartMotorPositionError_vTrend', 0.00015184663788958552), ('SmartMotorSpeed_vFreq', 0.00015171044828965474), ('Vacuum_vMax', 0.00015086048479299544), ('Pressure_vMin', 0.00014976509717475514), ('Vacuum_value', 0.00014859663399028703), ('VacuumFusePicked_value', 0.00014859647458638955), ('Pressure_vFreq', 0.0001454141930485875), ('FuseOutsideOperationalSpace_vCnt', 0.00014495653877259299), ('FuseHeatSlopeNOK_vMax', 0.00014403209340226505), ('EPOSVelocity_vTrend', 0.00014343248734016711), ('NumberFuseEstimated_vCnt', 0.00014247194952495454), ('Vacuum_vStd', 0.00013560571568186722), ('EPOSCurrent_vTrend', 0.00013131333628339328), ('FuseOutsideOperationalSpace_vFreq', 0.00013054325546812962), ('EPOSVelocity_vFreq', 0.00012890432487947892), ('VacuumValveClosed_value', 0.0001269451942923494), ('LightBarrierActiveTaskDuration1_vCnt', 0.00012304988960687435), ('FuseHeatSlope_value', 0.00012166527949897255), ('ValidFrame_vCnt', 0.00012002818482133766), ('Vacuum_vCnt', 0.00011369144978660521), ('TotalMemoryConsumption_vMax', 0.00010934197642198384), ('Vacuum_vMin', 0.00010894201382620021), ('TotalMemoryConsumption_vStd', 0.00010836229631590773), ('ProcessCpuLoadNormalized_vMax', 0.0001075538992859985), ('EPOSCurrent_vMax', 0.00010726205359971412), ('TotalCpuLoadNormalized_vStd', 0.00010449323426172293), ('ValidFrameOptrisPIIRCamera_vFreq', 0.0001042798385886649), ('FusePicked_vTrend', 0.00010248183330363592), ('DurationRobotFromTestBenchToFeeder_vFreq', 9.540763123080016e-05), ('FuseCycleDuration_value', 9.422284452117087e-05), ('EPOSCurrent_vMin', 9.25199599231356e-05), ('VacuumValveClosed_vMin', 9.082074255643373e-05), ('FeederAction2_vCnt', 9.077574706991183e-05), ('DurationPickToPick_vMax', 9.024403621937793e-05), ('FuseCycleDuration_vFreq', 8.97232641013564e-05), ('LightBarrierPassiveTaskDuration1_vStd', 8.852998551767133e-05), ('NumberFuseDetected_vFreq', 8.324511957954102e-05), ('DurationTestBenchClosed_vTrend', 8.209692107822668e-05), ('LightBarrierActiveTaskDuration1_vMin', 8.117080246368887e-05), ('DurationRobotFromFeederToTestBench_vMax', 7.921412641809233e-05), ('FusePicked_vCnt', 7.708020195011193e-05), ('EPOSVelocity_vCnt', 7.692011845698381e-05), ('DurationPickToPick_vCnt', 7.515076832701843e-05), ('FuseHeatSlope_vMin', 7.376624544460271e-05), ('SmartMotorPositionError_vStd', 7.249168009778224e-05), ('SmartMotorPositionError_vMin', 7.243653908898622e-05), ('EPOSVelocity_vMax', 6.976203892757511e-05), ('EPOSPosition_vMin', 6.929612299044763e-05), ('TemperatureThermoCam_vStd', 6.865407100457392e-05), ('DurationRobotFromFeederToTestBench_vFreq', 6.739187843407546e-05), ('FuseHeatSlope_vTrend', 6.400063864749524e-05), ('EPOSVelocity_vMin', 6.169977250643684e-05), ('SmartMotorSpeed_value', 6.06677091120611e-05), ('LightBarrierActiveTaskDuration1_value', 6.0654805596064655e-05), ('FuseTestResult_value', 5.908626538231321e-05), ('VacuumFusePicked_vTrend', 5.628915212637272e-05), ('FusePicked_vStd', 5.419274176532429e-05), ('EPOSPosition_vTrend', 4.5566468468353584e-05), ('VacuumValveClosed_vFreq', 4.5486464920307945e-05), ('LightBarrierPassiveTaskDuration1_vTrend', 4.2736178931357394e-05), ('ValidFrameOptrisPIIRCamera_vCnt', 4.2584082507087416e-05), ('DurationRobotFromTestBenchToFeeder_vCnt', 4.156254564916898e-05), ('LightBarrierActiveTaskDuration1_vFreq', 4.01814944014367e-05), ('FuseTestResult_vFreq', 3.8940522502935636e-05), ('EPOSPosition_vFreq', 3.8788672708160256e-05), ('IntensityTotalImage_vFreq', 3.8540100975000456e-05), ('EPOSCurrent_vFreq', 3.758081761828791e-05), ('FeederBackgroundIlluminationIntensity_vCnt', 3.3792081248096295e-05), ('DurationRobotFromTestBenchToFeeder_value', 3.3759796039145584e-05), ('EPOSCurrent_value', 3.345813665289743e-05), ('DurationPickToPick_vStd', 3.124930158030873e-05), ('SmartMotorSpeed_vTrend', 3.0539701684132815e-05), ('FuseCycleDuration_vStd', 3.0289469673723996e-05), ('DurationRobotFromFeederToTestBench_vCnt', 2.9941191305911847e-05), ('FuseHeatSlope_vStd', 2.9753656959452535e-05), ('Pressure_vCnt', 2.9714680624774264e-05), ('FuseIntoFeeder_vCnt', 2.8676733447604203e-05), ('ValidFrame_vFreq', 2.7731399279029567e-05), ('ProcessMemoryConsumption_vMin', 2.4962108784739218e-05), ('TotalCpuLoadNormalized_vMax', 2.47841468034767e-05), ('DurationTestBenchClosed_vStd', 2.312788221100111e-05), ('DurationPickToPick_value', 2.292873515785665e-05), ('CpuTemperature_vMin', 1.8668700948798467e-05), ('LightBarrierPassiveTaskDuration1_vFreq', 1.5705318908820765e-05), ('DurationPickToPick_vMin', 1.2447849874826685e-05), ('CpuTemperature_vMax', 9.50009593699471e-06), ('VacuumValveClosed_vMax', 7.848547736906871e-06), ('CpuTemperature_value', 6.716098701361102e-06), ('TotalCpuLoadNormalized_vMin', 6.124931094790102e-06), ('DurationTestBenchClosed_vMax', 5.738194516028787e-06), ('VacuumFusePicked_vStd', 4.674742853187179e-06), ('ProcessMemoryConsumption_vMax', 4.096746876218009e-06), ('EPOSCurrent_vCnt', 3.0224446741315227e-06), ('Pressure_value', 2.7301057287839234e-06), ('ProcessMemoryConsumption_value', 4.2819469011264505e-07)]
['Temperature_value', 'Humidity_value', 'DurationTestBenchClosed_vFreq', 'FuseHeatSlopeNOK_vMin', 'Pressure_vMax', 'FeederAction4_vCnt', 'ProcessCpuLoadNormalized_value', 'Vacuum_vFreq', 'TotalMemoryConsumption_vMin', 'CpuTemperature_vStd', 'SmartMotorPositionError_vMax', 'FuseCycleDuration_vMax', 'LightBarrierActiveTaskDuration1_vTrend', 'DurationRobotFromFeederToTestBench_vMin', 'SmartMotorSpeed_vMax', 'FusePicked_vFreq', 'DurationPickToPick_vFreq', 'DurationTestBenchClosed_value', 'DurationPickToPick_vTrend', 'VacuumValveClosed_vStd', 'LightBarrierPassiveTaskDuration1_vCnt', 'DurationRobotFromTestBenchToFeeder_vMin', 'NumberFuseEstimated_vFreq', 'VacuumFusePicked_vFreq', 'FuseTestResult_vStd', 'EPOSCurrent_vStd', 'DurationTestBenchClosed_vMin', 'VacuumFusePicked_vMin', 'TemperatureThermoCam_value', 'SmartMotorPositionError_vFreq', 'LightBarrierActiveTaskDuration1_vMax', 'IntensityTotalThermoImage_vMax', 'DurationRobotFromFeederToTestBench_vStd', 'TemperatureThermoCam_vTrend', 'EPOSVelocity_vStd', 'ProcessMemoryConsumption_vStd', 'VacuumFusePicked_vCnt', 'FuseHeatSlopeNOK_vFreq', 'FuseCycleDuration_vCnt', 'Vacuum_vTrend', 'LightBarrierActiveTaskDuration1_vStd', 'FuseCycleDuration_vMin', 'DurationRobotFromTestBenchToFeeder_vTrend', 'FuseHeatSlope_vMax', 'FusePicked_vMin', 'DurationTestBenchClosed_vCnt', 'SmartMotorPositionError_vTrend', 'SmartMotorSpeed_vFreq', 'Vacuum_vMax', 'Pressure_vMin', 'Vacuum_value', 'VacuumFusePicked_value', 'Pressure_vFreq', 'FuseOutsideOperationalSpace_vCnt', 'FuseHeatSlopeNOK_vMax', 'EPOSVelocity_vTrend', 'NumberFuseEstimated_vCnt', 'Vacuum_vStd', 'EPOSCurrent_vTrend', 'FuseOutsideOperationalSpace_vFreq', 'EPOSVelocity_vFreq', 'VacuumValveClosed_value', 'LightBarrierActiveTaskDuration1_vCnt', 'FuseHeatSlope_value', 'ValidFrame_vCnt', 'Vacuum_vCnt', 'TotalMemoryConsumption_vMax', 'Vacuum_vMin', 'TotalMemoryConsumption_vStd', 'ProcessCpuLoadNormalized_vMax', 'EPOSCurrent_vMax', 'TotalCpuLoadNormalized_vStd', 'ValidFrameOptrisPIIRCamera_vFreq', 'FusePicked_vTrend', 'DurationRobotFromTestBenchToFeeder_vFreq', 'FuseCycleDuration_value', 'EPOSCurrent_vMin', 'VacuumValveClosed_vMin', 'FeederAction2_vCnt', 'DurationPickToPick_vMax', 'FuseCycleDuration_vFreq', 'LightBarrierPassiveTaskDuration1_vStd', 'NumberFuseDetected_vFreq', 'DurationTestBenchClosed_vTrend', 'LightBarrierActiveTaskDuration1_vMin', 'DurationRobotFromFeederToTestBench_vMax', 'FusePicked_vCnt', 'EPOSVelocity_vCnt', 'DurationPickToPick_vCnt', 'FuseHeatSlope_vMin', 'SmartMotorPositionError_vStd', 'SmartMotorPositionError_vMin', 'EPOSVelocity_vMax', 'EPOSPosition_vMin', 'TemperatureThermoCam_vStd', 'DurationRobotFromFeederToTestBench_vFreq', 'FuseHeatSlope_vTrend', 'EPOSVelocity_vMin', 'SmartMotorSpeed_value', 'LightBarrierActiveTaskDuration1_value', 'FuseTestResult_value', 'VacuumFusePicked_vTrend', 'FusePicked_vStd', 'EPOSPosition_vTrend', 'VacuumValveClosed_vFreq', 'LightBarrierPassiveTaskDuration1_vTrend', 'ValidFrameOptrisPIIRCamera_vCnt', 'DurationRobotFromTestBenchToFeeder_vCnt', 'LightBarrierActiveTaskDuration1_vFreq', 'FuseTestResult_vFreq', 'EPOSPosition_vFreq', 'IntensityTotalImage_vFreq', 'EPOSCurrent_vFreq', 'FeederBackgroundIlluminationIntensity_vCnt', 'DurationRobotFromTestBenchToFeeder_value', 'EPOSCurrent_value', 'DurationPickToPick_vStd', 'SmartMotorSpeed_vTrend', 'FuseCycleDuration_vStd', 'DurationRobotFromFeederToTestBench_vCnt', 'FuseHeatSlope_vStd', 'Pressure_vCnt', 'FuseIntoFeeder_vCnt', 'ValidFrame_vFreq', 'ProcessMemoryConsumption_vMin', 'TotalCpuLoadNormalized_vMax', 'DurationTestBenchClosed_vStd', 'DurationPickToPick_value', 'CpuTemperature_vMin', 'LightBarrierPassiveTaskDuration1_vFreq', 'DurationPickToPick_vMin', 'CpuTemperature_vMax', 'VacuumValveClosed_vMax', 'CpuTemperature_value', 'TotalCpuLoadNormalized_vMin', 'DurationTestBenchClosed_vMax', 'VacuumFusePicked_vStd', 'ProcessMemoryConsumption_vMax', 'EPOSCurrent_vCnt', 'Pressure_value', 'ProcessMemoryConsumption_value']
_______________________________________________________________
----------------------------------------------------------
Fold:  ws_5_fold_1
[[10691    74    27    84    90    33    30    59     0]
 [  187   412    13    88     4     7     2     0     0]
 [  364     0   350     0     0     0     2     0     0]
 [  320     0     0    28     8     0     0     0     0]
 [  869     3     1   218   884   173     0     1     0]
 [  613     5     0     5     2   815     1     0     0]
 [   78     0     0     0     7     0  1348     0     0]
 [   74     0     0     0     0     0     2  1000     0]
 [    0     0     0     0     0     0     0     0  1074]]
LightGBM Fold: 0 ACC: 0.828 F1: 0.815 MCC: 0.733
[[10710    78    37   107    82    34    34     1     5]
 [  195   415     7    80     5     9     2     0     0]
 [  433     2   276     0     1     0     4     0     0]
 [  313     0     0    36     6     0     1     0     0]
 [  853     6     3   224   824   238     0     1     0]
 [  621     5     1     4     0   809     1     0     0]
 [   22     0     0     0     9     0  1402     0     0]
 [    3     0     0     0     0     0     0  1073     0]
 [    0     0     0     0     0     4     0     0  1070]]
XGBClassifier Fold: 0 ACC: 0.829 F1: 0.813 MCC: 0.734
----------------------------------------------------------
Fold:  ws_5_fold_2
[[8896  129   14   30  376  194   10    6    4]
 [ 374  630   10    3    1   11    2    0    0]
 [ 261    0  810    0    0    0    4    1    0]
 [ 314   67   19   61  611    1    0    0    0]
 [ 188    2    0    0  166    0    1    0    0]
 [ 192   21    5    1   77   59    0    0    2]
 [ 276    1    0    0    4    1   83    0    0]
 [  54    2    0    0    0    0    0  301    0]
 [  66    0    0    0    4    0    3    0 1003]]
LightGBM Fold: 1 ACC: 0.782 F1: 0.769 MCC: 0.612
[[8919  118   14   29  372  190   11    2    4]
 [ 368  638    6    1    1   11    2    0    4]
 [ 249    0  824    0    0    0    3    0    0]
 [ 321   73   21   64  593    1    0    0    0]
 [ 191    1    0    0  163    0    2    0    0]
 [ 202   25    3    2   72   53    0    0    0]
 [ 265    1    0    0    0    0   94    5    0]
 [   0    0    0    0    0    0    0  357    0]
 [   1    0    0    0    0    0    3    0 1072]]
XGBClassifier Fold: 1 ACC: 0.794 F1: 0.78 MCC: 0.634
----------------------------------------------------------
Fold:  ws_5_fold_3
[[9656  273   89  179  944  252   37    1    8]
 [ 412  653    4    2    0    2    0    2    0]
 [  85  138  793    2   11   41    0    0    5]
 [ 225   72    5    7    4   44    0    0    0]
 [ 152    2    0   84  118    0    0    0    0]
 [ 137    0    1    0  113  820    0    0    5]
 [ 218   24    1    0    2    0  828    2    0]
 [   4    0    0    0    6  116    0  948    0]
 [  90    0    7    0    0    1    3    0  253]]
LightGBM Fold: 2 ACC: 0.787 F1: 0.803 MCC: 0.64
[[9712  251   86  152  935  267   30    5    1]
 [ 411  659    1    3    0    0    0    1    0]
 [ 117   87  811    2   14   44    0    0    0]
 [ 226   79    9    3    4   36    0    0    0]
 [ 157    2    0   92  105    0    0    0    0]
 [ 136    2    2    0  122  814    0    0    0]
 [ 218   22    1    1    1    0  832    0    0]
 [   0    0    0    0    0    0    0 1074    0]
 [   0    0    0    0    0    0    0    0  354]]
XGBClassifier Fold: 2 ACC: 0.803 F1: 0.818 MCC: 0.668

LightGBM Avg ACC: 0.799 Avg F1: 0.796 Avg MCC: 0.662
XGBM Avg ACC: 0.809 Avg F1: 0.804 Avg MCC: 0.679
_______________________________________________________________
----------------------------------------------------------
Fold:  ws_10_fold_1
[[10551    51     3   190    66    17     1   103     1]
 [  112   419    16   153     3     0     0     0     0]
 [  171     0   533     0     0     0     2     0     0]
 [  295     0     0    56     0     0     0     0     0]
 [  172    65     0    15  1818    62     1     6     0]
 [  420     2     3     3     1  1001     0     1     0]
 [   16     0     0     3     6     2  1396     0     0]
 [   10     0     0     0     0     0     0  1061     0]
 [    2     0    12    16     0     1     0     0  1038]]
LightGBM Fold: 0 ACC: 0.899 F1: 0.898 MCC: 0.846
[[10475    52     3   356    75    19     2     0     1]
 [  120   440     8   132     3     0     0     0     0]
 [  151     1   552     0     0     0     2     0     0]
 [  278     0     0    73     0     0     0     0     0]
 [  126    12     0    14  1939    47     0     1     0]
 [  409     2     2     3     0  1012     0     3     0]
 [    3     0     0     3     4     4  1409     0     0]
 [    0     0     0     0     0     0     0  1071     0]
 [    0     0     0     0     0     2     0     0  1067]]
XGBClassifier Fold: 0 ACC: 0.908 F1: 0.91 MCC: 0.859
----------------------------------------------------------
Fold:  ws_10_fold_2
[[9105   63    5   29   72  265    1    7    7]
 [ 257  756    0    1    1    9    0    0    2]
 [  65    0 1006    0    0    0    0    0    0]
 [ 638   20   11  364   35    0    0    0    0]
 [  39    0    0    0  313    0    0    0    0]
 [ 105    3    0    3   32  209    0    0    0]
 [   2    0    0    0    4    0  354    0    0]
 [  44    1    0    0    0    0    0  307    0]
 [  22    0    3    0    1    0    2    0 1043]]
LightGBM Fold: 1 ACC: 0.885 F1: 0.877 MCC: 0.798
[[9118   63   11   34   70  246    1    0   11]
 [ 258  756    0    2    1    9    0    0    0]
 [  49    0 1022    0    0    0    0    0    0]
 [ 593   18    6  400   51    0    0    0    0]
 [  35    0    0    0  317    0    0    0    0]
 [ 111    4    0    3   29  205    0    0    0]
 [   1    0    0    0    1    0  358    0    0]
 [   0    0    0    0    0    0    0  352    0]
 [   0    0    0    0    0    0    1    0 1070]]
XGBClassifier Fold: 1 ACC: 0.894 F1: 0.888 MCC: 0.815
----------------------------------------------------------
Fold:  ws_10_fold_3
[[10612   182    20   218    69   215     9     3     1]
 [  259   808     0     0     0     0     0     0     3]
 [   35     6  1025     0     0     4     0     0     0]
 [  225    55     4     3    14    50     1     0     0]
 [   21     1     0    12   317     0     0     0     0]
 [   78     0     0     0    75   918     0     0     0]
 [   15     4     0     0     0     0  1051     0     0]
 [   14     0     0     7     3    97     0   948     0]
 [  134     2     7     0     0     0     1     0   205]]
LightGBM Fold: 2 ACC: 0.896 F1: 0.894 MCC: 0.818
[[10594   183    29   199   115   199     8     0     2]
 [  244   823     2     0     0     1     0     0     0]
 [   31     7  1028     0     0     4     0     0     0]
 [  218    51    10     2    17    54     0     0     0]
 [   27     1     0    10   313     0     0     0     0]
 [   84     0     0     0    81   906     0     0     0]
 [    5     1     0     0     0     0  1064     0     0]
 [    0     0     0     0     0     0     0  1069     0]
 [    0     0     0     0     0     0     0     0   349]]
XGBClassifier Fold: 2 ACC: 0.911 F1: 0.908 MCC: 0.845

LightGBM Avg ACC: 0.893 Avg F1: 0.89 Avg MCC: 0.821
XGBM Avg ACC: 0.904 Avg F1: 0.902 Avg MCC: 0.84
_______________________________________________________________
----------------------------------------------------------
Fold:  ws_15_fold_1
[[10518    27     2   265    22     9     2    31     2]
 [   85   443     1   159     0     4     0     1     0]
 [   33     0   663     0     0     0     0     0     0]
 [  250     0     1    95     0     0     0     0     0]
 [   64     7     0     5  2037    14     0     1     1]
 [  404     0     0     3     0  1012     0     2     0]
 [    0     0     0     1     0     0  1412     0     0]
 [    0     0     0     0     0     0     0  1066     0]
 [    0     1    13    71     0     1     0     0   978]]
LightGBM Fold: 0 ACC: 0.925 F1: 0.927 MCC: 0.886
[[10371    35     1   410    26     9     1    25     0]
 [   73   416     8   196     0     0     0     0     0]
 [   36     0   660     0     0     0     0     0     0]
 [  261     0     1    84     0     0     0     0     0]
 [   17     3     0     0  2098    10     0     1     0]
 [  400     1     0     3     0  1015     0     2     0]
 [    0     0     0     0     0     0  1413     0     0]
 [    0     0     0     0     0     0     0  1066     0]
 [    0     0     0     1     0     0     0     0  1063]]
XGBClassifier Fold: 0 ACC: 0.923 F1: 0.927 MCC: 0.884
----------------------------------------------------------
Fold:  ws_15_fold_2
[[9119   32    2   53   13  208    5   13    4]
 [ 202  812    0    1    2    3    0    0    1]
 [  11    0 1055    0    0    0    0    0    0]
 [ 668    1    1  391    2    0    0    0    0]
 [  10    0    0    0  337    0    0    0    0]
 [  54    0    1    1    8  283    0    0    0]
 [   0    0    0    0    0    0  355    0    0]
 [  39    0    0    0    0    0    0  308    0]
 [   0    0    0    0    0    0    1    0 1065]]
LightGBM Fold: 1 ACC: 0.911 F1: 0.904 MCC: 0.846
[[9108   54    4   44   16  208    1    0   14]
 [ 192  823    0    2    2    2    0    0    0]
 [   5    0 1061    0    0    0    0    0    0]
 [ 649   10    7  395    2    0    0    0    0]
 [   6    0    0    0  341    0    0    0    0]
 [  52    0    0    2   11  282    0    0    0]
 [   0    0    0    0    0    0  355    0    0]
 [   0    0    0    0    0    0    0  347    0]
 [   0    0    0    0    0    0    1    0 1065]]
XGBClassifier Fold: 1 ACC: 0.915 F1: 0.907 MCC: 0.852
----------------------------------------------------------
Fold:  ws_15_fold_3
[[10732   121    26   130     2   200     0     1     7]
 [  112   952     0     0     0     0     0     1     0]
 [   12     0  1049     0     0     4     0     0     0]
 [  236    28     8     0     9    66     0     0     0]
 [    0     0     0     4   342     0     0     0     0]
 [   61     2     0     0    24   979     0     0     0]
 [    0     0     0     0     0     0  1065     0     0]
 [   14     0     0     0     0    47     0  1003     0]
 [  144     0     0     0     0     0     0     0   200]]
LightGBM Fold: 2 ACC: 0.928 F1: 0.923 MCC: 0.875
[[10779   140    38    99     3   160     0     0     0]
 [  109   956     0     0     0     0     0     0     0]
 [   11     0  1052     0     0     2     0     0     0]
 [  245    46     8     0     8    40     0     0     0]
 [   16     0     0    22   308     0     0     0     0]
 [   62     1     0     0    33   970     0     0     0]
 [    0     0     0     0     0     0  1065     0     0]
 [    0     0     0     0     0     6     0  1058     0]
 [    0     0     0     0     0     0     0     0   344]]
XGBClassifier Fold: 2 ACC: 0.94 F1: 0.935 MCC: 0.896

LightGBM Avg ACC: 0.921 Avg F1: 0.918 Avg MCC: 0.869
XGBM Avg ACC: 0.926 Avg F1: 0.923 Avg MCC: 0.877
_______________________________________________________________
----------------------------------------------------------
Fold:  ws_20_fold_1
[[10333    26     4   362    42     4     0     1     1]
 [   48   442     0   191     0     1     0     1     0]
 [    6     0   680     0     0     0     0     0     0]
 [  293     0     0    47     0     0     0     0     1]
 [   25    21     0     2  2066     3     0     1     1]
 [  358     0     0     5     0  1048     0     0     0]
 [    0     0     0     0     0     0  1403     0     0]
 [    0     0     0     0     0     0     0  1061     0]
 [    0     0     0    69     0     2     0     0   988]]
LightGBM Fold: 0 ACC: 0.925 F1: 0.93 MCC: 0.887
[[10266    20     3   442    15     2     0    25     0]
 [   46   441     0   195     0     1     0     0     0]
 [   11     0   675     0     0     0     0     0     0]
 [  245     0     1    95     0     0     0     0     0]
 [    7     1     0     0  2109     1     0     1     0]
 [  365     0     0     4     1  1041     0     0     0]
 [    0     0     0     0     0     0  1403     0     0]
 [    0     0     0     0     0     0     0  1061     0]
 [    0     0     0   269     0     0     0     0   790]]
XGBClassifier Fold: 0 ACC: 0.915 F1: 0.926 MCC: 0.873
----------------------------------------------------------
Fold:  ws_20_fold_2
[[9049   42    3   54    4  161    7   18    6]
 [ 139  870    0    5    0    0    0    0    2]
 [   4    0 1057    0    0    0    0    0    0]
 [ 747    2    0  306    0    3    0    0    0]
 [   0    0    0    0  342    0    0    0    0]
 [  21    0    0    0    6  312    0    0    3]
 [   0    0    0    0    0    0  350    0    0]
 [  32    0    0    0    0    0    0  310    0]
 [   0    0    0    0    0    0    1    0 1060]]
LightGBM Fold: 1 ACC: 0.916 F1: 0.904 MCC: 0.854
[[9037   42    2   64   12  185    1    0    1]
 [ 124  886    0    5    0    1    0    0    0]
 [   0    0 1061    0    0    0    0    0    0]
 [ 675    3    0  376    0    4    0    0    0]
 [   0    0    0    0  342    0    0    0    0]
 [  12    0    0    0    4  326    0    0    0]
 [   0    0    0    0    0    1  349    0    0]
 [   0    0    0    0    0    0    0  342    0]
 [   0    0    0    0    0    0    1    0 1060]]
XGBClassifier Fold: 1 ACC: 0.924 F1: 0.916 MCC: 0.868
----------------------------------------------------------
Fold:  ws_20_fold_3
[[10718    59    10   172     1   128     2    17     2]
 [   65   995     0     0     0     0     0     0     0]
 [    8     0  1050     0     0     2     0     0     0]
 [  262    20     9     0     2    49     0     0     0]
 [   16     0     0    18   307     0     0     0     0]
 [   52     0     0     0     2  1007     0     0     0]
 [    0     0     0     0     0     0  1060     0     0]
 [    4     0     0     0     0     2     0  1053     0]
 [   59     0     0     0     0     0     0     0   280]]
LightGBM Fold: 2 ACC: 0.945 F1: 0.941 MCC: 0.904
[[10779    87     3   107     1   131     0     0     1]
 [   64   995     0     0     0     1     0     0     0]
 [    6     0  1049     0     0     5     0     0     0]
 [  248    38     7     0     3    46     0     0     0]
 [   19     0     0    21   301     0     0     0     0]
 [   53     0     0     0     0  1008     0     0     0]
 [    0     0     0     0     0     0  1060     0     0]
 [    0     0     0     0     0     4     0  1055     0]
 [    0     0     0     0     0     0     0     0   339]]
XGBClassifier Fold: 2 ACC: 0.952 F1: 0.946 MCC: 0.915

LightGBM Avg ACC: 0.929 Avg F1: 0.925 Avg MCC: 0.882
XGBM Avg ACC: 0.93 Avg F1: 0.929 Avg MCC: 0.885
_______________________________________________________________
----------------------------------------------------------
Fold:  ws_25_fold_1
[[10264    16     4   285    45     6     0    46     2]
 [   38   477     0   156     0     1     0     0     1]
 [    0     0   676     0     0     0     0     0     0]
 [  282     0     0    53     0     0     0     1     0]
 [   36    23     0     2  2037     1     0     7     3]
 [  352     0     0     5     0  1043     0     1     0]
 [    0     0     0     1     0     0  1392     0     0]
 [    0     0     0     0     0     0     0  1056     0]
 [    0     0     0   376     0     1     0     0   677]]
LightGBM Fold: 0 ACC: 0.913 F1: 0.922 MCC: 0.869
[[10315    23     0   287    15     5     0    23     0]
 [   35   494     0   143     0     1     0     0     0]
 [    6     0   670     0     0     0     0     0     0]
 [  219     0     1   116     0     0     0     0     0]
 [    2     0     0     0  2106     0     0     1     0]
 [  347     0     0     3     0  1049     0     0     2]
 [    0     0     0     0     0     0  1393     0     0]
 [    0     0     0     0     0     0     0  1056     0]
 [    0     0     1   274     0     0     0     0   779]]
XGBClassifier Fold: 0 ACC: 0.928 F1: 0.936 MCC: 0.892
----------------------------------------------------------
Fold:  ws_25_fold_2
[[8917   12    0   88   13  154   17   35    3]
 [  93  851    6   57    0    0    0    0    4]
 [   0    0 1056    0    0    0    0    0    0]
 [ 764    0    0  284    0    5    0    0    0]
 [   0    0    0    0  337    0    0    0    0]
 [   1    0    0    0    0  336    0    0    0]
 [   0    0    0    0    0    0  345    0    0]
 [  77    0    0    0    0    0    0  260    0]
 [ 318    0    0    0    0    2    1    0  735]]
LightGBM Fold: 1 ACC: 0.888 F1: 0.876 MCC: 0.804
[[8932   19    2   80   21  164    1    0   20]
 [  84  891    0   36    0    0    0    0    0]
 [   0    0 1056    0    0    0    0    0    0]
 [ 761    0    0  285    0    7    0    0    0]
 [   0    0    0    0  337    0    0    0    0]
 [   0    0    0    0    0  337    0    0    0]
 [   0    0    0    0    0    0  345    0    0]
 [   0    0    0    0    0    0    0  337    0]
 [   0    0    0    0    0    0    1    0 1055]]
XGBClassifier Fold: 1 ACC: 0.919 F1: 0.908 MCC: 0.86
----------------------------------------------------------
Fold:  ws_25_fold_3
[[10587    58    12   214     0   119     0     1     8]
 [   35  1020     0     0     0     0     0     0     0]
 [    2     0  1052     0     0     1     0     0     0]
 [  247    21    13     0     0    56     0     0     0]
 [   19     0     0    14   303     0     0     0     0]
 [   31     1     0     0     2  1021     0     0     1]
 [    0     0     0     0     0     0  1055     0     0]
 [    0     0     0     0     0     6     0  1048     0]
 [   14     2     0     0     0     0     0     0   318]]
LightGBM Fold: 2 ACC: 0.949 F1: 0.946 MCC: 0.912
[[10710    36     0   123     2   128     0     0     0]
 [   34  1021     0     0     0     0     0     0     0]
 [    1     0  1046     0     0     8     0     0     0]
 [  238    18     0     0     0    81     0     0     0]
 [   19     0     0    27   290     0     0     0     0]
 [   19     0     0     0     0  1037     0     0     0]
 [    0     0     0     0     0     0  1055     0     0]
 [    0     0     0     0     0    17     0  1037     0]
 [    0     0     0     0     0     0     0     0   334]]
XGBClassifier Fold: 2 ACC: 0.957 F1: 0.952 MCC: 0.924

LightGBM Avg ACC: 0.917 Avg F1: 0.915 Avg MCC: 0.862
XGBM Avg ACC: 0.935 Avg F1: 0.932 Avg MCC: 0.892
_______________________________________________________________
----------------------------------------------------------
Fold:  ws_30_fold_1
[[10146    11     3   298    41     9     0    52     3]
 [   34   467     0   161     0     1     0     0     0]
 [    0     0   666     0     0     0     0     0     0]
 [  204     0     1   126     0     0     0     0     0]
 [   33    14     0     4  2045     1     0     1     1]
 [  338     0     0     5     0  1048     0     0     0]
 [    0     0     0     1     0     0  1382     0     0]
 [   27     0     0     0     0     0     0  1024     0]
 [    0     1     0   742     0     4     0     0   302]]
LightGBM Fold: 0 ACC: 0.896 F1: 0.906 MCC: 0.847
[[10227    16     0   277    19     5     0    19     0]
 [   29   486     0   147     0     1     0     0     0]
 [    6     0   660     0     0     0     0     0     0]
 [  160     0     1   170     0     0     0     0     0]
 [    0     0     0     0  2098     1     0     0     0]
 [  337     1     0     4     0  1048     0     0     1]
 [    0     0     0     2     0     0  1381     0     0]
 [    0     0     0     0     0     0     0  1051     0]
 [    0     0     1     0     0     6     0     0  1042]]
XGBClassifier Fold: 0 ACC: 0.946 F1: 0.949 MCC: 0.919
----------------------------------------------------------
Fold:  ws_30_fold_2
[[8836   10    0   59   26  139    4   59    1]
 [  59  867    0   68    0    3    0    0    9]
 [   0    0 1051    0    0    0    0    0    0]
 [ 765    1    0  277    0    5    0    0    0]
 [   0    0    0    0  332    0    0    0    0]
 [   0    0    0    0    0  332    0    0    0]
 [   0    0    0    0    0    0  340    0    0]
 [   0    0    0    0    0    0    0  332    0]
 [ 193    0    0    0    0    1    1    0  856]]
LightGBM Fold: 1 ACC: 0.904 F1: 0.892 MCC: 0.834
[[8884   20    0   71   13  121    0    0   25]
 [  54  881    2   67    0    2    0    0    0]
 [   0    0 1051    0    0    0    0    0    0]
 [ 798    2    0  241    0    7    0    0    0]
 [   0    0    0    0  332    0    0    0    0]
 [   0    0    0    1    0  331    0    0    0]
 [   0    0    0    0    0    0  340    0    0]
 [   0    0    0    0    0    0    0  332    0]
 [   0    0    0    0    0    0    1    0 1050]]
XGBClassifier Fold: 1 ACC: 0.919 F1: 0.906 MCC: 0.861
----------------------------------------------------------
Fold:  ws_30_fold_3
[[10609     6     4   149     1    95     0    20     5]
 [   12  1038     0     0     0     0     0     0     0]
 [    0     0  1050     0     0     0     0     0     0]
 [  271    11    12     0     0    38     0     0     0]
 [    4     0     0     7   320     0     0     0     0]
 [    3     1     0     0     1  1046     0     0     0]
 [    0     0     0     0     0     0  1050     0     0]
 [    0     0     0     0     0     0     0  1049     0]
 [    2     0     0     0     0     0     0     0   327]]
LightGBM Fold: 2 ACC: 0.963 F1: 0.958 MCC: 0.935
[[10609     4     0   144     4   128     0     0     0]
 [   14  1036     0     0     0     0     0     0     0]
 [    1     0  1046     0     0     3     0     0     0]
 [  221    10     0     0     0   101     0     0     0]
 [    9     0     0    20   302     0     0     0     0]
 [    1     0     0     0     0  1050     0     0     0]
 [    0     0     0     0     0     0  1050     0     0]
 [    0     0     0     0     0    14     0  1035     0]
 [    0     0     0     0     0     0     0     0   329]]
XGBClassifier Fold: 2 ACC: 0.961 F1: 0.956 MCC: 0.932

LightGBM Avg ACC: 0.921 Avg F1: 0.919 Avg MCC: 0.872
XGBM Avg ACC: 0.942 Avg F1: 0.937 Avg MCC: 0.904
_______________________________________________________________
----------------------------------------------------------
Fold:  ws_35_fold_1
[[9974    3   16  337   44   13    0   67    4]
 [  15  550    0   87    0    1    0    0    0]
 [   0    0  656    0    0    0    0    0    0]
 [ 211    0    0  115    0    0    0    0    0]
 [  19    2    0    2 2062    0    0    4    0]
 [ 331    0    0    5    0 1045    0    0    0]
 [   0    0    0    1    0    0 1372    0    0]
 [   4    0    0    0    0    0    0 1042    0]
 [   0    1    0  216    0    0    0    0  827]]
LightGBM Fold: 0 ACC: 0.927 F1: 0.934 MCC: 0.891
[[10202    28     0   190    19     2     0    17     0]
 [   15   548     0    88     0     2     0     0     0]
 [    6     0   650     0     0     0     0     0     0]
 [  137     0     1   188     0     0     0     0     0]
 [    0     0     0     0  2088     1     0     0     0]
 [  331     0     0     1     0  1049     0     0     0]
 [    0     0     0     1     0     5  1367     0     0]
 [    0     0     0     0     0     0     0  1046     0]
 [    0     0     0     0     0    19     0     0  1025]]
XGBClassifier Fold: 0 ACC: 0.955 F1: 0.955 MCC: 0.932
----------------------------------------------------------
Fold:  ws_35_fold_2
[[8742   13    0   84   13  113    0   59    5]
 [  28  869    0   73    0    7    0    0   24]
 [   0    0 1046    0    0    0    0    0    0]
 [ 811    0    0  225    0    7    0    0    0]
 [   0    0    0    0  327    0    0    0    0]
 [   0    0    0    0    0  327    0    0    0]
 [   0    0    0    0    0    0  335    0    0]
 [   0    0    0    0    0    0    0  327    0]
 [ 178    0    0    0    0    2    1    0  865]]
LightGBM Fold: 1 ACC: 0.902 F1: 0.888 MCC: 0.83
[[8790   32    0   88   12  107    0    0    0]
 [  26  875   19   71    0   10    0    0    0]
 [   0    0 1046    0    0    0    0    0    0]
 [ 782    0    0  257    0    4    0    0    0]
 [   0    0    0    0  327    0    0    0    0]
 [   0    0    0    0    0  327    0    0    0]
 [   0    0    0    1    0    0  334    0    0]
 [   0    0    0    0    0    0    0  327    0]
 [   0    0    0    0    0    0    1    0 1045]]
XGBClassifier Fold: 1 ACC: 0.92 F1: 0.908 MCC: 0.863
----------------------------------------------------------
Fold:  ws_35_fold_3
[[10435     2     0   206     0   109     0    26     1]
 [   10  1035     0     0     0     0     0     0     0]
 [    0     0  1044     0     0     1     0     0     0]
 [  221     3     6     0     0    97     0     0     0]
 [    0     0     0     9   317     0     0     0     0]
 [    0     0     0     0     0  1046     0     0     0]
 [    0     0     0     0     0     0  1045     0     0]
 [    0     0     0     0     0    14     0  1030     0]
 [    1     0     0     0     0     0     0     0   323]]
LightGBM Fold: 2 ACC: 0.958 F1: 0.956 MCC: 0.929
[[10512     0     0   144     1   122     0     0     0]
 [    7  1038     0     0     0     0     0     0     0]
 [    0     0  1045     0     0     0     0     0     0]
 [  188     1     0     0     0   138     0     0     0]
 [    0     0     0    16   310     0     0     0     0]
 [    0     0     0     0     0  1046     0     0     0]
 [    0     0     0     0     0     0  1045     0     0]
 [    0     0     0     0     0    35     0  1009     0]
 [    0     0     0     0     0     0     0     0   324]]
XGBClassifier Fold: 2 ACC: 0.962 F1: 0.958 MCC: 0.934

LightGBM Avg ACC: 0.929 Avg F1: 0.926 Avg MCC: 0.883
XGBM Avg ACC: 0.946 Avg F1: 0.94 Avg MCC: 0.91
_______________________________________________________________
----------------------------------------------------------
Fold:  ws_40_fold_1
[[9973   15   16  220   52    6    0   71    0]
 [  14  576    0   52    0    1    0    0    0]
 [   0    0  646    0    0    0    0    0    0]
 [ 215    0    0  106    0    0    0    0    0]
 [   1    0    0    0 2073    1    0    4    0]
 [ 338    0    0    5    0 1028    0    0    0]
 [   0    0    0    0    0    0 1363    0    0]
 [   6    0    0    0    0    0    0 1035    0]
 [   0    1    0  796    0    0    0    0  242]]
LightGBM Fold: 0 ACC: 0.904 F1: 0.909 MCC: 0.858
[[10099    37     5   182    24     0     4     2     0]
 [   12   574     0    55     0     1     1     0     0]
 [    6     0   640     0     0     0     0     0     0]
 [   75     0     0   246     0     0     0     0     0]
 [    0     0     0    15  2063     0     0     1     0]
 [  326     0     1     4     0  1040     0     0     0]
 [    0     0     0     5     0     0  1358     0     0]
 [    0     0     0     0     0     0     0  1041     0]
 [    0     0     0     0     0     1     0     0  1038]]
XGBClassifier Fold: 0 ACC: 0.96 F1: 0.961 MCC: 0.94
----------------------------------------------------------
Fold:  ws_40_fold_2
[[8642   14    0   89    8  102    5   46   18]
 [  14  884    0   68    0   29    0    0    1]
 [   0    0 1041    0    0    0    0    0    0]
 [ 732    0    0  306    0    0    0    0    0]
 [   0    0    0    0  322    0    0    0    0]
 [   0    0    0    0    0  322    0    0    0]
 [   0    0    0    0    0    0  330    0    0]
 [   0    0    0    0    0    0    0  322    0]
 [ 551    0    0    0    0    0    1    0  489]]
LightGBM Fold: 1 ACC: 0.883 F1: 0.868 MCC: 0.796
[[8744    8    3   65    9   92    0    0    3]
 [  13  879   13   67    0   24    0    0    0]
 [   0    0 1041    0    0    0    0    0    0]
 [ 787    0    0  251    0    0    0    0    0]
 [   0    0    0    0  322    0    0    0    0]
 [   0    0    0    0    0  322    0    0    0]
 [   0    0    0    0    0    0  330    0    0]
 [   0    0    0    0    0    0    0  322    0]
 [   0    0    0    0    0    0    1    0 1040]]
XGBClassifier Fold: 1 ACC: 0.924 F1: 0.911 MCC: 0.87
----------------------------------------------------------
Fold:  ws_40_fold_3
[[10261     4     0   260     2   120     0    22     0]
 [    9  1031     0     0     0     0     0     0     0]
 [    0     0  1038     0     0     2     0     0     0]
 [  193     0     0     0     0   129     0     0     0]
 [    2     0     0     1   318     0     0     0     0]
 [    0     0     0     0     0  1041     0     0     0]
 [    0     0     0     0     0     0  1040     0     0]
 [    0     0     0     0     0     5     0  1034     0]
 [    0     0     0     0     0     0     0     0   319]]
LightGBM Fold: 2 ACC: 0.955 F1: 0.954 MCC: 0.924
[[10448     0     0   135     2    84     0     0     0]
 [   11  1029     0     0     0     0     0     0     0]
 [    0     0  1040     0     0     0     0     0     0]
 [  188     1     0     0     0   133     0     0     0]
 [    0     0     0    11   310     0     0     0     0]
 [    0     0     0     0     0  1041     0     0     0]
 [    0     0     0     0     0     0  1040     0     0]
 [    0     0     0     0     0     0     0  1039     0]
 [    0     0     0     0     0     0     0     0   319]]
XGBClassifier Fold: 2 ACC: 0.966 F1: 0.962 MCC: 0.942

LightGBM Avg ACC: 0.914 Avg F1: 0.91 Avg MCC: 0.859
XGBM Avg ACC: 0.95 Avg F1: 0.945 Avg MCC: 0.917
_______________________________________________________________
----------------------------------------------------------
Fold:  ws_45_fold_1
[[9844    6   15  181   58    3    0  139    2]
 [  16  558    0   59    0    0    0    0    0]
 [   0    0  635    0    0    1    0    0    0]
 [ 194    0    0  122    0    0    0    0    0]
 [   6    9    0    0 2045    1    0    8    0]
 [ 337    0    0    3    0 1019    0    2    0]
 [   0    0    0    0    0    0 1353    0    0]
 [   6    0    0    0    0    0    0 1030    0]
 [   0    0    0  269    0    0    0    0  765]]
LightGBM Fold: 0 ACC: 0.93 F1: 0.934 MCC: 0.895
[[9901   57    1  241   48    0    0    0    0]
 [  11  335    1  285    0    1    0    0    0]
 [   6    0  630    0    0    0    0    0    0]
 [  46    0    0  270    0    0    0    0    0]
 [   0    0    0   40 2028    0    0    1    0]
 [ 321    0    0    5    0 1035    0    0    0]
 [   0    0    0    4    0    0 1349    0    0]
 [   0    0    0    0    0    0    0 1036    0]
 [   0    0    0  719    0    0    0    0  315]]
XGBClassifier Fold: 0 ACC: 0.904 F1: 0.915 MCC: 0.861
----------------------------------------------------------
Fold:  ws_45_fold_2
[[8421   33    0  129   11  107    6   85   27]
 [   5  853    0   68    0   65    0    0    0]
 [   0    0 1036    0    0    0    0    0    0]
 [ 833    0    0  199    0    1    0    0    0]
 [   0    0    0    0  317    0    0    0    0]
 [   0    0    0    0    0  317    0    0    0]
 [   0    0    0    0    0    0  325    0    0]
 [   6    0    0    0    0    0    0  311    0]
 [ 141    0    0    0    0    0    1    0  894]]
LightGBM Fold: 1 ACC: 0.893 F1: 0.879 MCC: 0.816
[[8560   27    3  111    9   96    0    6    7]
 [   4  863   12   63    0   49    0    0    0]
 [   0    0 1036    0    0    0    0    0    0]
 [ 786    0    0  246    0    1    0    0    0]
 [   0    0    0    0  317    0    0    0    0]
 [   0    0    0    0    0  317    0    0    0]
 [   0    0    0    0    0    0  325    0    0]
 [   0    0    0    0    0    0    0  317    0]
 [   0    0    0    0    0    0    1    0 1035]]
XGBClassifier Fold: 1 ACC: 0.917 F1: 0.905 MCC: 0.858
----------------------------------------------------------
Fold:  ws_45_fold_3
[[10250     1     0   166    13   107     0    22     0]
 [    8  1027     0     0     0     0     0     0     0]
 [    0     0  1027     0     0     8     0     0     0]
 [  220     0     2     0     0    95     0     0     0]
 [    0     0     0     0   316     0     0     0     0]
 [    0     0     0     0     0  1036     0     0     0]
 [    0     0     0     0     0     0  1035     0     0]
 [    0     0     0     0     0     0     0  1034     0]
 [    0     0     0     0     0     0     0     0   314]]
LightGBM Fold: 2 ACC: 0.962 F1: 0.958 MCC: 0.934
[[10316     3     0   163     2    74     1     0     0]
 [   10  1025     0     0     0     0     0     0     0]
 [    0     0  1035     0     0     0     0     0     0]
 [  205     0     0     0     0   112     0     0     0]
 [    0     0     0    11   305     0     0     0     0]
 [    0     0     0     0     0  1036     0     0     0]
 [    0     0     0     0     0     0  1035     0     0]
 [    0     0     0     0     0     0     0  1034     0]
 [    0     0     0     0     0     0     0     0   314]]
XGBClassifier Fold: 2 ACC: 0.965 F1: 0.961 MCC: 0.94

LightGBM Avg ACC: 0.928 Avg F1: 0.924 Avg MCC: 0.882
XGBM Avg ACC: 0.929 Avg F1: 0.927 Avg MCC: 0.886
_______________________________________________________________
----------------------------------------------------------
Fold:  ws_50_fold_1
[[9552    9   13  273   68    5    0  222    1]
 [   5  581    0   36    0    1    0    0    0]
 [   0    0  626    0    0    0    0    0    0]
 [  29    0    0  280    0    0    0    1    1]
 [   2   16    0    4 2029    1    0    7    0]
 [ 338    0    0    3    0 1008    0    2    0]
 [   0    0    0    0    0    0 1343    0    0]
 [   6    0    0    0    0    0    0 1025    0]
 [   0    0    0  751    0    0    0    0  278]]
LightGBM Fold: 0 ACC: 0.903 F1: 0.909 MCC: 0.86
[[9771   72   20  228   14    0    0   17   21]
 [   4  337    0  281    0    1    0    0    0]
 [   4    0  622    0    0    0    0    0    0]
 [  40    0    0  271    0    0    0    0    0]
 [   0    0    0   73 1985    0    0    1    0]
 [ 316    0    0    5    0 1030    0    0    0]
 [   0    0    0    1    0    0 1342    0    0]
 [   0    0    0    0    0    0    0 1031    0]
 [   0    0    0  722    0    0    0    0  307]]
XGBClassifier Fold: 0 ACC: 0.902 F1: 0.913 MCC: 0.858
----------------------------------------------------------
Fold:  ws_50_fold_2
[[8254   44    0  117    0  124    8  135   32]
 [  29  776    0   65    0   75    0    0   41]
 [   0    0 1031    0    0    0    0    0    0]
 [ 625    0    0  401    0    2    0    0    0]
 [   0    0    0    0  312    0    0    0    0]
 [   0    0    0    0    0  312    0    0    0]
 [   0    0    0    0    0    0  320    0    0]
 [   1    0    0    0    0    0    0  311    0]
 [ 160    0    0    1    0    0    1    0  869]]
LightGBM Fold: 1 ACC: 0.896 F1: 0.89 MCC: 0.822
[[8404   50    4  148    4   90    0    5    9]
 [   2  832   11   36    0  105    0    0    0]
 [   0    0 1031    0    0    0    0    0    0]
 [ 780    0    0  248    0    0    0    0    0]
 [   0    0    0    0  312    0    0    0    0]
 [   0    0    0    0    0  312    0    0    0]
 [   0    0    0    0    0    0  320    0    0]
 [   0    0    0    0    0    0    0  312    0]
 [  36    0    0    0    0    0    1    0  994]]
XGBClassifier Fold: 1 ACC: 0.909 F1: 0.897 MCC: 0.844
----------------------------------------------------------
Fold:  ws_50_fold_3
[[10077    17     0   230     8   100     0    17     0]
 [    5  1025     0     0     0     0     0     0     0]
 [    0     0  1022     0     0     8     0     0     0]
 [  231     0    15     0     0    66     0     0     0]
 [    0     0     0     2   309     0     0     0     0]
 [    0     0     0     0     0  1031     0     0     0]
 [    0     0     0     0     0     0  1030     0     0]
 [    0     0     0     0     0     0     0  1029     0]
 [    0     0     0     0     0     0     0     0   309]]
LightGBM Fold: 2 ACC: 0.958 F1: 0.956 MCC: 0.928
[[10215     9     0   166     5    54     0     0     0]
 [    6  1024     0     0     0     0     0     0     0]
 [    0     0  1016     0     0    14     0     0     0]
 [  226     0     5     0     0    81     0     0     0]
 [    0     0     0     0   311     0     0     0     0]
 [    0     0     0     0     0  1031     0     0     0]
 [    0     0     0     0     0     0  1030     0     0]
 [    0     0     0     0     0     0     0  1029     0]
 [    0     0     0     0     0     0     0     0   309]]
XGBClassifier Fold: 2 ACC: 0.966 F1: 0.962 MCC: 0.941

LightGBM Avg ACC: 0.919 Avg F1: 0.918 Avg MCC: 0.87
XGBM Avg ACC: 0.926 Avg F1: 0.924 Avg MCC: 0.881

"""



