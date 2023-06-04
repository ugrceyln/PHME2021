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
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import ExtraTreeClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.svm import SVC
import pickle

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

# 25, 30 -> 125, 130

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

fold_num = 3
cv = StratifiedKFold(n_splits=fold_num, shuffle=True, random_state=41)

# df_ = df.copy(deep=True)
# del df_['runId']
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
# importance_df = importance_df.loc[(importance_df["importance_mean"] > 0.0001) & (importance_df["importance_std"] < 0.001)]

df_ori = df.copy(deep=True)

importance_df = pd.read_pickle("importance_df.pkl")
importance_df = importance_df.loc[(importance_df["importance_mean"] > 0)]
sorted_features_imp = list(importance_df["feature"].values)
value_features_imp = list(importance_df["importance_mean"].values)
f_imp = [(name, value) for name, value in zip(sorted_features_imp, value_features_imp)]
print("fold_num:", fold_num, len(f_imp), f_imp)
print(sorted_features_imp)
# num_leaves = [10, 20, 30, 40, 50]
# learning_rates = [0.1, 0.05, 0.01]
# n_estimators = [100, 350, 700, 1000]

ws = 30
fn = len(sorted_features_imp)# 135

print("-------------------------------------------------------------")
model_property = "ws_{}_fn_{}".format(ws, fn)
print(model_property)

sensor_list = sorted_features_imp[:fn].copy()

acc_sum_1 = 0
f1_sum_1 = 0

acc_sum_4 = 0
f1_sum_4 = 0

acc_sum_2 = 0
f1_sum_2 = 0

print(pd.unique(run_df['class']))
for fold, (training_indices, validation_indices) in enumerate(cv.split(run_df['runId'], run_df['class'])):
    # print("Fold: ", fold)

    training_runIds = run_df.loc[training_indices]['runId']
    validation_runIds = run_df.loc[validation_indices]['runId']

    X_train_df = df[df['runId'].isin(training_runIds)].copy()
    X_val_df = df[df['runId'].isin(validation_runIds)].copy()

    X_train_df = X_train_df[sensor_list + ["class", "runId"]].copy()
    X_val_df = X_val_df[sensor_list + ["class", "runId"]].copy()

    X_train, y_train = create_datasets(X_train_df, ws)
    X_val, y_val = create_datasets(X_val_df, ws)

    # pca = PCA(n_components=0.95)
    # X_train = pca.fit_transform(X_train)
    # X_val = pca.transform(X_val)

    lda = LinearDiscriminantAnalysis()
    X_train = lda.fit_transform(X_train, y_train)
    X_val = lda.transform(X_val)

    print("X_train_df,  X_train:", X_train_df.shape, X_train.shape)

    # model1 = LGBMClassifier(random_state=41)
    model1 = OneVsOneClassifier(LGBMClassifier(random_state=41))
    model1.fit(X_train, y_train)

    pred = model1.predict(X_val)

    acc_val1 = accuracy_score(y_val, pred)
    f1_val1 = f1_score(y_val, pred, average='weighted')
    cm1 = confusion_matrix(y_val, pred)

    df_result = pd.DataFrame()
    df_result["actual"] = y_val
    df_result["pred"] = pred
    df_result.to_excel("lgb_fold_{}_{}.xlsx".format(fold, model_property))
    with open('lgb_fold_{}_{}_.pickle'.format(fold, model_property), 'wb') as handle:
        pickle.dump(model1, handle, protocol=pickle.HIGHEST_PROTOCOL)

    acc_sum_1 += acc_val1
    f1_sum_1 += f1_val1

    print(cm1)
    print("LightGBM Fold:", fold, "ACC:", acc_val1, "F1:", f1_val1)
    print("----------------------------------------------------------")

    # model2 = SVC(random_state=41)  # OneVsOneClassifier(LGBMClassifier(random_state=41))
    # model2.fit(X_train, y_train)
    #
    # pred = model2.predict(X_val)
    #
    # acc_val2 = accuracy_score(y_val, pred)
    # f1_val2 = f1_score(y_val, pred, average='weighted')
    # cm2 = confusion_matrix(y_val, pred)
    #
    # df_result = pd.DataFrame()
    # df_result["actual"] = y_val
    # df_result["pred"] = pred
    # df_result.to_excel("svc_fold_{}_{}.xlsx".format(fold, model_property))
    # with open('svc_fold_{}_{}_.pickle'.format(fold, model_property), 'wb') as handle:
    #     pickle.dump(model1, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #
    # acc_sum_2 += acc_val2
    # f1_sum_2 += f1_val2
    #
    # print(cm2)
    # print("SVC Fold:", fold, "ACC:", acc_val2, "F1:", f1_val2)

    param = {
        'objective': 'multi:softprob',  # error evaluation for multiclass training
        'num_class': len(pd.unique(y_train))}  # the number of classes that exist in this datset
    # model4 = OneVsOneClassifier(XGBClassifier(param, random_state=41, verbosity=0))
    model4 = XGBClassifier(param, random_state=41, verbosity=0)
    model4.fit(X_train, y_train)

    pred = model4.predict(X_val)

    acc_val4 = accuracy_score(y_val, pred)
    f1_val4 = f1_score(y_val, pred, average='weighted')
    cm4 = confusion_matrix(y_val, pred)

    acc_sum_4 += acc_val4
    f1_sum_4 += f1_val4
    print(cm4)
    print("XGBClassifier Fold:", fold, "ACC:", acc_val4, "F1:", f1_val4)

    df_result = pd.DataFrame()
    df_result["actual"] = y_val
    df_result["pred"] = pred
    df_result.to_excel("xgb_fold_{}_{}.xlsx".format(fold, model_property))
    with open('xgb_fold_{}_{}_.pickle'.format(fold, model_property), 'wb') as handle:
        pickle.dump(model4, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("LightGBM Avg ACC:", acc_sum_1 / fold_num, "Avg F1:", f1_sum_1 / fold_num)
# print("SVC Avg ACC:", acc_sum_2 / fold_num, "Avg F1:", f1_sum_2 / fold_num)
print("XGBClassifier Avg ACC:", acc_sum_4 / fold_num, "Avg F1:", f1_sum_4 / fold_num)

"""
with lda
(57971, 207)
fold_num: 3 168 [('Temperature_value', 0.002083508379755757), ('Humidity_value', 0.0019965867611539956), ('LightBarrierActiveTaskDuration1_vMax', 0.0011226723303719988), ('LightBarrierActiveTaskDuration1_vFreq', 0.001038187969761591), ('SmartMotorSpeed_vTrend', 0.0010168188480015328), ('DurationPickToPick_value', 0.0009976703361661565), ('Pressure_vStd', 0.000973858367136556), ('VacuumFusePicked_vStd', 0.0009368374658364301), ('EPOSVelocity_vStd', 0.0008707961462408598), ('FusePicked_vMin', 0.0008438838658729372), ('VacuumFusePicked_vTrend', 0.0008070312203783958), ('TotalMemoryConsumption_vStd', 0.000789727825846002), ('IntensityTotalThermoImage_vCnt', 0.0007854227428496877), ('ProcessCpuLoadNormalized_vMax', 0.0007500097754290872), ('SmartMotorPositionError_vMax', 0.0007478189657127432), ('TotalMemoryConsumption_vMin', 0.0007430787627330515), ('TemperatureThermoCam_vFreq', 0.0006983435975816225), ('ValidFrame_vFreq', 0.0006888356165436113), ('LightBarrierPassiveTaskDuration1_vFreq', 0.000648904920755052), ('FuseHeatSlopeOK_vFreq', 0.0006043099152935513), ('DurationRobotFromFeederToTestBench_value', 0.0006002509815666857), ('FuseOutsideOperationalSpace_vMax', 0.000596075378702432), ('FuseHeatSlopeNOK_vMax', 0.0005859452579851654), ('Vacuum_vStd', 0.0005785659995580783), ('VacuumValveClosed_vTrend', 0.0005739837580593191), ('IntensityTotalImage_vCnt', 0.0005676559530248445), ('VacuumValveClosed_vMin', 0.0005616942147530132), ('FeederAction1_vCnt', 0.0005604288523016384), ('VacuumValveClosed_vCnt', 0.0005578213072400251), ('FuseHeatSlopeNOK_value', 0.0005487039698534035), ('IntensityTotalThermoImage_vFreq', 0.0005392012438445818), ('ProcessMemoryConsumption_vMin', 0.0005387205830975997), ('SharpnessImage_vFreq', 0.0005294476260450098), ('FusePicked_vTrend', 0.000528087128991587), ('EPOSVelocity_vMin', 0.0005272978742871063), ('LightBarrierPassiveTaskDuration1_value', 0.0005267128080874336), ('CpuTemperature_vStd', 0.0005137701680684911), ('EPOSCurrent_vFreq', 0.0005108426294746135), ('EPOSPosition_vCnt', 0.0004969695957578827), ('FuseHeatSlope_vFreq', 0.0004890896133750896), ('EPOSCurrent_vTrend', 0.00048731542353404783), ('IntensityTotalThermoImage_vMin', 0.00048729322145505777), ('DurationRobotFromTestBenchToFeeder_vCnt', 0.00048664735677102683), ('FusePicked_vMax', 0.00048624199817374575), ('DurationTestBenchClosed_value', 0.00048552395606445603), ('IntensityTotalThermoImage_vStd', 0.00048265104147308485), ('DurationRobotFromFeederToTestBench_vStd', 0.0004805037757118491), ('LightBarrierActiveTaskDuration1_vMin', 0.00046475192401658944), ('LightBarrierPassiveTaskDuration1_vMax', 0.0004619028105106171), ('TotalCpuLoadNormalized_vMin', 0.0004611253103253092), ('Vacuum_vMin', 0.0004575245276037994), ('LightBarrierActiveTaskDuration1_value', 0.0004486001151145687), ('Vacuum_vMax', 0.0004397022141640061), ('NumberFuseDetected_vCnt', 0.00043906126532726003), ('DurationRobotFromFeederToTestBench_vMin', 0.0004366159994330839), ('IntensityTotalThermoImage_vTrend', 0.00042703458701150626), ('FuseHeatSlope_value', 0.00042507177004132607), ('SmartMotorPositionError_vMin', 0.00042109801120511953), ('Vacuum_vTrend', 0.00041253051509108446), ('TemperatureThermoCam_vMax', 0.0004048906405595594), ('IntensityTotalThermoImage_vMax', 0.0004048615863461909), ('LightBarrierActiveTaskDuration1_vCnt', 0.0004042536443821613), ('ProcessMemoryConsumption_vMax', 0.0004038743763895569), ('VacuumFusePicked_value', 0.00040380298754294497), ('CpuTemperature_value', 0.00038667919338419843), ('DurationTestBenchClosed_vTrend', 0.00038225409335187105), ('TemperatureThermoCam_vMin', 0.00037997499082035685), ('FuseOutsideOperationalSpace_value', 0.0003781324326128284), ('FuseHeatSlope_vCnt', 0.00037188906214769), ('SmartMotorSpeed_vFreq', 0.00036627008082960416), ('TemperatureThermoCam_value', 0.00035993224599410273), ('LightBarrierPassiveTaskDuration1_vMin', 0.00035463760913786074), ('DurationRobotFromFeederToTestBench_vTrend', 0.00035087370389715505), ('FuseCycleDuration_vMax', 0.0003444634105397674), ('NumberFuseEstimated_vCnt', 0.00034168854980406077), ('TemperatureThermoCam_vCnt', 0.0003394536863697839), ('EPOSCurrent_vStd', 0.0003392811214768532), ('FeederBackgroundIlluminationIntensity_vFreq', 0.0003359854921001955), ('FeederAction3_vCnt', 0.0003348235190061555), ('LightBarrierPassiveTaskDuration1_vTrend', 0.00033440118671616376), ('SmartMotorPositionError_vTrend', 0.00033136255315979746), ('EPOSVelocity_value', 0.00033092327553855644), ('FuseHeatSlopeNOK_vFreq', 0.00032365376300909815), ('EPOSPosition_vTrend', 0.00032285154723147674), ('Pressure_vFreq', 0.0003181806273295414), ('TotalCpuLoadNormalized_value', 0.0003176697704787568), ('FuseCycleDuration_value', 0.0003176093758547512), ('SharpnessImage_vCnt', 0.00031606059493786515), ('DurationTestBenchClosed_vCnt', 0.0003154578016942485), ('DurationRobotFromTestBenchToFeeder_vFreq', 0.00030406556192996703), ('FuseHeatSlope_vMin', 0.00030155417201960244), ('DurationPickToPick_vStd', 0.0002995365397349496), ('DurationTestBenchClosed_vFreq', 0.0002986692940962395), ('DurationRobotFromFeederToTestBench_vCnt', 0.0002983016017201005), ('LightBarrierActiveTaskDuration1_vTrend', 0.00029228821677588596), ('IntensityTotalImage_vFreq', 0.00028735904903121057), ('FusePicked_vStd', 0.0002871047485348092), ('SmartMotorSpeed_vCnt', 0.00027849298805467565), ('SmartMotorPositionError_value', 0.0002736961172195758), ('ProcessCpuLoadNormalized_vMin', 0.0002736135385741356), ('IntensityTotalThermoImage_value', 0.0002688916201881719), ('FuseOutsideOperationalSpace_vFreq', 0.000263796987669096), ('TemperatureThermoCam_vTrend', 0.00025566416051392427), ('DurationPickToPick_vTrend', 0.00024147879766234226), ('FuseHeatSlope_vTrend', 0.000235095620162958), ('FuseTestResult_vTrend', 0.00022763489476414675), ('DurationRobotFromTestBenchToFeeder_vMin', 0.00022694783203214275), ('FuseTestResult_value', 0.0002252453239844169), ('Vacuum_vFreq', 0.00022247029468602797), ('ProcessMemoryConsumption_vStd', 0.0002224335733237437), ('DurationRobotFromFeederToTestBench_vMax', 0.00022074770243540875), ('VacuumValveClosed_vStd', 0.00022025302130590507), ('EPOSPosition_vMax', 0.0002163687642791808), ('EPOSPosition_vStd', 0.00021412167096420318), ('DurationTestBenchClosed_vMin', 0.00020664888775327275), ('Vacuum_vCnt', 0.00020490327965346408), ('EPOSPosition_vMin', 0.00020276174577482653), ('DurationRobotFromTestBenchToFeeder_vMax', 0.00018822570450812334), ('FuseTestResult_vStd', 0.00018632620660145408), ('FeederBackgroundIlluminationIntensity_vCnt', 0.00018523582879011644), ('ErrorFrame_vCnt', 0.00018216250480133459), ('SmartMotorPositionError_vStd', 0.0001821248704257187), ('SmartMotorSpeed_value', 0.00017557155008692446), ('FuseOutsideOperationalSpace_vStd', 0.00017172776016212884), ('LightBarrierActiveTaskDuration1_vStd', 0.00017068046461428862), ('DurationPickToPick_vCnt', 0.0001671605611965147), ('VacuumValveClosed_vMax', 0.0001665726878065099), ('FuseTestResult_vMin', 0.0001634847247032519), ('EPOSVelocity_vCnt', 0.00015378159742766387), ('EPOSCurrent_vMax', 0.00015261576688578118), ('TotalCpuLoadNormalized_vStd', 0.00014932434698296712), ('EPOSCurrent_vCnt', 0.00014921814894623223), ('NumberEmptyFeeder_vCnt', 0.00014626996756343194), ('ProcessCpuLoadNormalized_value', 0.00014456995121958638), ('FuseCycleDuration_vStd', 0.00013859298775657317), ('EPOSCurrent_value', 0.0001355392365557373), ('FuseCycleDuration_vMin', 0.00013246377804487786), ('FusePicked_vCnt', 0.00013206929475987708), ('DurationRobotFromTestBenchToFeeder_vTrend', 0.00013092391431213102), ('FusePicked_value', 0.00012912391750097285), ('VacuumFusePicked_vCnt', 0.00012403767471791394), ('Pressure_vCnt', 0.00012065917548764737), ('Pressure_vTrend', 0.00011845956821354697), ('FeederAction2_vCnt', 0.0001130030297322356), ('VacuumValveClosed_value', 0.00011168906548701212), ('FuseHeatSlopeNOK_vCnt', 0.00011052870522406295), ('NumberFuseEstimated_vFreq', 0.0001042490168406222), ('FuseCycleDuration_vFreq', 9.200589438777367e-05), ('FuseIntoFeeder_vCnt', 8.44430756455165e-05), ('ValidFrame_vCnt', 8.00755478943908e-05), ('VacuumFusePicked_vMin', 7.71795985602397e-05), ('Vacuum_value', 7.465711910422297e-05), ('FeederAction4_vCnt', 6.810886853292342e-05), ('DurationTestBenchClosed_vMax', 6.79216958561953e-05), ('LightBarrierPassiveTaskDuration1_vStd', 6.184642127034697e-05), ('SmartMotorPositionError_vCnt', 5.15874509154172e-05), ('DurationRobotFromTestBenchToFeeder_value', 5.1538357237257415e-05), ('ErrorFrame_vFreq', 4.647492987582454e-05), ('LightBarrierPassiveTaskDuration1_vCnt', 4.572901986821254e-05), ('ProcessMemoryConsumption_value', 3.765466338005267e-05), ('FuseCycleDuration_vTrend', 3.6911169585115054e-05), ('TemperatureThermoCam_vStd', 3.6447589428741565e-05), ('EPOSCurrent_vMin', 3.531878700950101e-05), ('FuseTestResult_vCnt', 2.8659716869753698e-05), ('EPOSVelocity_vTrend', 2.457690899904108e-05), ('ProcessCpuLoadNormalized_vStd', 2.3710204330229523e-05), ('SmartMotorPositionError_vFreq', 1.103028019970124e-05), ('FuseHeatSlopeOK_vCnt', 6.133520711622727e-06)]
['Temperature_value', 'Humidity_value', 'LightBarrierActiveTaskDuration1_vMax', 'LightBarrierActiveTaskDuration1_vFreq', 'SmartMotorSpeed_vTrend', 'DurationPickToPick_value', 'Pressure_vStd', 'VacuumFusePicked_vStd', 'EPOSVelocity_vStd', 'FusePicked_vMin', 'VacuumFusePicked_vTrend', 'TotalMemoryConsumption_vStd', 'IntensityTotalThermoImage_vCnt', 'ProcessCpuLoadNormalized_vMax', 'SmartMotorPositionError_vMax', 'TotalMemoryConsumption_vMin', 'TemperatureThermoCam_vFreq', 'ValidFrame_vFreq', 'LightBarrierPassiveTaskDuration1_vFreq', 'FuseHeatSlopeOK_vFreq', 'DurationRobotFromFeederToTestBench_value', 'FuseOutsideOperationalSpace_vMax', 'FuseHeatSlopeNOK_vMax', 'Vacuum_vStd', 'VacuumValveClosed_vTrend', 'IntensityTotalImage_vCnt', 'VacuumValveClosed_vMin', 'FeederAction1_vCnt', 'VacuumValveClosed_vCnt', 'FuseHeatSlopeNOK_value', 'IntensityTotalThermoImage_vFreq', 'ProcessMemoryConsumption_vMin', 'SharpnessImage_vFreq', 'FusePicked_vTrend', 'EPOSVelocity_vMin', 'LightBarrierPassiveTaskDuration1_value', 'CpuTemperature_vStd', 'EPOSCurrent_vFreq', 'EPOSPosition_vCnt', 'FuseHeatSlope_vFreq', 'EPOSCurrent_vTrend', 'IntensityTotalThermoImage_vMin', 'DurationRobotFromTestBenchToFeeder_vCnt', 'FusePicked_vMax', 'DurationTestBenchClosed_value', 'IntensityTotalThermoImage_vStd', 'DurationRobotFromFeederToTestBench_vStd', 'LightBarrierActiveTaskDuration1_vMin', 'LightBarrierPassiveTaskDuration1_vMax', 'TotalCpuLoadNormalized_vMin', 'Vacuum_vMin', 'LightBarrierActiveTaskDuration1_value', 'Vacuum_vMax', 'NumberFuseDetected_vCnt', 'DurationRobotFromFeederToTestBench_vMin', 'IntensityTotalThermoImage_vTrend', 'FuseHeatSlope_value', 'SmartMotorPositionError_vMin', 'Vacuum_vTrend', 'TemperatureThermoCam_vMax', 'IntensityTotalThermoImage_vMax', 'LightBarrierActiveTaskDuration1_vCnt', 'ProcessMemoryConsumption_vMax', 'VacuumFusePicked_value', 'CpuTemperature_value', 'DurationTestBenchClosed_vTrend', 'TemperatureThermoCam_vMin', 'FuseOutsideOperationalSpace_value', 'FuseHeatSlope_vCnt', 'SmartMotorSpeed_vFreq', 'TemperatureThermoCam_value', 'LightBarrierPassiveTaskDuration1_vMin', 'DurationRobotFromFeederToTestBench_vTrend', 'FuseCycleDuration_vMax', 'NumberFuseEstimated_vCnt', 'TemperatureThermoCam_vCnt', 'EPOSCurrent_vStd', 'FeederBackgroundIlluminationIntensity_vFreq', 'FeederAction3_vCnt', 'LightBarrierPassiveTaskDuration1_vTrend', 'SmartMotorPositionError_vTrend', 'EPOSVelocity_value', 'FuseHeatSlopeNOK_vFreq', 'EPOSPosition_vTrend', 'Pressure_vFreq', 'TotalCpuLoadNormalized_value', 'FuseCycleDuration_value', 'SharpnessImage_vCnt', 'DurationTestBenchClosed_vCnt', 'DurationRobotFromTestBenchToFeeder_vFreq', 'FuseHeatSlope_vMin', 'DurationPickToPick_vStd', 'DurationTestBenchClosed_vFreq', 'DurationRobotFromFeederToTestBench_vCnt', 'LightBarrierActiveTaskDuration1_vTrend', 'IntensityTotalImage_vFreq', 'FusePicked_vStd', 'SmartMotorSpeed_vCnt', 'SmartMotorPositionError_value', 'ProcessCpuLoadNormalized_vMin', 'IntensityTotalThermoImage_value', 'FuseOutsideOperationalSpace_vFreq', 'TemperatureThermoCam_vTrend', 'DurationPickToPick_vTrend', 'FuseHeatSlope_vTrend', 'FuseTestResult_vTrend', 'DurationRobotFromTestBenchToFeeder_vMin', 'FuseTestResult_value', 'Vacuum_vFreq', 'ProcessMemoryConsumption_vStd', 'DurationRobotFromFeederToTestBench_vMax', 'VacuumValveClosed_vStd', 'EPOSPosition_vMax', 'EPOSPosition_vStd', 'DurationTestBenchClosed_vMin', 'Vacuum_vCnt', 'EPOSPosition_vMin', 'DurationRobotFromTestBenchToFeeder_vMax', 'FuseTestResult_vStd', 'FeederBackgroundIlluminationIntensity_vCnt', 'ErrorFrame_vCnt', 'SmartMotorPositionError_vStd', 'SmartMotorSpeed_value', 'FuseOutsideOperationalSpace_vStd', 'LightBarrierActiveTaskDuration1_vStd', 'DurationPickToPick_vCnt', 'VacuumValveClosed_vMax', 'FuseTestResult_vMin', 'EPOSVelocity_vCnt', 'EPOSCurrent_vMax', 'TotalCpuLoadNormalized_vStd', 'EPOSCurrent_vCnt', 'NumberEmptyFeeder_vCnt', 'ProcessCpuLoadNormalized_value', 'FuseCycleDuration_vStd', 'EPOSCurrent_value', 'FuseCycleDuration_vMin', 'FusePicked_vCnt', 'DurationRobotFromTestBenchToFeeder_vTrend', 'FusePicked_value', 'VacuumFusePicked_vCnt', 'Pressure_vCnt', 'Pressure_vTrend', 'FeederAction2_vCnt', 'VacuumValveClosed_value', 'FuseHeatSlopeNOK_vCnt', 'NumberFuseEstimated_vFreq', 'FuseCycleDuration_vFreq', 'FuseIntoFeeder_vCnt', 'ValidFrame_vCnt', 'VacuumFusePicked_vMin', 'Vacuum_value', 'FeederAction4_vCnt', 'DurationTestBenchClosed_vMax', 'LightBarrierPassiveTaskDuration1_vStd', 'SmartMotorPositionError_vCnt', 'DurationRobotFromTestBenchToFeeder_value', 'ErrorFrame_vFreq', 'LightBarrierPassiveTaskDuration1_vCnt', 'ProcessMemoryConsumption_value', 'FuseCycleDuration_vTrend', 'TemperatureThermoCam_vStd', 'EPOSCurrent_vMin', 'FuseTestResult_vCnt', 'EPOSVelocity_vTrend', 'ProcessCpuLoadNormalized_vStd', 'SmartMotorPositionError_vFreq', 'FuseHeatSlopeOK_vCnt']
-------------------------------------------------------------
ws_30_fn_168
[ 0  2  3  5  7  9  4 11 12]
X_train_df,  X_train: (35269, 170) (33471, 8)
[[11385    23     3  1304    55    29   149    33    15]
 [   24   526     0    81     0     8     8     0    16]
 [   95     0   570     0     0     0     1     0     0]
 [  331     0     0     0     0     0     0     0     0]
 [  220     2     0     5  1747    82     0     1    42]
 [  341     3     0     0     0  1018    24     0     5]
 [  145     0     0     0     0     0  1238     0     0]
 [  864     0     0     0     0     0     1   186     0]
 [   29     0     0    85     0     1     0     0   934]]
LightGBM Fold: 0 ACC: 0.8139072541495215 F1: 0.8258203863004259
----------------------------------------------------------
[[11523    35     5  1289    29     3   103     9     0]
 [   20   505     0   119     0    19     0     0     0]
 [    6     0   660     0     0     0     0     0     0]
 [  331     0     0     0     0     0     0     0     0]
 [    0     0     0     9  2089     0     0     1     0]
 [  335     2     1     0     0  1051     0     0     2]
 [  347     0     0     0     0     0  1036     0     0]
 [    0     0     0     0     0     0     0  1051     0]
 [    0     0     0     1     0     0     0     0  1048]]
XGBClassifier Fold: 0 ACC: 0.8767395626242545 F1: 0.8983478505244471
X_train_df,  X_train: (41780, 170) (39808, 8)
[[9277   19   15  391    3   79    5    0   11]
 [  92  838    0   76    0    0    0    0    0]
 [  10    0 1041    0    0    0    0    0    0]
 [1047    0    0    1    0    0    0    0    0]
 [   1    0    0    0  331    0    0    0    0]
 [   4    0    0    0    0  328    0    0    0]
 [ 309    0    0    0    0    0   31    0    0]
 [   0    0    0    0    0    3    0  329    0]
 [  51    0    0    1    0    0    3    0  996]]
LightGBM Fold: 1 ACC: 0.8613654198273607 F1: 0.8371949685876358
----------------------------------------------------------
[[9464    7    0  222    6   91    7    0    3]
 [  59  919    0   27    0    0    0    0    1]
 [   0    0 1051    0    0    0    0    0    0]
 [1047    0    0    1    0    0    0    0    0]
 [   0    0    0    0  332    0    0    0    0]
 [   0    0    0    0    0  332    0    0    0]
 [ 315    0    0    0    5    0   20    0    0]
 [   0    0    0    0    0    0    0  332    0]
 [   0    0    0    1    1    0    0    0 1049]]
XGBClassifier Fold: 1 ACC: 0.8828145435521841 F1: 0.8509284880079387
X_train_df,  X_train: (38893, 170) (36921, 8)
[[11150    34     8    58     1    47   639     0     0]
 [   12  1027    11     0     0     0     0     0     0]
 [    2     0  1046     0     0     1     1     0     0]
 [  260     8     9     0     0    55     0     0     0]
 [   57     0     0     0   274     0     0     0     0]
 [    3     0     0     0     0  1048     0     0     0]
 [  200     0     0     0     0     0   850     0     0]
 [   92     0     0     0     0     0     0   957     0]
 [   63     0     0     2     0     0     0     0   264]]
LightGBM Fold: 2 ACC: 0.9140216733593707 F1: 0.9100743267380373
----------------------------------------------------------
[[11191    32     0    45     1    77   591     0     0]
 [   11  1039     0     0     0     0     0     0     0]
 [    2     0  1045     0     0     2     1     0     0]
 [  246    11     0     0     0    75     0     0     0]
 [   53     0     0     0   278     0     0     0     0]
 [    2     0     0     0     0  1049     0     0     0]
 [  202     0     0     0     0     0   848     0     0]
 [    0     0     0     0     0     0     0  1049     0]
 [    0     0     0     0     0     0     0     0   329]]
XGBClassifier Fold: 2 ACC: 0.9256834809395457 F1: 0.9210572759888632
LightGBM Avg ACC: 0.863098115778751 Avg F1: 0.857696560542033
XGBClassifier Avg ACC: 0.895079195705328 Avg F1: 0.8901112048404163
"""


"""
with pca
(57971, 207)
fold_num: 3 168 [('Temperature_value', 0.002083508379755757), ('Humidity_value', 0.0019965867611539956), ('LightBarrierActiveTaskDuration1_vMax', 0.0011226723303719988), ('LightBarrierActiveTaskDuration1_vFreq', 0.001038187969761591), ('SmartMotorSpeed_vTrend', 0.0010168188480015328), ('DurationPickToPick_value', 0.0009976703361661565), ('Pressure_vStd', 0.000973858367136556), ('VacuumFusePicked_vStd', 0.0009368374658364301), ('EPOSVelocity_vStd', 0.0008707961462408598), ('FusePicked_vMin', 0.0008438838658729372), ('VacuumFusePicked_vTrend', 0.0008070312203783958), ('TotalMemoryConsumption_vStd', 0.000789727825846002), ('IntensityTotalThermoImage_vCnt', 0.0007854227428496877), ('ProcessCpuLoadNormalized_vMax', 0.0007500097754290872), ('SmartMotorPositionError_vMax', 0.0007478189657127432), ('TotalMemoryConsumption_vMin', 0.0007430787627330515), ('TemperatureThermoCam_vFreq', 0.0006983435975816225), ('ValidFrame_vFreq', 0.0006888356165436113), ('LightBarrierPassiveTaskDuration1_vFreq', 0.000648904920755052), ('FuseHeatSlopeOK_vFreq', 0.0006043099152935513), ('DurationRobotFromFeederToTestBench_value', 0.0006002509815666857), ('FuseOutsideOperationalSpace_vMax', 0.000596075378702432), ('FuseHeatSlopeNOK_vMax', 0.0005859452579851654), ('Vacuum_vStd', 0.0005785659995580783), ('VacuumValveClosed_vTrend', 0.0005739837580593191), ('IntensityTotalImage_vCnt', 0.0005676559530248445), ('VacuumValveClosed_vMin', 0.0005616942147530132), ('FeederAction1_vCnt', 0.0005604288523016384), ('VacuumValveClosed_vCnt', 0.0005578213072400251), ('FuseHeatSlopeNOK_value', 0.0005487039698534035), ('IntensityTotalThermoImage_vFreq', 0.0005392012438445818), ('ProcessMemoryConsumption_vMin', 0.0005387205830975997), ('SharpnessImage_vFreq', 0.0005294476260450098), ('FusePicked_vTrend', 0.000528087128991587), ('EPOSVelocity_vMin', 0.0005272978742871063), ('LightBarrierPassiveTaskDuration1_value', 0.0005267128080874336), ('CpuTemperature_vStd', 0.0005137701680684911), ('EPOSCurrent_vFreq', 0.0005108426294746135), ('EPOSPosition_vCnt', 0.0004969695957578827), ('FuseHeatSlope_vFreq', 0.0004890896133750896), ('EPOSCurrent_vTrend', 0.00048731542353404783), ('IntensityTotalThermoImage_vMin', 0.00048729322145505777), ('DurationRobotFromTestBenchToFeeder_vCnt', 0.00048664735677102683), ('FusePicked_vMax', 0.00048624199817374575), ('DurationTestBenchClosed_value', 0.00048552395606445603), ('IntensityTotalThermoImage_vStd', 0.00048265104147308485), ('DurationRobotFromFeederToTestBench_vStd', 0.0004805037757118491), ('LightBarrierActiveTaskDuration1_vMin', 0.00046475192401658944), ('LightBarrierPassiveTaskDuration1_vMax', 0.0004619028105106171), ('TotalCpuLoadNormalized_vMin', 0.0004611253103253092), ('Vacuum_vMin', 0.0004575245276037994), ('LightBarrierActiveTaskDuration1_value', 0.0004486001151145687), ('Vacuum_vMax', 0.0004397022141640061), ('NumberFuseDetected_vCnt', 0.00043906126532726003), ('DurationRobotFromFeederToTestBench_vMin', 0.0004366159994330839), ('IntensityTotalThermoImage_vTrend', 0.00042703458701150626), ('FuseHeatSlope_value', 0.00042507177004132607), ('SmartMotorPositionError_vMin', 0.00042109801120511953), ('Vacuum_vTrend', 0.00041253051509108446), ('TemperatureThermoCam_vMax', 0.0004048906405595594), ('IntensityTotalThermoImage_vMax', 0.0004048615863461909), ('LightBarrierActiveTaskDuration1_vCnt', 0.0004042536443821613), ('ProcessMemoryConsumption_vMax', 0.0004038743763895569), ('VacuumFusePicked_value', 0.00040380298754294497), ('CpuTemperature_value', 0.00038667919338419843), ('DurationTestBenchClosed_vTrend', 0.00038225409335187105), ('TemperatureThermoCam_vMin', 0.00037997499082035685), ('FuseOutsideOperationalSpace_value', 0.0003781324326128284), ('FuseHeatSlope_vCnt', 0.00037188906214769), ('SmartMotorSpeed_vFreq', 0.00036627008082960416), ('TemperatureThermoCam_value', 0.00035993224599410273), ('LightBarrierPassiveTaskDuration1_vMin', 0.00035463760913786074), ('DurationRobotFromFeederToTestBench_vTrend', 0.00035087370389715505), ('FuseCycleDuration_vMax', 0.0003444634105397674), ('NumberFuseEstimated_vCnt', 0.00034168854980406077), ('TemperatureThermoCam_vCnt', 0.0003394536863697839), ('EPOSCurrent_vStd', 0.0003392811214768532), ('FeederBackgroundIlluminationIntensity_vFreq', 0.0003359854921001955), ('FeederAction3_vCnt', 0.0003348235190061555), ('LightBarrierPassiveTaskDuration1_vTrend', 0.00033440118671616376), ('SmartMotorPositionError_vTrend', 0.00033136255315979746), ('EPOSVelocity_value', 0.00033092327553855644), ('FuseHeatSlopeNOK_vFreq', 0.00032365376300909815), ('EPOSPosition_vTrend', 0.00032285154723147674), ('Pressure_vFreq', 0.0003181806273295414), ('TotalCpuLoadNormalized_value', 0.0003176697704787568), ('FuseCycleDuration_value', 0.0003176093758547512), ('SharpnessImage_vCnt', 0.00031606059493786515), ('DurationTestBenchClosed_vCnt', 0.0003154578016942485), ('DurationRobotFromTestBenchToFeeder_vFreq', 0.00030406556192996703), ('FuseHeatSlope_vMin', 0.00030155417201960244), ('DurationPickToPick_vStd', 0.0002995365397349496), ('DurationTestBenchClosed_vFreq', 0.0002986692940962395), ('DurationRobotFromFeederToTestBench_vCnt', 0.0002983016017201005), ('LightBarrierActiveTaskDuration1_vTrend', 0.00029228821677588596), ('IntensityTotalImage_vFreq', 0.00028735904903121057), ('FusePicked_vStd', 0.0002871047485348092), ('SmartMotorSpeed_vCnt', 0.00027849298805467565), ('SmartMotorPositionError_value', 0.0002736961172195758), ('ProcessCpuLoadNormalized_vMin', 0.0002736135385741356), ('IntensityTotalThermoImage_value', 0.0002688916201881719), ('FuseOutsideOperationalSpace_vFreq', 0.000263796987669096), ('TemperatureThermoCam_vTrend', 0.00025566416051392427), ('DurationPickToPick_vTrend', 0.00024147879766234226), ('FuseHeatSlope_vTrend', 0.000235095620162958), ('FuseTestResult_vTrend', 0.00022763489476414675), ('DurationRobotFromTestBenchToFeeder_vMin', 0.00022694783203214275), ('FuseTestResult_value', 0.0002252453239844169), ('Vacuum_vFreq', 0.00022247029468602797), ('ProcessMemoryConsumption_vStd', 0.0002224335733237437), ('DurationRobotFromFeederToTestBench_vMax', 0.00022074770243540875), ('VacuumValveClosed_vStd', 0.00022025302130590507), ('EPOSPosition_vMax', 0.0002163687642791808), ('EPOSPosition_vStd', 0.00021412167096420318), ('DurationTestBenchClosed_vMin', 0.00020664888775327275), ('Vacuum_vCnt', 0.00020490327965346408), ('EPOSPosition_vMin', 0.00020276174577482653), ('DurationRobotFromTestBenchToFeeder_vMax', 0.00018822570450812334), ('FuseTestResult_vStd', 0.00018632620660145408), ('FeederBackgroundIlluminationIntensity_vCnt', 0.00018523582879011644), ('ErrorFrame_vCnt', 0.00018216250480133459), ('SmartMotorPositionError_vStd', 0.0001821248704257187), ('SmartMotorSpeed_value', 0.00017557155008692446), ('FuseOutsideOperationalSpace_vStd', 0.00017172776016212884), ('LightBarrierActiveTaskDuration1_vStd', 0.00017068046461428862), ('DurationPickToPick_vCnt', 0.0001671605611965147), ('VacuumValveClosed_vMax', 0.0001665726878065099), ('FuseTestResult_vMin', 0.0001634847247032519), ('EPOSVelocity_vCnt', 0.00015378159742766387), ('EPOSCurrent_vMax', 0.00015261576688578118), ('TotalCpuLoadNormalized_vStd', 0.00014932434698296712), ('EPOSCurrent_vCnt', 0.00014921814894623223), ('NumberEmptyFeeder_vCnt', 0.00014626996756343194), ('ProcessCpuLoadNormalized_value', 0.00014456995121958638), ('FuseCycleDuration_vStd', 0.00013859298775657317), ('EPOSCurrent_value', 0.0001355392365557373), ('FuseCycleDuration_vMin', 0.00013246377804487786), ('FusePicked_vCnt', 0.00013206929475987708), ('DurationRobotFromTestBenchToFeeder_vTrend', 0.00013092391431213102), ('FusePicked_value', 0.00012912391750097285), ('VacuumFusePicked_vCnt', 0.00012403767471791394), ('Pressure_vCnt', 0.00012065917548764737), ('Pressure_vTrend', 0.00011845956821354697), ('FeederAction2_vCnt', 0.0001130030297322356), ('VacuumValveClosed_value', 0.00011168906548701212), ('FuseHeatSlopeNOK_vCnt', 0.00011052870522406295), ('NumberFuseEstimated_vFreq', 0.0001042490168406222), ('FuseCycleDuration_vFreq', 9.200589438777367e-05), ('FuseIntoFeeder_vCnt', 8.44430756455165e-05), ('ValidFrame_vCnt', 8.00755478943908e-05), ('VacuumFusePicked_vMin', 7.71795985602397e-05), ('Vacuum_value', 7.465711910422297e-05), ('FeederAction4_vCnt', 6.810886853292342e-05), ('DurationTestBenchClosed_vMax', 6.79216958561953e-05), ('LightBarrierPassiveTaskDuration1_vStd', 6.184642127034697e-05), ('SmartMotorPositionError_vCnt', 5.15874509154172e-05), ('DurationRobotFromTestBenchToFeeder_value', 5.1538357237257415e-05), ('ErrorFrame_vFreq', 4.647492987582454e-05), ('LightBarrierPassiveTaskDuration1_vCnt', 4.572901986821254e-05), ('ProcessMemoryConsumption_value', 3.765466338005267e-05), ('FuseCycleDuration_vTrend', 3.6911169585115054e-05), ('TemperatureThermoCam_vStd', 3.6447589428741565e-05), ('EPOSCurrent_vMin', 3.531878700950101e-05), ('FuseTestResult_vCnt', 2.8659716869753698e-05), ('EPOSVelocity_vTrend', 2.457690899904108e-05), ('ProcessCpuLoadNormalized_vStd', 2.3710204330229523e-05), ('SmartMotorPositionError_vFreq', 1.103028019970124e-05), ('FuseHeatSlopeOK_vCnt', 6.133520711622727e-06)]
['Temperature_value', 'Humidity_value', 'LightBarrierActiveTaskDuration1_vMax', 'LightBarrierActiveTaskDuration1_vFreq', 'SmartMotorSpeed_vTrend', 'DurationPickToPick_value', 'Pressure_vStd', 'VacuumFusePicked_vStd', 'EPOSVelocity_vStd', 'FusePicked_vMin', 'VacuumFusePicked_vTrend', 'TotalMemoryConsumption_vStd', 'IntensityTotalThermoImage_vCnt', 'ProcessCpuLoadNormalized_vMax', 'SmartMotorPositionError_vMax', 'TotalMemoryConsumption_vMin', 'TemperatureThermoCam_vFreq', 'ValidFrame_vFreq', 'LightBarrierPassiveTaskDuration1_vFreq', 'FuseHeatSlopeOK_vFreq', 'DurationRobotFromFeederToTestBench_value', 'FuseOutsideOperationalSpace_vMax', 'FuseHeatSlopeNOK_vMax', 'Vacuum_vStd', 'VacuumValveClosed_vTrend', 'IntensityTotalImage_vCnt', 'VacuumValveClosed_vMin', 'FeederAction1_vCnt', 'VacuumValveClosed_vCnt', 'FuseHeatSlopeNOK_value', 'IntensityTotalThermoImage_vFreq', 'ProcessMemoryConsumption_vMin', 'SharpnessImage_vFreq', 'FusePicked_vTrend', 'EPOSVelocity_vMin', 'LightBarrierPassiveTaskDuration1_value', 'CpuTemperature_vStd', 'EPOSCurrent_vFreq', 'EPOSPosition_vCnt', 'FuseHeatSlope_vFreq', 'EPOSCurrent_vTrend', 'IntensityTotalThermoImage_vMin', 'DurationRobotFromTestBenchToFeeder_vCnt', 'FusePicked_vMax', 'DurationTestBenchClosed_value', 'IntensityTotalThermoImage_vStd', 'DurationRobotFromFeederToTestBench_vStd', 'LightBarrierActiveTaskDuration1_vMin', 'LightBarrierPassiveTaskDuration1_vMax', 'TotalCpuLoadNormalized_vMin', 'Vacuum_vMin', 'LightBarrierActiveTaskDuration1_value', 'Vacuum_vMax', 'NumberFuseDetected_vCnt', 'DurationRobotFromFeederToTestBench_vMin', 'IntensityTotalThermoImage_vTrend', 'FuseHeatSlope_value', 'SmartMotorPositionError_vMin', 'Vacuum_vTrend', 'TemperatureThermoCam_vMax', 'IntensityTotalThermoImage_vMax', 'LightBarrierActiveTaskDuration1_vCnt', 'ProcessMemoryConsumption_vMax', 'VacuumFusePicked_value', 'CpuTemperature_value', 'DurationTestBenchClosed_vTrend', 'TemperatureThermoCam_vMin', 'FuseOutsideOperationalSpace_value', 'FuseHeatSlope_vCnt', 'SmartMotorSpeed_vFreq', 'TemperatureThermoCam_value', 'LightBarrierPassiveTaskDuration1_vMin', 'DurationRobotFromFeederToTestBench_vTrend', 'FuseCycleDuration_vMax', 'NumberFuseEstimated_vCnt', 'TemperatureThermoCam_vCnt', 'EPOSCurrent_vStd', 'FeederBackgroundIlluminationIntensity_vFreq', 'FeederAction3_vCnt', 'LightBarrierPassiveTaskDuration1_vTrend', 'SmartMotorPositionError_vTrend', 'EPOSVelocity_value', 'FuseHeatSlopeNOK_vFreq', 'EPOSPosition_vTrend', 'Pressure_vFreq', 'TotalCpuLoadNormalized_value', 'FuseCycleDuration_value', 'SharpnessImage_vCnt', 'DurationTestBenchClosed_vCnt', 'DurationRobotFromTestBenchToFeeder_vFreq', 'FuseHeatSlope_vMin', 'DurationPickToPick_vStd', 'DurationTestBenchClosed_vFreq', 'DurationRobotFromFeederToTestBench_vCnt', 'LightBarrierActiveTaskDuration1_vTrend', 'IntensityTotalImage_vFreq', 'FusePicked_vStd', 'SmartMotorSpeed_vCnt', 'SmartMotorPositionError_value', 'ProcessCpuLoadNormalized_vMin', 'IntensityTotalThermoImage_value', 'FuseOutsideOperationalSpace_vFreq', 'TemperatureThermoCam_vTrend', 'DurationPickToPick_vTrend', 'FuseHeatSlope_vTrend', 'FuseTestResult_vTrend', 'DurationRobotFromTestBenchToFeeder_vMin', 'FuseTestResult_value', 'Vacuum_vFreq', 'ProcessMemoryConsumption_vStd', 'DurationRobotFromFeederToTestBench_vMax', 'VacuumValveClosed_vStd', 'EPOSPosition_vMax', 'EPOSPosition_vStd', 'DurationTestBenchClosed_vMin', 'Vacuum_vCnt', 'EPOSPosition_vMin', 'DurationRobotFromTestBenchToFeeder_vMax', 'FuseTestResult_vStd', 'FeederBackgroundIlluminationIntensity_vCnt', 'ErrorFrame_vCnt', 'SmartMotorPositionError_vStd', 'SmartMotorSpeed_value', 'FuseOutsideOperationalSpace_vStd', 'LightBarrierActiveTaskDuration1_vStd', 'DurationPickToPick_vCnt', 'VacuumValveClosed_vMax', 'FuseTestResult_vMin', 'EPOSVelocity_vCnt', 'EPOSCurrent_vMax', 'TotalCpuLoadNormalized_vStd', 'EPOSCurrent_vCnt', 'NumberEmptyFeeder_vCnt', 'ProcessCpuLoadNormalized_value', 'FuseCycleDuration_vStd', 'EPOSCurrent_value', 'FuseCycleDuration_vMin', 'FusePicked_vCnt', 'DurationRobotFromTestBenchToFeeder_vTrend', 'FusePicked_value', 'VacuumFusePicked_vCnt', 'Pressure_vCnt', 'Pressure_vTrend', 'FeederAction2_vCnt', 'VacuumValveClosed_value', 'FuseHeatSlopeNOK_vCnt', 'NumberFuseEstimated_vFreq', 'FuseCycleDuration_vFreq', 'FuseIntoFeeder_vCnt', 'ValidFrame_vCnt', 'VacuumFusePicked_vMin', 'Vacuum_value', 'FeederAction4_vCnt', 'DurationTestBenchClosed_vMax', 'LightBarrierPassiveTaskDuration1_vStd', 'SmartMotorPositionError_vCnt', 'DurationRobotFromTestBenchToFeeder_value', 'ErrorFrame_vFreq', 'LightBarrierPassiveTaskDuration1_vCnt', 'ProcessMemoryConsumption_value', 'FuseCycleDuration_vTrend', 'TemperatureThermoCam_vStd', 'EPOSCurrent_vMin', 'FuseTestResult_vCnt', 'EPOSVelocity_vTrend', 'ProcessCpuLoadNormalized_vStd', 'SmartMotorPositionError_vFreq', 'FuseHeatSlopeOK_vCnt']
-------------------------------------------------------------
ws_30_fn_135
X_train_df,  X_train: (35269, 137) (33471, 115)
[[12766     5     4    24     3    93     1    52    48]
 [   77   568     4     0     8     3     0     0     3]
 [   21     0   644     0     0     1     0     0     0]
 [  331     0     0     0     0     0     0     0     0]
 [  293     0     0    49  1440   298     0     0    19]
 [  663     6     0     5     2   705     0     0    10]
 [   48     0     0     0    16     0  1315     0     4]
 [  354     0     0     0     0     0     0   697     0]
 [  658     0     0     9     1     3     0     0   378]]
LightGBM Fold: 0 ACC: 0.8559341624670581 F1: 0.8408809481994125
X_train_df,  X_train: (41780, 137) (39808, 102)
[[9473    3    2    3   66  113    0   49   91]
 [  91  870    0    0    1   35    0    0    9]
 [  11    0 1035    1    3    1    0    0    0]
 [1016    0    0    3    0    8    1    0   20]
 [   6    0    0    0  324    2    0    0    0]
 [ 266   11    0    0    1   54    0    0    0]
 [ 311    2    0    0    0    0   27    0    0]
 [ 317    0    0    0    2    0    0   12    1]
 [ 558    1    0    0    0    2    0    0  490]]
LightGBM Fold: 1 ACC: 0.8035574156421659 F1: 0.7534037343990471
X_train_df,  X_train: (38893, 137) (36921, 103)
[[10519     4    13    89    15    92   757    84   364]
 [   97   948     1     0     0     3     0     0     1]
 [    7    15  1028     0     0     0     0     0     0]
 [  298     1     7     0     4    13     0     0     9]
 [  229     0     0     0   101     1     0     0     0]
 [  306     2     3     3   121   599     0     0    17]
 [   91     0     0     1     0     0   955     0     3]
 [  160     0     0     0     0     0     1   888     0]
 [  314     0     3     0     1    11     0     0     0]]
LightGBM Fold: 2 ACC: 0.8272182188239177 F1: 0.8234192393123216
LightGBM Avg ACC: 0.8289032656443807 Avg F1: 0.8059013073035937
"""
