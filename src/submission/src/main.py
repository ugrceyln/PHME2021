# %%

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

# %%

# # Convert field names to dict for easy access.
# # Can be hard coded
#
# fields_path = '../data/training_validation_2/fields.csv'
# fields_df = pd.read_csv(fields_path)
# fields_df.columns = ['name', 'f0', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6']
#
# fields_dict = {}
#
# for idx in range(fields_df.shape[0]):
#     name = fields_df.loc[idx, 'name']
#
#     _fields = []
#
#     for f in fields_df.columns[1:]:
#         if not (str(fields_df.loc[idx, f]) == 'nan'):
#             _fields.append(name + "_" + str(fields_df.loc[idx, f]))
#
#     fields_dict[idx] = {'name': fields_df.loc[idx, 'name'], 'fields': _fields}
#
#
# # %%
#
# # Get class id and run id from filename
# def parse_class_name(fname):
#     p = re.compile("^class[^\d]*(\d+)_(\d+).*.csv")
#     m = p.match(fname)
#
#     return m.groups()
#
#
# # %%
#
# # Load one data file and return in a data frame
# def load_data_file(path, fname):
#     fullpath = join(path, fname)
#     df = pd.read_csv(fullpath)
#     df.columns = ['name', 'data']
#
#     dfx = []
#
#     for f in fields_dict:
#         name = fields_dict[f]['name']
#         fields = fields_dict[f]['fields']
#
#         data = eval(df.loc[f, 'data'])  # convert data to array
#
#         new_df = pd.DataFrame(data)
#         if (f == 33) and (new_df.shape[1] == 6):  # NumberFuseDetected has a special case!
#             new_df[6] = new_df[5]
#             new_df[5] = np.NaN
#
#         new_df.columns = fields_dict[f]['fields']
#
#         dfx.append(new_df)
#
#     merged_df = pd.concat(dfx, axis=1)  # Merge columns
#
#     c, r = parse_class_name(fname)  # Get class id and run id
#
#     # Add class labels and run id
#     merged_df['class'] = int(c)
#     merged_df['run'] = int(r)
#
#     return merged_df
#
#
# # %%
#
# # Load data files from a directory and return merged data frame
# def load_data_files(path):
#     print("In", path)
#     files = []
#     for f in listdir(path):
#         if (isfile(join(path, f)) and (f.startswith("class"))):
#             files.append(f)
#
#     data_df_list = []
#     for fname in files:
#         print("Loading:", fname)
#
#         df = load_data_file(path, fname)
#
#         data_df_list.append(df)
#
#     data_df = pd.concat(data_df_list, axis=0)  # Merge data frames
#
#     return data_df
#
# # %%
#
# data_df_1 = load_data_files("../data/training_validation_1/")
# data_df_2 = load_data_files("../data/training_validation_2/")
# data_df_3 = load_data_files("../data/ModelRefinement/")
#
# # %%
#
# data_df_1.to_csv("../data/training_validation_1.csv", index=False)
# data_df_2.to_csv("../data/training_validation_2.csv", index=False)
# data_df_3.to_csv("../data/model_refinement.csv", index=False)
#
# # %%
#
# data_df_1 = pd.read_csv("../data/training_validation_1.csv")
# data_df_2 = pd.read_csv("../data/training_validation_2.csv")
# data_df_3 = pd.read_csv("../data/model_refinement.csv")
#
# # %%
#
# data_df = pd.concat([data_df_1, data_df_2, data_df_3], axis=0, ignore_index=True)
#
#
# # %%
#
# def get_nan_info_table(data, fields):
#     field_df = data[fields]
#
#     return field_df.isnull().sum().T
#
#
# # %%
#
# df = data_df.copy()
# feature_sets = data_df.columns
# nan_info_table = df.groupby(["class", "run"]).apply(get_nan_info_table, feature_sets)
#
# plt.figure(figsize=(20, 20))
# sns.heatmap(nan_info_table.T, annot=False)
# plt.show()
#
# # %%
#
# droped_cols = []
# first_cols = df.columns.tolist()
#
# df = df.dropna(thresh=int(df.shape[0] * 0.7), axis=1)  # Drop column if it does not have at least x values that are **not** NaN
# print("col: ", df.shape)
# droped_cols.extend(list(set(first_cols).difference(df.columns)))
#
#
# # %%
#
# def drop_nan_and_unique_data(df):
#     for column in df.columns:
#         if column not in ["class", "run"]:
#             if (len(df[column].unique()) == 1) or (df[column].isnull().all()):
#                 df.drop(column, inplace=True, axis=1)
#                 droped_cols.append(column)
#                 print(column, "droped-unique")
#
#             else:
#                 zero_rows = df.loc[df[column] == float(0)]
#                 if zero_rows.shape[0] >= df.shape[0] * 50:
#                     df.drop(column, inplace=True, axis=1)
#                     droped_cols.append(column)
#                     print(column, "droped-zero")
#     return df
#
#
# # %%
#
# df = drop_nan_and_unique_data(df)
#
#
# # %%
#
# def fill_nan_values(data, name, fields):
#     field_df = data[fields]
#
#     if field_df.isnull().values.any():
#         data[fields] = field_df.interpolate(method='linear', limit_direction='both')
#
#     return data[fields]
#
#
# # %%
#
# for f in fields_dict:
#     name = fields_dict[f]['name']
#     fields = fields_dict[f]['fields']
#
#     print("\nname:", name, "fields:", fields)
#     fields = list(set(fields).difference(droped_cols))
#     df_ = df.groupby(["class", "run"]).apply(fill_nan_values, name, fields)
#     df_.reset_index(drop=True, inplace=True)
#     df[fields] = df_[fields]
#
# # %%
#
# print(df.isnull().sum().any())
#
# # %%
#
# df = drop_nan_and_unique_data(df)
#
#
# # %%
#
# df.to_csv("../data/df_cleaned.csv", index=False)
#
# # %%

data_df = pd.read_csv("../data/df_cleaned.csv")
df = data_df.copy()

# %%

# refinement_folds = {0: [100, 103, 10, 13, 26, 28, 42, 44, 49, 51, 58, 61, 64, 69, 73, 82, 89, 92, 94, 99],
#                     4: [1, 2, 4],
#                     11: [0, 1, 3],
#                     12: [0, 1, 3]
#                     }

# for key, value in refinement_folds.items():
#     df = df.loc[~((df["class"] == key) & df["run"].isin(value)

# %%

scaler_cols = list(set(df.columns).difference(["class", "run"]))

# %%

scaler = RobustScaler()
scaler_data = scaler.fit_transform(df[scaler_cols])
scaler_data = pd.DataFrame(scaler_data, index=df.index, columns=scaler_cols)

# %%

df = pd.concat([df[["class", "run"]], scaler_data], axis=1)
df['runId'] = 1000 * df['class'] + df['run']

# %%

run_df = df[['class', 'runId']].copy()
run_df.drop_duplicates(inplace=True)
run_df.reset_index(inplace=True)
# run_df = run_df.sample(frac=1, random_state=14).reset_index(drop=True)
del run_df['index']


# %%

run_df_ = df['run'].copy(deep=True)
del df['run']


# %%

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


# %%

# fold_num = 3
# cv = StratifiedKFold(n_splits=fold_num, shuffle=True, random_state=41)
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
# importance_df.to_pickle("../data/importance_df.pkl")
#
# plot_importance(importance_df, figsize=(12, 20))
# plt.savefig("importance_df_f1.png")
# plt.show()
# importance_df = importance_df.loc[(importance_df["importance_mean"] > 0.0001) & (importance_df["importance_std"] < 0.001)]

# %%

fold_num = 3
cv = StratifiedKFold(n_splits=fold_num, shuffle=True, random_state=41)

# %%

df_ori = df.copy(deep=True)

importance_df = pd.read_pickle("../data/importance_df.pkl")
importance_df = importance_df.loc[(importance_df["importance_mean"] > 0)]
sorted_features_imp = list(importance_df["feature"].values)
value_features_imp = list(importance_df["importance_mean"].values)
f_imp = [(name, value) for name, value in zip(sorted_features_imp, value_features_imp)]

# print("fold_num:", fold_num, len(f_imp), f_imp)
print(sorted_features_imp)

# %%

# num_leaves = [10, 20, 30, 40, 50]
# learning_rates = [0.1, 0.05, 0.01]
# n_estimators = [100, 350, 700, 1000]

# ws = 30
# fn = len(sorted_features_imp) # 135

# print("-------------------------------------------------------------")
# model_property = "ws_{}_fn_{}".format(ws, fn)
# print(model_property)
# print(pd.unique(run_df['class']))

df_report = pd.DataFrame()
sensor_list = sorted_features_imp[:len(sorted_features_imp)].copy()

df_result_ave = pd.DataFrame()

for ws in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]:

    acc_sum_2 = 0
    f1_sum_2 = 0
    mcc_sum_2 = 0

    print("_______________________________________________________________")

    for fold, (training_indices, validation_indices) in enumerate(cv.split(run_df['runId'], run_df['class'])):
        print("----------------------------------------------------------")

        report_index = "ws_{}_fold_{}".format(ws, fold + 1)
        print("Fold: ", report_index)

        training_runIds = run_df.loc[training_indices]['runId']
        validation_runIds = run_df.loc[validation_indices]['runId']

        df_ = df.loc[~(df["runId"].isin([56, 74, 49, 23, 35, 6, 83, 54]))]

        X_train_df = df_[df_['runId'].isin(training_runIds)].copy()
        X_val_df = df_[df_['runId'].isin(validation_runIds)].copy()

        X_train_df = X_train_df[sensor_list + ["class", "runId"]].copy()
        X_val_df = X_val_df[sensor_list + ["class", "runId"]].copy()

        X_train, y_train, runList_tr, l_len_runs_tr = create_datasets(X_train_df, ws)
        X_val, y_val, runList_val, l_len_runs_val = create_datasets(X_val_df, ws)

        # pca = PCA(n_components=0.95)
        # X_train = pca.fit_transform(X_train)
        # X_val = pca.transform(X_val)
        # print("X_train_df,  X_train:", X_train_df.shape, X_train.shape)

        lda = LinearDiscriminantAnalysis()
        X_train = lda.fit_transform(X_train, y_train)
        X_val = lda.transform(X_val)

        # model2 = OneVsOneClassifier(SVC(random_state=41))
        model2 = SVC(random_state=41)
        model2.fit(X_train, y_train)
        pred = model2.predict(X_val)

        acc_val2 = round(accuracy_score(y_val, pred), 3)
        f1_val2 = round(f1_score(y_val, pred, average='weighted'), 3)
        mcc_val2 = round(matthews_corrcoef(y_val, pred), 3)
        cm2 = confusion_matrix(y_val, pred)

        l_index = []
        for run_, num_run in zip(runList_val, l_len_runs_val):
            l_index.extend([run_] * num_run)

        # # 56, 74, 49; 7000
        # # 23, 35, 6, 83; 9001
        # # 54
        # print(len(l_index), len(y_val))
        df_result = pd.DataFrame()
        df_result["actual"] = y_val
        df_result["pred"] = pred
        df_result.loc[df_result.index[0], "cm"] = str(cm2)
        df_result.index = l_index

        df_result.to_excel("../data/results/excels/svm_{}.xlsx".format(report_index))
        with open('../data/results/models/svm_{}.pickle'.format(report_index), 'wb') as handle:
            pickle.dump(model2, handle, protocol=pickle.HIGHEST_PROTOCOL)

        acc_sum_2 += acc_val2
        f1_sum_2 += f1_val2
        mcc_sum_2 += mcc_val2

        print(cm2)
        print("SVM Fold:", fold, "ACC:", acc_val2, "F1:", f1_val2, "MCC:", mcc_val2)
        df_report.loc[report_index, "SVM_ACC"] = acc_val2
        df_report.loc[report_index, "SVM_F1"] = f1_val2
        df_report.loc[report_index, "SVM_MCC"] = mcc_val2


    ave_acc2 = round(acc_sum_2 / fold_num, 3)
    ave_f1_score2 = round(f1_sum_2 / fold_num, 3)
    ave_mcc2 = round(mcc_sum_2 / fold_num, 3)

    print("SVM Avg ACC:", ave_acc2, "Avg F1:", ave_f1_score2, "Avg MCC:", ave_mcc2)

    df_result_ave.loc[ws, "SVM_ACC"] = ave_acc2
    df_result_ave.loc[ws, "SVM_F1"] = ave_f1_score2
    df_result_ave.loc[ws, "SVM_MCC"] = ave_mcc2

df_result_ave.to_excel("../data/results/reports/ws_all_svm.xlsx".format(ws))
df_report.to_excel("../data/results/reports/report_all_svm.xlsx")


