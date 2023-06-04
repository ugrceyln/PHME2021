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

fields_path = '../data/training_validation_2/fields.csv'
fields_df = pd.read_csv(fields_path)
fields_df.columns = ['name', 'f0', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6']

fields_dict = {}

for idx in range(fields_df.shape[0]):
    name = fields_df.loc[idx, 'name']

    _fields = []

    for f in fields_df.columns[1:]:
        if not (str(fields_df.loc[idx, f]) == 'nan'):
            _fields.append(name + "_" + str(fields_df.loc[idx, f]))

    fields_dict[idx] = {'name': fields_df.loc[idx, 'name'], 'fields': _fields}


# %%

# Get class id and run id from filename
def parse_class_name(fname):
    p = re.compile("^class[^\d]*(\d+)_(\d+).*.csv")
    m = p.match(fname)

    return m.groups()


# %%

features_with_x_na = ['LightBarrieActiveTaskDuration2_vCnt', 'LightBarrieActiveTaskDuration2_vFreq',
                      'LightBarrierActiveTaskDuration1b_vCnt', 'LightBarrierActiveTaskDuration1b_vFreq',
                      'LightBarrierPassiveTaskDuration1b_vCnt', 'LightBarrierPassiveTaskDuration1b_vFreq',
                      'LightBarrierPassiveTaskDuration2_vCnt', 'LightBarrierPassiveTaskDuration2_vFreq',
                      'LightBarrierTaskDuration_vCnt', 'LightBarrierTaskDuration_vFreq']


# %%

def impute_df(df):
    for f in features_with_x_na:
        new_f_name = f + "_na"
        df[new_f_name] = df[f].isna().astype(np.int32)

        del df[f]

    df = df.interpolate(limit_direction='both')

    return df


# %%

# Load one data file and return in a data frame
def load_data_file(path, fname):
    fullpath = join(path, fname)
    df = pd.read_csv(fullpath)
    df.columns = ['name', 'data']

    dfx = []

    for f in fields_dict:
        name = fields_dict[f]['name']
        fields = fields_dict[f]['fields']

        data = eval(df.loc[f, 'data'])  # convert data to array

        new_df = pd.DataFrame(data)
        if (f == 33) and (new_df.shape[1] == 6):  # NumberFuseDetected has a special case!
            new_df[6] = new_df[5]
            new_df[5] = np.NaN

        new_df.columns = fields_dict[f]['fields']

        dfx.append(new_df)

    merged_df = pd.concat(dfx, axis=1)  # Merge columns

    # Do some imputation on the data file
    merged_df = impute_df(merged_df.copy())

    c, r = parse_class_name(fname)  # Get class id and run id

    # Add class labels and run id
    merged_df['class'] = int(c)
    merged_df['run'] = int(r)

    return merged_df


# %%

# Load data files from a directory and return merged data frame
def load_data_files(path):
    print("In", path)
    files = []
    for f in listdir(path):
        if (isfile(join(path, f)) and (f.startswith("class"))):
            files.append(f)

    data_df_list = []
    for fname in files:
        print("Loading:", fname)

        df = load_data_file(path, fname)

        data_df_list.append(df)

    data_df = pd.concat(data_df_list, axis=0)  # Merge data frames

    return data_df

# %%

data_df_1 = load_data_files("../data/training_validation_1/")
data_df_2 = load_data_files("../data/training_validation_2/")
data_df_3 = load_data_files("../data/ModelRefinement/")

# %%

data = pd.concat([data_df_1, data_df_2, data_df_3], axis=0, ignore_index=True)

# %%

features_with_x_na_ = [x + '_na' for x in features_with_x_na]

# %%

data_na = data[features_with_x_na_]
data_na.columns = data_na.columns.str.replace("_na", "")


# %%

data_df = pd.read_csv("../data/df_cleaned.csv")
df = data_df.copy()

# %%
#
df = pd.concat([data_na, df], axis=1)
#
# df = data_df.copy()


# %%

base_features = fields_df['name'].tolist()
print(base_features)

# %%

new_field_dict = {}
new_field_dict["Base"] = []

for index, name_fields in fields_dict.items():
    name = name_fields["name"]
    fields = name_fields["fields"]

    if any(name in s for s in df.columns):
        with_s = [x for x in df.columns if x.startswith(name + "_")]
        new_field_dict[name] = with_s
    else:
        print(name, fields)


# %%

df['runId'] = 1000 * df['class'] + df['run']

# %%

run_df = df[['class', 'runId']].copy()
run_df.drop_duplicates(inplace=True)
run_df.reset_index(inplace=True)
# run_df = run_df.sample(frac=1, random_state=14).reset_index(drop=True)
del run_df['index']


# %%

run_df_ = df['run'].copy()
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

fold_num = 3
cv = StratifiedKFold(n_splits=fold_num, shuffle=True, random_state=41)

# %%

l_classes = [0, 7]

scores_f1 = {}
scores_mcc = {}
ws = 20
print(new_field_dict)
print("----------------------------------------------------------")
for name, fields in new_field_dict.items():
    # print("----------------------------------------------------------")

    sensor_list = list(set(df.columns.tolist()).difference(fields))
    sensor_list = list(set(sensor_list).difference(["class", "runId"]))
    # print(fields, len(sensor_list))

    ave_f1 = 0
    ave_mcc = 0

    for fold, (training_indices, validation_indices) in enumerate(cv.split(run_df['runId'], run_df['class'])):
        # print(report_index)

        training_runIds = run_df.loc[training_indices]['runId']
        validation_runIds = run_df.loc[validation_indices]['runId']

        X_train_df = df[df['runId'].isin(training_runIds)].copy()
        X_val_df = df[df['runId'].isin(validation_runIds)].copy()

        X_train_df = X_train_df[sensor_list + ["class", "runId"]].copy()
        X_val_df = X_val_df[sensor_list + ["class", "runId"]].copy()

        X_train_df = X_train_df[X_train_df['class'].isin(l_classes)]
        X_val_df = X_val_df[X_val_df['class'].isin(l_classes)]

        scaler_cols = sensor_list.copy() # list(set(sensor_list).difference(["class", "runId"]))

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
        # print(X_train_df.shape, X_train.shape)

        model1 = LGBMClassifier(random_state=41)
        model1.fit(X_train, y_train)
        pred = model1.predict(X_val)

        f1_val1 = f1_score(y_val, pred, average='weighted')
        mcc_val1 = matthews_corrcoef(y_val, pred)
        cm1 = confusion_matrix(y_val, pred)

        # print(cm1)

        ave_f1 += f1_val1
        ave_mcc += mcc_val1

        # print("LightGBM Fold:", fold, "F1:", f1_val1, "MCC:", mcc_val1)

    ave_f1_ = ave_f1 / fold_num
    ave_mcc_ = ave_mcc / fold_num

    scores_f1[name] = ave_f1_
    scores_mcc[name] = ave_mcc_
    print(name, "Avg F1:", ave_f1_, "Avg MCC:", ave_mcc_)

with open('score_f1_{}.pickle'.format(str(l_classes)), 'wb') as handle:
    pickle.dump(scores_f1, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('score_mcc_{}.pickle'.format(str(l_classes)), 'wb') as handle:
    pickle.dump(scores_mcc, handle, protocol=pickle.HIGHEST_PROTOCOL)

print(scores_f1)
print(scores_mcc)

# %%


