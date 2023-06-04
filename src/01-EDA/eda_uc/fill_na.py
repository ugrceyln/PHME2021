import numpy as np
import pandas as pd
from numpy import nan
import matplotlib.pyplot as plt
import seaborn as sns


data_df_1 = pd.read_csv("D:/Datasets/PHME21/training_validation_1.csv")
data_df_2 = pd.read_csv("D:/Datasets/PHME21/training_validation_2.csv")
data_df_3 = pd.read_csv("D:/Datasets/PHME21/training_validation_3.csv")

data_df = pd.concat([data_df_1, data_df_2, data_df_3], axis=0, ignore_index=True)

fields_path = 'D:/Datasets/PHME21/ModelRefinement/fields.csv'
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


def get_nan_info_table(data, fields):

    # class_ = data["class"].iloc[0]
    # run_ = data["run"].iloc[0]
    field_df = data[fields]
    # print(field_df.isnull().sum())

    return field_df.isnull().sum().T


df = data_df.copy()
feature_sets = data_df.columns

nan_info1_table = df.groupby(["class", "run"]).apply(get_nan_info_table, feature_sets)

plt.figure (figsize=(20, 20))
sns.heatmap(nan_info1_table.T, annot=False)
plt.show()

droped_cols = []
first_cols = df.columns.tolist()
df = df.dropna(thresh=int(df.shape[0] * 0.7), axis=1)  # Drop column if it does not have at least x values that are **not** NaN
print("col: ", df.shape)
droped_cols.extend(list(set(first_cols).difference(df.columns)))

for column in df.columns:
    if column not in ["class", "run"]:
        if (len(df[column].unique()) == 1) or (df[column].isnull().all()):
            df.drop(column, inplace=True, axis=1)
            droped_cols.append(column)
            print(column, "droped-unique")

        else:
            zero_rows = df.loc[df[column] == float(0)]
            if zero_rows.shape[0] >= df.shape[0] * 50:
                df.drop(column, inplace=True, axis=1)
                droped_cols.append(column)
                print(column, "droped-zero")

print(pd.unique(df["class"]))

def fill_nan_values(data, name, fields):

    field_df = data[fields]

    if field_df.isnull().values.any():
        data[fields] = field_df.interpolate(method='linear', limit_direction='both')

    return data[fields]


for f in fields_dict:

    name = fields_dict[f]['name']
    fields = fields_dict[f]['fields']

    print("\nname:", name, "fields:", fields)
    fields = list(set(fields).difference(droped_cols))
    df_ = df.groupby(["class", "run"]).apply(fill_nan_values, name, fields)
    df_.reset_index(drop=True, inplace=True)
    df[fields] = df_[fields]

print(df.isnull().sum().any())

for column in df.columns:
    if column not in ["class", "run"]:
        if (len(df[column].unique()) == 1) or (df[column].isnull().all()):
            df.drop(column, inplace=True, axis=1)
            print(column, "droped-unique")

        else:
            zero_rows = df.loc[df[column] == float(0)]
            if zero_rows.shape[0] >= df.shape[0] * 50:
                df.drop(column, inplace=True, axis=1)
                print(column, "droped-zero")

df.to_csv("D:/Datasets/PHME21/df_cleaned.csv", index=False)
print("last df shape:", df.shape)
