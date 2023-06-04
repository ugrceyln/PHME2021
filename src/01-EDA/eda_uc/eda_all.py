

import numpy as np
from numpy import nan
import pandas as pd
from os import listdir
from os.path import isfile, join

import re

# Convert field names to dict for easy access.
# Can be hard coded

fields_path = 'D:/Datasets/PHME21/training_validation_2/fields.csv'
fields_df = pd.read_csv(fields_path)
fields_df.columns = ['name', 'f0', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6']

# fields_path = 'D:/Datasets/PHME21/ModelRefinement/fields.csv'
# fields_df2 = pd.read_csv(fields_path)
# fields_df2.columns = ['name', 'f0', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6']
#
# if fields_df.equals(fields_df2):
#     print("OK")

fields_dict = {}

for idx in range(fields_df.shape[0]):
    name = fields_df.loc[idx, 'name']

    _fields = []

    for f in fields_df.columns[1:]:
        if not (str(fields_df.loc[idx, f]) == 'nan'):
            _fields.append(name + "_" + str(fields_df.loc[idx, f]))

    fields_dict[idx] = {'name': fields_df.loc[idx, 'name'], 'fields': _fields}

print(fields_dict)


# Get class id and run id from filename
def parse_class_name(fname):
    p = re.compile("^class.*(\d+)_(\d+).*.csv")
    m = p.match(fname)

    return m.groups()


# Load one data file and return in a data frame
def load_data_file(path, fname):
    fullpath = join(path, fname)
    df = pd.read_csv(fullpath)
    df.columns = ['name', 'data']

    dfx = []

    for f in fields_dict:
        # name = fields_dict[f]['name']
        # fields = fields_dict[f]['fields']

        data = eval(df.loc[f, 'data'])  # convert data to array

        new_df = pd.DataFrame(data)
        if (f == 33) and (new_df.shape[1] == 6):  # NumberFuseDetected has a special case!
            new_df[6] = new_df[5]
            new_df[5] = np.NaN

        new_df.columns = fields_dict[f]['fields']

        dfx.append(new_df)

    merged_df = pd.concat(dfx, axis=1)  # Merge columns

    c, r = parse_class_name(fname)  # Get class id and run id

    # Add class labels and run id
    merged_df['class'] = int(c)
    merged_df['run'] = int(r)

    return merged_df


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


data_df_1 = load_data_files("D:/Datasets/PHME21/training_validation_1/")
data_df_2 = load_data_files("D:/Datasets/PHME21/training_validation_2/")
data_df_3 = load_data_files("D:/Datasets/PHME21/ModelRefinement/")
data_df_3["class"] = data_df_3["class"].replace(1, 11)
data_df_3["class"] = data_df_3["class"].replace(2, 12)

data_df_1.to_csv("D:/Datasets/PHME21/training_validation_1.csv", index=False)
data_df_2.to_csv("D:/Datasets/PHME21/training_validation_2.csv", index=False)
data_df_3.to_csv("D:/Datasets/PHME21/training_validation_3.csv", index=False)

print(data_df_1.shape, data_df_2.shape, data_df_3.shape)






