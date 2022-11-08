import os
import shutil
import numpy as np
import pandas as pd


# Save subset into csv file
def create_subset_csv(df: pd.DataFrame, subset_name: str):
    subset_name = '../subsets/' + subset_name + '.csv'
    df.to_csv(subset_name, index=False)


def get_ids_to_remove_files_from_train(df_ref, df_subset):
    filenames = df_subset['Path'].to_list()
    ids_to_remove = []
    for fname in filenames:
        #id = np.where(df_ref['Path'].str.find(fname).to_numpy() != -1)[0]
        #id = int(id)
        id = np.asarray(df_ref[df_ref['Path'].str.contains(fname)].index)[0]
        ids_to_remove.append(id)
    return ids_to_remove


def construct_my_csv(df, drain, label):
    filenames = df['Path'].to_list()
    info_list = []
    for filename in filenames:
        id = np.where(df['Path'].str.find(filename).to_numpy() != -1)[0]
        id = int(id)
        # Path, Age, Sex, Pneumothorax, Drain
        info_list.append([df['Path'].iloc[id],
                          df['Age'].iloc[id],
                          df['Sex'].iloc[id],
                          label, drain])

    # Create the pandas DataFrame
    columns = ['Path', 'Age', 'Sex', 'Pneumothorax', 'Drain']
    df = pd.DataFrame(info_list, columns=columns)
    return df

print('creating subsets')
pd.set_option('display.max_colwidth', 255)  # Increases column width to examine un-truncated strings

train_csv = '../config/train.csv'
df_train = pd.read_csv(train_csv)
#train_filenames = df_train['Path'].to_list()

# only frontal
df_train = df_train[df_train['Path'].str.contains('frontal')]

# ~300 drain filenames
drain_csv = 'drains_376.csv'
df_drains = pd.read_csv(drain_csv)
ids_to_remove = get_ids_to_remove_files_from_train(df_train, df_drains)
df_train = df_train.drop(index=ids_to_remove)

# 300 negative filenames
n_negative_test = 300
df_thorax_negative = df_train.loc[df_train['Pneumothorax'] == 0.0]
df_thorax_negative_subset = df_thorax_negative.sample(n=n_negative_test, random_state=0)
ids_to_remove = get_ids_to_remove_files_from_train(df_train, df_thorax_negative_subset)
df_train = df_train.drop(index=ids_to_remove)

# write thorax negative csv file
df_thorax_negative_subset = construct_my_csv(df_thorax_negative_subset, label=0, drain=0)
df_thorax_negative_subset.to_csv('thorax_negative_300.csv', index=False)

# 150 positive without drains
n_positive_test = 300
df_thorax_positive = df_train.loc[df_train['Pneumothorax'] == 1.0]
df_thorax_positive_subset = df_thorax_positive.sample(n=n_positive_test, random_state=0)
ids_to_remove = get_ids_to_remove_files_from_train(df_train, df_thorax_positive_subset)
df_train = df_train.drop(index=ids_to_remove)

# write thorax positive to csv file
df_thorax_positive_subset = construct_my_csv(df_thorax_positive_subset, label=1, drain=0)
df_thorax_positive_subset.to_csv('thorax_positive_300.csv', index=False)

# Construct train / validation data
# Pneumothorax positive filter
data_thorax_positive = df_train.loc[df_train['Pneumothorax'] == 1.0]
data_thorax_negative = df_train.loc[df_train['Pneumothorax'] == 0.0]

n_thorax_positive = 15000
n_thorax_negative = 15000
n_subset = n_thorax_positive + n_thorax_negative

data_seed = 2
df_both = pd.concat([data_thorax_positive.sample(n=n_thorax_positive, random_state=data_seed),
                     data_thorax_negative.sample(n=n_thorax_negative, random_state=data_seed)])
create_subset_csv(df_both, str(int(n_subset/1000))+'k-'+str(data_seed))