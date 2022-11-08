import os
import sys
import random
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

# Get the datasets embedded in a 1024-dimensional space by the trained network
emb_chexpert = np.load('embedding_chexpert.npy')
emb_nihcxr14 = np.load('embedding_nihcx14.npy')
emb_combined = np.concatenate((emb_chexpert, emb_nihcxr14), axis=0)

# TSNE randomizes so let's set seed
# Note this does not make the plot look exactly the same, but a rotated version
random.seed(1)

# Embed in 2D space
emb = TSNE(n_components=2, init='random', perplexity=30.).fit_transform(emb_combined)

# Get meta-data
df_chexpert = pd.read_csv('my_test_with_preds.csv')
df_nihcxr14 = pd.read_csv('nih_test_with_preds.csv')

# Create to-plot dataset
n_chexpert = emb_chexpert.shape[0]
n_nihcxr14 = emb_nihcxr14.shape[0]

dataset_list = ['CheXpert'] * n_chexpert + ['NIH-CXR14'] * n_nihcxr14
drain_list = df_chexpert['Drain'].tolist() + df_nihcxr14['Drain'].tolist()
pneumo_list = df_chexpert['Pneumothorax'].tolist() + df_nihcxr14['Pneumothorax'].tolist()

data_dict = {'Dataset': dataset_list, 'Drain': drain_list, 'Pneumothorax': pneumo_list,
        'comp-1': pd.Series(emb[:, 0].tolist()), 'comp-2': pd.Series(emb[:, 1].tolist())}
df = pd.DataFrame(data_dict)

# Plot
fig = plt.figure()
ax = plt.subplot(111)


# create a palette dict with a known color_palette
attributes = [(1, 1), (1, 0), (0, 0)]
colors = ['#1f78b4', '#a6cee3', '#b2df8a']
palette = dict(zip(attributes, sns.color_palette(colors)))

plt.show()

g = sns.jointplot(data=df, x="comp-1", y="comp-2", ec='dimgray', palette=palette,
                  hue=df[['Pneumothorax', 'Drain']].apply(tuple, axis=1),
                  style=df['Dataset'], markers=['^', 'o'], s=30)
plt.legend(ncol=1, loc='lower right')
plt.show()