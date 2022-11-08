import pandas as pd
import numpy as np
import os
import seaborn as sns
from sklearn import metrics
import matplotlib.pyplot as plt
import matplotlib as mpl


def compute_auc(df, model_name, data_seed, prob_thred=0.5):
    y_pred = df['y_pred_' + model_name + '_' + data_seed]
    y_true = df['Pneumothorax']
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    #print('auc', auc)
    ppv = metrics.precision_score(y_true, y_pred, pos_label=1)
    acc = metrics.accuracy_score(
        y_true, (y_pred >= prob_thred).astype(int), normalize=True
    )
    #print('acc', acc)
    return auc, fpr, tpr, ppv

cfg_path = '../config/'
df = pd.read_csv(os.path.join(cfg_path, 'my_test_with_preds.csv'))

exp_settings = ['baseline', 'with_drain', 'without_drain']
model_names = ['5k', '10k', '20k']  # 30k
training_data = ['4k', '8k', '16k']  # 24k
data_seeds = ['1', '2', '3']
dataset = 'CheXpert'

# all patients
auc_list = []
ppv_list = []
tpr_list = []
fpr_list = []
model_list = []
training_data_list = []
data_seed_list = []
exp_setting_list = []
dataset_list = []
sex_list = []
for setting in exp_settings:
    df_setting = df[df[setting] == 1.0]
    for k, model_name in enumerate(model_names):
        for data_seed in data_seeds:
            auc, fpr, tpr, ppv = compute_auc(df_setting, model_name, data_seed)
            auc_list.append(auc)
            ppv_list.append(ppv)
            tpr_list.append(tpr)
            fpr_list.append(fpr)
            exp_setting_list.append(setting)
            model_list.append(model_name)
            training_data_list.append(training_data[k])
            data_seed_list.append(data_seed)
            dataset_list.append(dataset)
            sex_list.append('All')

df_metrics = pd.DataFrame({'AUC': auc_list,
                           'PPV': ppv_list,
                           'TPR': tpr_list,
                           'FPR': fpr_list,
                           'Model name': model_list,
                           'Training data': training_data_list,
                           'Data seed': data_seed_list,
                           'Scenario': exp_setting_list,
                           'Dataset': dataset_list,
                           'Sex': sex_list,
                           })

g = sns.pointplot(data=df_metrics, x='Training data', y='AUC', hue='Scenario')
plt.show()

# sex & scenario plot
exp_settings = ['baseline', 'with_drain', 'without_drain']
sex_settings = ['M', 'F']  # [Male, Female] for CheXpert; ['M', 'F'] for NIH-CXR14
auc_list = []
ppv_list = []
tpr_list = []
fpr_list = []
model_list = []
training_data_list = []
data_seed_list = []
exp_setting_list = []
sex_list = []
dataset_list = []
for sex in sex_settings:
    for setting in exp_settings:
        df_setting = df[(df[setting] == 1.0) & (df['Sex'] == sex)]
        for k, model_name in enumerate(model_names):
            for data_seed in data_seeds:
                auc, fpr, tpr, ppv = compute_auc(df_setting, model_name, data_seed)
                auc_list.append(auc)
                ppv_list.append(ppv)
                tpr_list.append(tpr)
                fpr_list.append(fpr)
                exp_setting_list.append(setting)
                if sex == 'M':
                    sex_save = 'Male'
                elif sex == 'F':
                    sex_save = 'Female'
                sex_list.append(sex_save)
                model_list.append(model_name)
                training_data_list.append(training_data[k])
                data_seed_list.append(data_seed)
                dataset_list.append('NIH-CXR14')

df_metrics_sex = pd.DataFrame({'AUC': auc_list,
                               'PPV': ppv_list,
                               'TPR': tpr_list,
                               'FPR': fpr_list,
                               'Model name': model_list,
                               'Training data': training_data_list,
                               'Data seed': data_seed_list,
                               'Scenario': exp_setting_list,
                               'Dataset': dataset_list,
                               'Sex': sex_list})

df_to_save = pd.concat([df_metrics_sex, df_metrics])
df_to_save.to_csv(os.path.join(cfg_path, 'chexpert-results.csv'))

# boxplot for sex & scenario & dataset
df_vis = pd.read_csv(os.path.join(cfg_path, 'chexpert-results.csv'))

fig = plt.figure()
hatches = ['//']*12
hue_order = ['All', 'Male', 'Female']
colors = ['#66c2a5', '#8da0cb', '#fc8d62']  # 'yellow', 'red', 'blue'

g = sns.boxplot(data=df_vis[df_vis['Scenario'] == 'without_drain'], x='Training data', y='AUC',
                hue=df_vis[df_vis['Scenario'] == 'without_drain']['Sex'], hue_order=hue_order, palette=colors)
g = sns.boxplot(data=df_vis[df_vis['Scenario'] == 'with_drain'], x='Training data', y='AUC',
                hue=df_vis[df_vis['Scenario'] == 'with_drain']['Sex'], hue_order=hue_order, palette=colors)

patches = [patch for patch in g.patches if type(patch) == (mpl.patches.PathPatch)]
h = hatches * (len(patches) // len(patches))
# iterate through the patches for each subplot
for patch, hatch in zip(patches[0:12], h):
    patch.set_hatch(hatch)
    fc = patch.get_facecolor()
    patch.set_edgecolor(fc)
    patch.set_facecolor('none')

#plt.legend([],[], frameon=False)
l = g.legend(ncol=2)
for lp, hatch in zip(l.get_patches()[3:6], hatches):
    lp.set_hatch(hatch)
    fc = lp.get_facecolor()
    lp.set_edgecolor(fc)
    lp.set_facecolor('none')
plt.ylim([0.5, 0.9])
plt.show()
#plt.savefig("boxplot-nihcxr14.pdf")
