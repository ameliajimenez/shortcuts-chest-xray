import pandas as pd
import numpy as np
import os


pd.options.mode.chained_assignment = None  # default='warn'


# load drains file
drain_csv = 'drains_376.csv'
df_drain = pd.read_csv(drain_csv)  # 376 samples
# sample 300 drains with pneumothorax positive
df_drain = df_drain[df_drain['Pneumothorax'] == 1.0].sample(300, random_state=0)
df_drain['with_drain'] = 1.0
df_drain['without_drain'] = 0.0
df_drain['baseline'] = 0.0
df_drain_baseline = df_drain.sample(150, random_state=0)
df_drain['baseline'].loc[df_drain_baseline.index] = 1.0

# load thorax neg
thorax_neg_csv = 'thorax_negative_300.csv'
df_thorax_neg = pd.read_csv(thorax_neg_csv)  # 300 samples
df_thorax_neg['with_drain'] = 1.0
df_thorax_neg['without_drain'] = 1.0
df_thorax_neg['baseline'] = 1.0

# load thorax pos
thorax_pos_csv = 'thorax_positive_300.csv'
df_thorax_pos = pd.read_csv(thorax_pos_csv)  # 300 samples
df_thorax_pos['with_drain'] = 0.0
df_thorax_pos['without_drain'] = 1.0
df_thorax_pos['baseline'] = 0.0
df_thorax_pos_baseline = df_thorax_pos.sample(150, random_state=0)
df_thorax_pos['baseline'].loc[df_thorax_pos_baseline.index] = 1.0

pd_list = [df_drain, df_thorax_pos, df_thorax_neg]
df = pd.concat(pd_list)
df.to_csv('../config/my_test.csv', index=False)