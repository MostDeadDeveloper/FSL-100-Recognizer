
#%%

import pandas as pd
df = pd.read_csv('FSL-100-Recognizer/train.csv')


# assume df is your DataFrame
allowed = ["COLOR", "CALENDAR", "GREETING"]
filtered_df = df[df['category'].isin(allowed)].reset_index(drop=True)

filtered_df=  filtered_df.drop(columns='label')
# optional: verify
# print(filtered_df['category'].value_counts())

filtered_df= filtered_df.rename(columns={'category': 'label_str'})

filtered_df['label_str'] = filtered_df['label_str'].astype('category')

filtered_df['label'] = filtered_df['label_str'].cat.codes

filtered_df.to_csv('spliced_train.csv')