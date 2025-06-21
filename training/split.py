
#%%

import pandas as pd
df = pd.read_csv('train.csv')


# assume df is your DataFrame
# allowed = ["COLOR", "CALENDAR", "GREETING"]
allowed = ["GOOD EVENING", "COLD"]
# allowed = ['GOOD EVENING']

# filtered_df = df[df['category'].isin(allowed)].reset_index(drop=True)
# filtered_df = df[df['label'].isin(allowed)].reset_index(drop=True)
filtered_df = df
# filtered_df=  filtered_df.drop(columns='label')
filtered_df=  filtered_df.drop(columns='category')
# optional: verify
# print(filtered_df['category'].value_counts())

filtered_df= filtered_df.rename(columns={'label': 'label_str'})

filtered_df['label_str'] = filtered_df['label_str'].astype('category')

filtered_df['label'] = filtered_df['label_str'].cat.codes

filtered_df = filtered_df.rename(columns={'vid_path':'filename'})

filtered_df = filtered_df.drop(columns=['id_label','label_str'])

# filtered_df.drop(filtered_df.columns[0], axis=1, inplace=True)

filtered_df.to_csv('spliced_train.csv', index=False)