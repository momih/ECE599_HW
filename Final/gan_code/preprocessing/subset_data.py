import pandas as pd
import os
import tarfile
from tqdm import tqdm
from shutil import copy

def assign_label(x, terms):
    x = x.split("|")
    if 'No Finding' in x:
        lab = 'healthy'        
    elif any(pathology in terms for pathology in x):
        lab = 'shadow'
    else:
        lab = 'other'
    return lab

tqdm.pandas(desc="my bar!")  
df_orig = pd.read_csv('../data/attributes.csv', usecols=range(7))
bbox = pd.read_csv('../data/bbox.csv')
bbox_ids = sorted(list(set(pd.merge(df_orig, bbox, how='inner', on=['filename'])['id'])))
# =============================================================================
# Getting bounding box files
# =============================================================================
lung_shadow = ['Atelectasis', 'Infiltrate', 'Pneumonia', 'Mass', 'Nodule']
bbox = bbox[bbox['bbox'].isin(lung_shadow)]
bbox = pd.merge(df_orig, bbox, how='inner', on=['filename'])
bbox['label'] = 'shadow'
#bbox.progress_apply(lambda x: copy('../data/uncompressed/images/' + x['filename'], '../data/xrays/shadow' + '/'), 1)

# removing bbox rows from df_orig 
df_orig = df_orig[~df_orig['filename'].isin(bbox['filename'].tolist())]

# making a new column for label
df_orig['label'] =  df_orig['pathology'].apply(assign_label,terms = lung_shadow)

#new dfs for shadow and healthy
shadow_df = df_orig[df_orig['label'] == 'shadow'].sort_values('id').reset_index(drop=True)
healthy_df = df_orig[df_orig['label'] == 'healthy'].sort_values('id').reset_index(drop=True)

# random samples from both df
shadow_df = shadow_df.sample(n=1000, random_state=10)
healthy_df = healthy_df.sample(n=1000, random_state=10)

shadow_df.progress_apply(lambda x: copy('../data/uncompressed/images/' + x['filename'], '../data/xrays/test_shadow/'), 1)
healthy_df.progress_apply(lambda x: copy('../data/uncompressed/images/' + x['filename'], '../data/xrays/test_healthy/'), 1)

shadow_df.to_csv('../data/xrays/shadow.csv')
healthy_df.to_csv('../data/xrays/healthy.csv')