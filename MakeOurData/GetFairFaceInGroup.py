

import os,sys,re,pickle
import pandas as pd
import numpy as np
from pandas.core.dtypes.missing import notna 

# ! partition FF into the same as our data 

os.chdir('/data/duongdb/FairFace')

# 0-2
# 2-10
# 10-19
# 20-34
# 35-over

srcdir = 'train_align_as_stylegan'
newdir = 'FairFace-aligned-60k-agegroup-06012021'
if not os.path.exists(newdir): 
  os.mkdir(newdir)

# 

df = pd.read_csv('Labels-FairFace-AlignedStyleGan.csv')

# face_name_align,race,race4,gender,age,race_scores_fair,race_scores_fair_4,gender_scores_fair,age_scores_fair,expected_age,in_aligned_stylegan
for index,row in df.iterrows() : 
  if row['expected_age'] < 2: 
    label = '2y'
  elif (row['expected_age'] >= 2) and (row['expected_age'] < 10):
    label = 'youngchild'
  elif (row['expected_age'] >= 10) and (row['expected_age'] < 20):
    label = 'adolescence'
  elif (row['expected_age'] >= 20) and (row['expected_age'] < 35):
    label = 'youngadult'
  else: 
    label = 'olderadult'
  # 
  label = 'Normal'+label
  # move files over ? 
  # ! follow the same name structure as 22q11DS_early108_22q11DSyoungchild.png
  newname = re.sub(r'\.png','',row['face_name_align']) + '_' + label + '.png'
  os.system('scp '+os.path.join(srcdir,row['face_name_align']) + ' ' + os.path.join(newdir,newname) )
  # break 


