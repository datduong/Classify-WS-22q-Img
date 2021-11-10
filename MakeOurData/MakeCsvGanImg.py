
import os,sys,re,pickle
import pandas as pd 
import numpy as np 

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--fout_path", type=str, default=None)
parser.add_argument("--img_data_path", type=str, default=None)
parser.add_argument("--datatype", type=str, default='train')
parser.add_argument("--suffix", type=str, default='') # '/data/duongdb/FairFace/FairFace-aligned-sub-agegroup'
parser.add_argument("--fold", type=int, default=None) # ! fold to skip out
parser.add_argument("--original_train_csv", type=str, default=None)
parser.add_argument("--wholegroup", type=int, default=0)
parser.add_argument("--keep_label_original", type=str, default=None)
parser.add_argument("--soft_label", type=int, default=0)

args = parser.parse_args()

args.wholegroup = True if args.wholegroup == 1 else False

# ----------------------------------------------------------------------------------------------------------------

if not os.path.exists (args.fout_path): os.mkdir(args.fout_path)

os.chdir(args.fout_path)

np.random.seed(seed=1)

# ! make csv input for the images. 
# col names: name,path,label,person_id,row_num,fold
# go over this label file, add in img name and path. 
images_dict = {'name':[], 
               'path':[], 
               'label': [], 
               'fold': [], 
               'is_ext': [] # ! so we can exclude normal faces
              }

if args.soft_label: 
  images_dict['softlabel'] = [] # ! extra col


if args.wholegroup: 
  labelset = sorted('WS,22q11DS,Controls'.split(','))
else: 
  labelset = sorted('WS_early,WS_late,WS_inter,22q11DS_early,22q11DS_late,22q11DS_inter,Controls_early,Controls_inter,Controls_late'.split(','))

# ! we make GAN for each label for each fold, so we have to iter through them ?
images = [j for j in os.listdir(args.img_data_path) if ('png' in j) or ('jpg' in j)] # all img names

images = np.random.permutation(images)
images = {v:k for k,v in enumerate(images)} # faster look-up
print ('total num img {}'.format(len(images)))

# ! put our conditions + other conditions + normal ? 
if args.wholegroup:
  labelset = sorted ( labelset + ['Normal'] )
else: 
  labelset = sorted ( labelset + ['Normal_early','Normal_inter','Normal_late'] )

#
labelset = {v.strip():i for i,v in enumerate(labelset)}
numlabel = len(labelset)

FOLDS = [1,2,3,4,5] # ! by default 0 will be true testset 
FOLDS.remove(args.fold)
print (FOLDS)

for index,filename in enumerate(images): # file,age,gender,race,service_test
  #
  for l in labelset: 
    if l in filename: 
      condition = l # string
  #
  if args.soft_label: 
    temp_ = [0]*numlabel
    temp_[ labelset[condition] ] = 1 # ! 1 hot array, write this out
    soft = ';'.join(str(i) for i in temp_)
  #
  fold = FOLDS[index % 4] # ! divide into @FOLDS
  # okay to use fold!=0 because @images ordering is known beforehand. 
  images_dict['name'].append(filename)
  if 'Normal' in condition: 
    images_dict['path'].append(os.path.join(args.normal_data_path,filename)) 
    images_dict['is_ext'].append(1)
  else:  
    images_dict['path'].append(os.path.join(args.img_data_path,filename))
    images_dict['is_ext'].append(0)
  # 
  images_dict['label'].append(condition) # ! take a string as a label
  images_dict['fold'].append(fold)
  if args.soft_label: 
    images_dict['softlabel'].append(soft)

# 
df = pd.DataFrame.from_dict(images_dict)
df.shape

print ( df.groupby('label').count() ) 

if args.original_train_csv is not None: # ! append to original training dataset 
  df_original = pd.read_csv(args.original_train_csv)
  print ('original df size (at read in) {}'.format(df_original.shape))
  if args.keep_label_original is not None: 
    args.keep_label_original = args.keep_label_original.split(',') # keep these labels from original (so also keep the images)
    df_original = df_original[df_original.label.isin(args.keep_label_original)]
  print ('original df size (may filter out some stuffs) {}'.format(df_original.shape))
  print ( df_original.groupby('label').count() ) 
  df = pd.concat([df,df_original])


print ('final size {}'.format(df.shape))
df.to_csv(os.path.join(args.fout_path,args.datatype+'-oursplit'+args.suffix+'.csv'), index=False)

print ( df.groupby('label').count() ) 

# # ! look at labels count 
# if args.datatype == 'train': 
#   for i in labelset:
#     print (i)
#     print ( images_dict['label'].count(str(i))/df.shape[0] ) 



# ! check size 

