
import os,sys,re,pickle
import pandas as pd 
import numpy as np 
from copy import deepcopy

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--root_data_path", type=str, default=None)
parser.add_argument("--datatype", type=str, default='train')
parser.add_argument("--suffix", type=str, default='') # '/data/duongdb/FairFace/FairFace-aligned-sub-agegroup'
parser.add_argument("--normal_data_path", type=str, default=None)
parser.add_argument("--wholegroup", type=int, default=0)
parser.add_argument("--soft_label", type=int, default=0)
parser.add_argument("--source_folder", type=str, default='Align1024')
parser.add_argument("--qualtric_test_img_csv", type=str, default=None)
parser.add_argument("--only_these_labels", type=str, default=None)
parser.add_argument("--normal_test_img_csv", type=str, default=None)



args = parser.parse_args()
args.wholegroup = True if args.wholegroup == 1 else False
args.soft_label = True if args.soft_label == 1 else False

# ----------------------------------------------------------------------------------------------------------------

qualtric_test_img = pd.read_csv(args.qualtric_test_img_csv,header=None) # ! write these test images into a file to backtrack easier. 
qualtric_test_img = qualtric_test_img[0].values.tolist() # convert to a numpy list 

fout_path = os.path.join(args.root_data_path,'Classify') 
if not os.path.exists (fout_path): os.mkdir(fout_path)

os.chdir(args.root_data_path)

img_data_path = os.path.join(args.root_data_path,args.source_folder)

np.random.seed(seed=1)

# ! WE READ IN "ALIGN-AS-IN-STYLEGAN2" IMAGES LOCATED IN THE SAME FOLDER
images = [j for j in os.listdir(img_data_path) if ('png' in j) or ('jpg' in j)] # all img names
images = np.random.permutation(images).tolist() # ! randomize here first, so that the ordering stay the same in very other iteration

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
  labelset = sorted(['22q11DS2y', '22q11DSadolescence', '22q11DSolderadult', '22q11DSyoungadult', '22q11DSyoungchild', 'Controls2y', 'Controlsadolescence', 'Controlsolderadult', 'Controlsyoungadult', 'Controlsyoungchild', 'WS2y', 'WSadolescence', 'WSolderadult', 'WSyoungadult', 'WSyoungchild'])

if args.only_these_labels is not None: 
  temp_ = []
  for i in args.only_these_labels.strip().split(','): 
    i = i.strip() 
    temp_ = temp_ + [l for l in labelset if i in l] # get the labels we want, so we can train WS and Control only
  labelset = deepcopy(temp_)
  labelset = sorted(labelset)

if args.normal_data_path is not None: 
  if args.wholegroup:
    labelset = sorted ( labelset + ['Normal'] )
  else: 
    labelset = sorted ( labelset + ['Normal2y', 'Normaladolescence', 'Normalolderadult', 'Normalyoungadult', 'Normalyoungchild'] )
  # ! randomize just the normal dataset to keep same consistency over the other experiments
  # args.normal_data_path = '/data/duongdb/FairFace/FairFace-aligned-sub-agegroup'
  images = images + np.random.permutation( os.listdir(args.normal_data_path) ).tolist() 
  # ! read in from older experiment which normal were tested. 
  # ! WE DON'T WANT TO RESAMPLE FROM NORMAL, THIS WILL NEED SIGNIFICANT CHANGE TO THE CODE TO MAKE LABEL JSON IN STYLEGAN2
  normal_test_img = pd.read_csv (args.normal_test_img_csv)
  normal_test_img = normal_test_img[normal_test_img['fold']==0] # ! in older runs, we used fold=0 as test 
  normal_test_img = normal_test_img[normal_test_img['label'].str.contains("Normal")]
  normal_test_img = normal_test_img['name'].values.tolist() 
  qualtric_test_img = normal_test_img + qualtric_test_img # expand the set of test images 

#
labelset = {v.strip():i for i,v in enumerate(labelset)}
numlabel = len(labelset)

print ('num label ', numlabel)
print (labelset)
print ('total train + test num img {}'.format(len(images)))

for index,filename in enumerate(images): # file,age,gender,race,service_test
  #
  filename_temp = filename.split('_')[-1]
  condition = ''
  for l in labelset: # ! search the label name in the file name 
    if l in filename_temp: # ! @filename not needed in new format? 
      condition = l # string
  #
  if len(condition) == 0: 
    continue # ! if we just use WS+Control, we want to remove 22q
  # 
  if args.soft_label: 
    temp_ = [0]*numlabel
    temp_[ labelset[condition] ] = 1 # ! 1 hot array, write this out
    soft = ';'.join(str(i) for i in temp_)
  #
  fold = index % 5 # ! make 5 fold ? # @images ordering is known beforehand. 
  if args.datatype == 'test': 
    fold = 5 # ! just use test as fold=5 ------- THIS IS DIFFERENT FROM BEFORE, WHERE TEST=0
    if filename in qualtric_test_img : # ! set fold 0 as test, WE WILL WRITE IT INTO ITS OWN FILE
      images_dict['name'].append(filename)
      if 'Normal' in condition: # ! need to specify what is the test set for 'normal' 
        images_dict['path'].append(os.path.join(args.normal_data_path,filename)) 
        images_dict['is_ext'].append(1)
      else:  
        images_dict['path'].append(os.path.join(img_data_path,filename)) 
        images_dict['is_ext'].append(0)
      #
      images_dict['label'].append(condition) # ! take a string as a label
      images_dict['fold'].append(fold)
      if args.soft_label: 
        images_dict['softlabel'].append(soft)
  else: 
    if filename not in qualtric_test_img : # ! only record not test img
      images_dict['name'].append(filename)
      if 'Normal' in condition: 
        images_dict['path'].append(os.path.join(args.normal_data_path,filename)) 
        images_dict['is_ext'].append(1)
      else:  
        images_dict['path'].append(os.path.join(img_data_path,filename))
        images_dict['is_ext'].append(0)
      #
      images_dict['label'].append(condition) # ! take a string as a label
      images_dict['fold'].append(fold)
      if args.soft_label: 
        images_dict['softlabel'].append(soft)
  

# 
df = pd.DataFrame.from_dict(images_dict)
print ('df size' , df.shape)

if len(args.suffix) > 0: 
  args.suffix = '+' + args.suffix
#
df.to_csv(os.path.join(fout_path,args.datatype+args.suffix+'.csv'), index=False)

# ! look at labels count 
for i in labelset:
  print ( i,'\t',images_dict['label'].count(str(i))/df.shape[0] ) 



