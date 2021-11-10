
import os,sys,re,pickle
import pandas as pd 
import numpy as np 

# ! set seed ?? 
np.random.seed(seed=1)

from copy import deepcopy

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
parser.add_argument("--soft_label", type=int, default=1)
parser.add_argument("--normal_in_gan", type=int, default=0)
parser.add_argument("--mix_ratio", type=float, default=None)
parser.add_argument("--full_fold", type=float, default=0)
parser.add_argument("--use_label_from_gan", type=str, default=None) # comma sep? 


args = parser.parse_args()

args.wholegroup = True if args.wholegroup == 1 else False
args.normal_in_gan = True if args.normal_in_gan == 1 else False

if args.use_label_from_gan is not None: 
  args.use_label_from_gan = [i.strip() for i in args.use_label_from_gan.split(',')]


# ----------------------------------------------------------------------------------------------------------------


def GetLabelFromString(namestring): # model train with split-label, 1st 3 are diseases, 2nd 5 are age-groups 
  # namestring is '0,6'
  namestring = [int (i) for i in namestring.split(',') ]
  if (namestring[0] == 2) and not args.normal_in_gan : # ! it is label,age, so only need to adjust the first one
    namestring[0] = 3 # ! WS=2 in GAN, but Normal=2 in classifier
  return namestring[0], namestring[1]


def GetLabel1Img (namestring, disease_reverse, age_reverse): 
  # seed00000026F1C0,6C1,6M.6T0.722q11DSyoungadult.png
  namestring = namestring.split('C')
  majordisease, majorage = GetLabelFromString(namestring[1]) # ! return index
  minordisease, minorage = GetLabelFromString(namestring[2][0:3]) # 1,6M.6T0.722q11DSyoungadult.png, so we take first 3
  majordiseasefullname = disease_reverse [majordisease] + age_reverse[majorage]
  minordiseasefullname = disease_reverse [minordisease] + age_reverse[minorage]
  return majordisease, majorage, minordisease, minorage, majordiseasefullname, minordiseasefullname # ! return index, and string names
  


def GetFoldFromSeedName (namestring, folds_to_use): # @folds_to_use is array [0,1,2,3], @namestring is array ['seed00005003F0C3,4C3,6M.75T0.6WS.png'...]
  # define what seeds in what fold. 
  folds_assign = {}
  len_folds_to_use = len(folds_to_use)
  seed = sorted ( list (set ([i.split('F')[0].strip() for i in namestring])) ) 
  seed = np.random.permutation(seed) # randomize ordering 
  for index,s in enumerate(seed): 
    folds_assign[s] = folds_to_use[ index % len_folds_to_use ]
  return folds_assign



# ----------------------------------------------------------------------------------------------------------------

if args.normal_in_gan:
  labelset_head = sorted('WS,22q11DS,Controls,Normal'.split(','))
else: 
  labelset_head = sorted('WS,22q11DS,Controls'.split(','))
  
# labelset_head_in_gan = {v.strip():i for i,v in enumerate(labelset_head)}
labelset_head_in_gan_reverse = {i:v.strip() for i,v in enumerate(labelset_head)} # ! gans took index as labels

if not args.normal_in_gan:
  # ! because in classifier, normal=3rd label, so we have to move "WS" into 4th position (index=3)
  labelset_head_in_gan_reverse[3] = 'WS' # 0-22q 1-control 2-ws, changes to 3-ws
  labelset_head_in_gan_reverse.pop(2,None)

print ('see label dict')
# print (labelset_head_in_gan)
print (labelset_head_in_gan_reverse)

#
labelset_tail = sorted('2y,adolescence,olderadult,youngadult,youngchild'.split(','))
labelset_tail_in_gan = {v.strip():(i+len(labelset_head)) for i,v in enumerate(labelset_tail)} # ! shift +len(labelset_head) because label takes the first 3 spots
labelset_tail_in_gan_reverse = {(i+len(labelset_head)):v.strip() for i,v in enumerate(labelset_tail)} # ! shift +len(labelset_head) because label takes the first 3 spots


# ----------------------------------------------------------------------------------------------------------------

if not os.path.exists (args.fout_path): os.mkdir(args.fout_path)

os.chdir(args.fout_path)

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
  labelset = sorted('WS,22q11DS,Controls,Normal'.split(',')) # ! put our conditions + other conditions + normal ? 
else: 
  labelset='22q11DS2y,22q11DSadolescence,22q11DSolderadult,22q11DSyoungadult,22q11DSyoungchild,Controls2y,Controlsadolescence,Controlsolderadult,Controlsyoungadult,Controlsyoungchild,WS2y,WSadolescence,WSolderadult,WSyoungadult,WSyoungchild'.split(',')
  labelset = sorted ( labelset + ['Normal2y', 'Normaladolescence', 'Normalolderadult', 'Normalyoungadult', 'Normalyoungchild'] )
  

# ! we make GAN for each label for each fold, so we have to iter through them ?
images = [j for j in os.listdir(args.img_data_path) if ('png' in j) or ('jpg' in j)] # all img names
images = sorted(images) # ! do not need to permu because we will do it in @GetFoldFromSeedName
images = {v:k for k,v in enumerate(images)} # faster look-up
print ('total num img {}'.format(len(images)))

#
labelset = {v.strip():i for i,v in enumerate(labelset)}
numlabel = len(labelset)

FOLDS = [0,1,2,3,4] # ! which fold to run mix label
divisor = 5
if args.full_fold == 0: 
  FOLDS.remove(args.fold)
  divisor = 4
#
print ('fold to use {}'.format(FOLDS))

# ! all mix images of same seed will be in same fold 
FOLDS = GetFoldFromSeedName (list (images.keys()) , FOLDS)

do_this_label = True # just set it, and use it if args.use_label_from_gan is given. 

for index,filename in enumerate(images): # file,age,gender,race,service_test
  
  if args.use_label_from_gan is not None: # ! filter out to just use "some" of the labels in GAN
    do_this_label = False
    for i in args.use_label_from_gan: 
      if i in filename: 
        do_this_label = True

  if not do_this_label: 
    continue
  
  # fold = FOLDS[index % divisor] # ! divide into @FOLDS,   FOLDS MUST BE BASED ON SEED, SO EVERYTHING GO TOGETHER 
  fold = FOLDS [ filename.split('F')[0].strip() ]
  
  # okay to use fold!=0 because @images ordering is known beforehand. 
  images_dict['name'].append(filename)
  #
  majordisease, majorage, minordisease, minorage, majordiseasefullname, minordiseasefullname = GetLabel1Img (filename, labelset_head_in_gan_reverse, labelset_tail_in_gan_reverse)
  #
  temp_ = [0]*numlabel # ! don't need to worry about "normal"
  if args.wholegroup: # whole set
    temp_[ majordisease ] = args.mix_ratio # ! 1 hot array, write this out, whole label set, so don't need age
    temp_[ minordisease ] = np.round (1-args.mix_ratio,3)
    images_dict['label'].append(labelset_head_in_gan_reverse[majordisease]) # ! take a string as a label, don't care about age here. 
  else: 
    # account for age group here. 
    temp_[ labelset[majordiseasefullname] ] = args.mix_ratio
    temp_[ labelset[minordiseasefullname] ] = np.round (1-args.mix_ratio,3)
    images_dict['label'].append(majordiseasefullname) # ! take a string as a label
  #
  soft = ';'.join(str(i) for i in temp_)
  #
  images_dict['path'].append(os.path.join(args.img_data_path,filename))
  images_dict['is_ext'].append(0)
  # 
  images_dict['fold'].append(fold)
  images_dict['softlabel'].append(soft)


# 
df = pd.DataFrame.from_dict(images_dict)
df.shape

if args.original_train_csv is not None: # ! append to original training dataset
  for f in args.original_train_csv.split(','):
    print ('read {}'.format(f))
    df_original = pd.read_csv(f)
    print ('original df size (at read in) {}'.format(df_original.shape))
    if args.keep_label_original is not None: 
      args.keep_label_original = args.keep_label_original.split(',') # keep these labels from original (so also keep the images)
      df_original = df_original[df_original.label.isin(args.keep_label_original)]
    print ('original df size (may filter out some stuffs) {}'.format(df_original.shape))
    df = pd.concat([df,df_original])


print ('final size {}'.format(df.shape))
df.to_csv(os.path.join(args.fout_path,args.datatype+'+'+args.suffix+'.csv'), index=False)

# ! look at labels count 
for i in labelset:
  print (i, images_dict['label'].count(str(i))/df.shape[0] ) 

