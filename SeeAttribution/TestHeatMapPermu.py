
import os, numpy, PIL 
import re, pickle
import numpy as np 
from PIL import Image
from numpy.core.records import array
from tqdm import tqdm 
from scipy import stats
import statsmodels

from copy import deepcopy

import pandas as pd 

import seaborn as sn
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, balanced_accuracy_score

from scipy.spatial import distance

import torch
import torch.nn as nn

import geffnet

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--disease_name', type=str) # 
parser.add_argument('--num_permu', type=int, default=100)
parser.add_argument('--sample_replace', action='store_true', default=False)
parser.add_argument('--use_sample_wt', action='store_true', default=False)
parser.add_argument('--suffix', type=str, default='') # 
parser.add_argument('--csv_input', type=str, default=None) # 
parser.add_argument('--img_dir', type=str, default=None) # 
parser.add_argument('--col_label_name', type=str, default=None) # 
parser.add_argument('--labels', type=str, default=None) # 
parser.add_argument('--enet_type', type=str, default='tf_efficientnet_b4_ns')
parser.add_argument('--positive_np', action='store_true', default=False) # ! 
parser.add_argument('--scale_np_01', action='store_true', default=False) # ! 

args = parser.parse_args()
args.disease_name = args.disease_name.split(',') # ! can pass in a list of ',' ??
if args.labels is not None: 
  args.labels = [i.strip() for i in args.labels.split(',')]
  
np.random.seed(100)


## ------------------------------------------------------------------------------------

nn_model_on_attr_vec = geffnet.create_model(args.enet_type, pretrained=True)
nn_model_on_attr_vec.classifier = nn.Identity()
nn_model_on_attr_vec.eval() # ! turn on eval mode to avoid random dropout during eval

def GetEmb (arr): # ! get embedding 
  arr = torch.FloatTensor(arr).unsqueeze(0) # batch x 3 x h x w
  arr = nn_model_on_attr_vec (arr)
  arr = arr.squeeze().detach().numpy() 
  return arr


## ------------------------------------------------------------------------------------

def ComputePairwiseDistance (image_dict,labels):
  N = len ( labels )
  DIST = np.zeros((N,N)) # ! pairwise @label1 vs @label2
  for i, label1 in enumerate(labels): 
    for j, label2 in enumerate(labels):
      if i > j : # lower triangle 
        emb1 = GetEmb ( np.mean ( image_dict[label1], 0 ) ) # ! average num_img x 3 x h x w, then get the emb
        emb2 = GetEmb ( np.mean ( image_dict[label2], 0 ) )
        DIST[i,j] = distance.euclidean( emb1 , emb2 ) # average emb-classA v.s. emb-classB
  # 
  return DIST


def PermutationTestStatistic (image_dict, labels, num_permu=10): 
  # ! run permutation ?? let's see how slow.
  num_label = len(labels) 
  all_permu_test_stat = np.zeros((num_permu,num_label,num_label))

  for i, label1 in enumerate(labels): 
    for j, label2 in enumerate(labels):

      if i < j : 
        continue # need just lower triangle 
      
      len1 = len(image_dict[label1]) # concat and then random sample 
      len2 = len(image_dict[label2])
      all_img = np.concatenate([image_dict[label1],image_dict[label2]]) # concat and then random sample 
      print ('all_img for permu', all_img.shape)

      sample_wt = None
      if args.use_sample_wt: 
        sample_wt = np.array([len1/(len1+len2) for i in range(len1)] + [len2/(len1+len2) for i in range(len2)]) # concat 
        sample_wt = sample_wt / np.sum(sample_wt) # normalize 
        
      for n in range(num_permu): 

        if args.sample_replace: 
          choice = np.random.choice(all_img.shape[0], size=all_img.shape[0], replace=True, p=sample_wt)
          temp_ = all_img [ choice ]
      
        else: 
          temp_ = np.random.permutation(all_img)

        #
        permu_dict = {  label1: temp_[0:len1], 
                        label2: temp_[len1::]} # new dict will have 2 random lists of images. 

        emb1 = GetEmb( np.mean ( permu_dict[label1], 0 ) )
        emb2 = GetEmb( np.mean ( permu_dict[label2], 0 ) )
        all_permu_test_stat [n,i,j] = distance.euclidean( emb1, emb2 )
        
  # 
  return all_permu_test_stat
  

def RankObserveTestStatistic (input_val, permu_array): 
  return np.sum(input_val > permu_array, 0) / permu_array.shape[0]


def MakeImgListForLabel(df,col_label_name): 
  df = pd.read_csv(df)
  # get all this_label ? 
  this_label_list_img = dict()
  for name,this_label in zip(df['name'].values, df[col_label_name].values ): 
    if this_label in this_label_list_img: 
      # this_label_list_img[ this_label ].append(re.sub(r'\.png','',name))
      this_label_list_img[ this_label ].append(name)
    else: 
      # this_label_list_img[ this_label ] = [ re.sub(r'\.png','',name) ] 
      this_label_list_img[ this_label ] = [ name ] 
  # delete empty or those with just 1 img ??? 
  print ('found these labels in csv ', this_label_list_img.keys() )
  return this_label_list_img


## ------------------------------------------------------------------------------------

for DISEASE in args.disease_name: 
  
  os.chdir(args.img_dir)

  image_dict_from_csv = MakeImgListForLabel(args.csv_input, args.col_label_name)

  if args.labels is None: 
    args.labels = sorted( list ( image_dict_from_csv.keys() ) )
  else: 
    args.labels = [l for l in args.labels if l in image_dict_from_csv] # ! filter to labels found in csv

  allimg = sorted ( [i for i in os.listdir(os.path.join(args.img_dir)) if '_attr_np_average.pickle' in i] ) # 

  image_dict = {} # {'A':[img1, img2]...}

  ALL_MIN = 255
  ALL_MAX = -255
  
  for label in args.labels:
    # ! get images in this @label 
    imlist = []
    for i in allimg: 
      shortname = '_'.join(i.split('_')[0:3]) + '.png' # ! take whole original name before we turn it into 'WS_early101_WSyoungchild_bysideSignAverage.png'
      if shortname in image_dict_from_csv[ label ]: 
        imlist.append (i)
    #
    if len ( imlist ) == 0: 
      continue

    temp = []
    for im in imlist: 
      
      im = np.array( pickle.load ( open( im , 'rb' ) ))
      
      if args.positive_np: 
        im [ im < 0 ] = 0 # ! just take positive ? 
      
      temp.append(im)
      
      if args.scale_np_01 : 
        if ALL_MAX < np.max(im): 
          ALL_MAX = np.max(im)
        if ALL_MIN > np.min(im): 
          ALL_MIN = np.min(im)
        
    #
    image_dict[label] = np.array(temp)


  keysval = sorted(image_dict.keys())
  for l in keysval: 
    print (l, len(image_dict[l]))
    if len(image_dict[l]) <= 1: 
      del image_dict[l]

  # ! scale into scale of 0-1 ? 
  if args.scale_np_01 : 
    ALL_MIN = abs(ALL_MIN)
    for label in image_dict: 
      image_dict[label] = (image_dict[label] + ALL_MIN ) / (ALL_MAX + ALL_MIN)
    # SEE EXAMPLE ?? 
    print ( image_dict[label][0] )

  
  # ! rename @labels again, because some images may not have been done yet. 
  print ('labels found', image_dict.keys() )

  # ! compare all pairwise 
  num_label = len(args.labels)
  
  # ! compute observed test statistic
  observe_test_stat = ComputePairwiseDistance(image_dict, args.labels)

  # ! do permutation 
  permu = PermutationTestStatistic (image_dict, args.labels, num_permu=args.num_permu)
  rank_of_obs_test_stat = RankObserveTestStatistic (observe_test_stat, permu)
  
  # ! save 

  if args.positive_np: 
    args.suffix = args.suffix + '_posnp'
  
  foutname = DISEASE+'_emb_cosdist_'+str(args.num_permu)+args.suffix

  pickle.dump (rank_of_obs_test_stat,open(os.path.join(args.img_dir,foutname+'.pickle'),'wb'))

  print ('\nobserve_test_stat\n', observe_test_stat)    
  print ('\ndistance\n', rank_of_obs_test_stat)
  
  df_cm = pd.DataFrame( rank_of_obs_test_stat, 
                        index = [i for i in args.labels],
                        columns = [i for i in args.labels]).astype(float).round(3)

  mask = np.zeros_like(rank_of_obs_test_stat, dtype=np.bool)
  mask[np.triu_indices_from(mask)] = True

  plt.figure(figsize=(10,10))
  sn.set(font_scale=2) # for label size
  sn.heatmap(df_cm, annot=True, annot_kws={"size": 20}, fmt=".2f", cbar=False, mask=mask) # font size
  plt.savefig(os.path.join(args.img_dir,foutname+'.png'))


