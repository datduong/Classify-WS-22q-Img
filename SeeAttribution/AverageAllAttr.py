
# ! take average of img 1,2,3,.... 

import os, numpy, PIL 
import re
import numpy as np 
from PIL import Image
from copy import deepcopy

import pandas as pd 
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, balanced_accuracy_score

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--image_path', type=str) #
parser.add_argument('--output_name', type=str, default='all_img') #
parser.add_argument('--keyword', type=str, default=None) #
parser.add_argument('--maptype', type=str, default='bysideSign') #
parser.add_argument('--from_csv', type=str, default=None) # !
parser.add_argument('--filter_col', type=str, default=None) # !
parser.add_argument('--filter_by_word', type=str, default=None) # !
parser.add_argument('--suffix', type=str, default='') # ! 

args = parser.parse_args()
    
os.chdir(args.image_path)

imlist = [f for f in os.listdir(args.image_path) if args.maptype in f] 
if args.keyword is not None: 
  imlist = [ f for f in imlist if args.keyword in f ]
  args.output_name = args.output_name+args.keyword

assert len(imlist) > 0


img_list_from_csv = None # ! best to just subset based on race ?? 
if args.from_csv is not None: 
  csv = pd.read_csv(args.from_csv)
  if args.filter_col is not None: 
    csv = csv [csv[args.filter_col] == args.filter_by_word] # ! just look at these images
    print (csv)
  # 
  img_list_from_csv = sorted ( csv['name'].values )
  # now filter img list to have just these images 
  temp = []
  # print (imlist[0])
  for i in imlist: 
    shortname = '_'.join(i.split('_')[0:3]) + '.png' # 'WS_early101_WSyoungchild_bysideSignAverage.png'
    if shortname in img_list_from_csv: 
      temp.append (i)
  imlist = deepcopy(temp)
  print (args.from_csv, len(imlist))


#
assert len(imlist) > 0


w,h=Image.open(imlist[0]).size
N=len(imlist)
arr=numpy.zeros((h,w,3),numpy.float)
for im in imlist:
  imarr=numpy.array(Image.open(im),dtype=numpy.float)
  arr=arr+imarr


# Round values in array and cast as 8-bit integer
arr = arr/N # average
arr=numpy.array(numpy.round(arr),dtype=numpy.uint8)

# Generate, save and preview final image
out=Image.fromarray(arr,mode="RGB")
fout = os.path.join(args.image_path,args.output_name+args.maptype+args.suffix+'.png')
print ('save', fout)
out.save(fout)



