
import os, numpy, re, pickle
import numpy as np 
import pandas as pd 
from PIL import Image

from copy import deepcopy

from tqdm import tqdm 

import torch
import torch.nn as nn

import geffnet

# ! because we use 5-fold cv. we can average attribution for each fold. 
# https://stackoverflow.com/questions/17291455/how-to-get-an-average-picture-from-100-pictures-using-pil

# Access all PNG files in directory
# allfiles=os.listdir(os.getcwd())
# imlist=[filename for filename in allfiles if  filename[-4:] in [".png",".PNG",".jpeg",".jpg"]]

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model-dir', type=str) # /data/duongdb/WS22qOther_05052021/Classify/b4ns448Wl1ss10lr0.0001dp0.2b48ntest1
parser.add_argument('--fold', type=str, help='something like 1,2,3')
parser.add_argument('--keyword', type=str, default='Sign') #
parser.add_argument('--condition_name', type=str, default=None)
parser.add_argument('--folder_name', type=str, default='_test_Occlusion2') # ! _Occlusion2 or _test_Occlusion2
parser.add_argument('--from_csv', type=str, default=None) # !
parser.add_argument('--suffix', type=str, default='') # ! 
parser.add_argument('--use_attr_np', action='store_true', default=False) # ! 
parser.add_argument('--positive_np', action='store_true', default=False) # ! 
parser.add_argument('--enet_type', type=str, default='tf_efficientnet_b4_ns')


args = parser.parse_args()

folder_list_todo = args.folder_name.split(',')

if args.positive_np: args.suffix = args.suffix + '_pos_' 

maindir = os.path.join(args.model_dir) # ,'EvalDev', should just put @AverageAttr inside @EvalDev, so we remove "EvalDev" in "join dir"

# ! best to just subset based on race ?? 
img_list_from_csv = None
if args.from_csv is not None: 
  csv = pd.read_csv(args.from_csv)
  img_list_from_csv = sorted ( csv['name'].values ) 


if args.use_attr_np:
  nn_model_on_attr_vec = geffnet.create_model(args.enet_type, pretrained=True)
  nn_model_on_attr_vec.classifier = nn.Identity()
  nn_model_on_attr_vec.eval() # ! turn on eval mode to avoid random dropout during eval

for level in folder_list_todo: # _test_1 # ,'_test_integrated_gradient2'

  print ('\n', level)
  
  outdir = os.path.join(args.model_dir,'AverageAttr'+level)
  
  if not os.path.exists(outdir): 
    os.mkdir(outdir)

  # 
  fold = [ str(i.strip()) + level for i in args.fold.split(',')]

  os.chdir(maindir)

  imlist_in_1_fold = sorted ( os.listdir(os.path.join(maindir,fold[0])) ) 
  imlist_in_1_fold = [i for i in imlist_in_1_fold if args.keyword in i] # ! what are we averaging ? 

  if args.condition_name is not None: 
    imlist_in_1_fold = [i for i in imlist_in_1_fold if args.condition_name in i] # disease name

  if img_list_from_csv is not None: # WS_early2_WS2y.png, Controls_early288_Controlsadolescence_bysideSignAverage.png
    temp = []
    for i in imlist_in_1_fold: 
      i = '_'.join(i.split('_')[0:3])+'.png'
      if i in img_list_from_csv: 
        temp.append (i)
    #
    imlist_in_1_fold = deepcopy(temp)
    print (imlist_in_1_fold)
    
  # ! 
  # this_img = imlist_in_1_fold[0]

  counter = 0 
  
  for this_img in tqdm(imlist_in_1_fold): 

    imlist = [os.path.join(maindir,i,this_img) for i in fold]
    if any ( not os.path.exists( i ) for i in imlist ): # ! may not have all images done?  
      continue

    N=len(imlist)

    if args.use_attr_np: # ! combine attribution in raw values or after they are projected into vector space? 
      arr = []
      for im in imlist: 
        temp = pickle.load(open(im,'rb'))
        if args.positive_np: 
          temp [ temp < 0 ] = 0 # ! run just positive ??
        arr.append ( temp ) 

      arr = np.array(arr) # 5 fold x 3 channel x size x size 
      arr = np.mean(arr, axis=0) # average matrices 
      pickle.dump (arr, open ( os.path.join(outdir,re.sub(r"(\.png|\.jpg|\.pickle)","",this_img)) + args.suffix + "_average.pickle", 'wb') ) 

      # ! get embedding 
      arr = torch.FloatTensor(arr).unsqueeze(0) # batch x 3 x h x w
      arr = nn_model_on_attr_vec (arr)
      arr = arr.squeeze().detach().numpy() 
      pickle.dump (arr, open ( os.path.join(outdir,re.sub(r"(\.png|\.jpg|\.pickle)","",this_img)) + args.suffix + "_attr_as_vec_average.pickle", 'wb') )
      
    else: 
      # Assuming all images are the same size, get dimensions of first image
      w,h=Image.open(imlist[0]).size
    
      # Create a numpy array of floats to store the average (assume RGB images)
      arr=numpy.zeros((h,w,3),numpy.float)

      # Build up average pixel intensities, casting each image as an array of floats
      for im in imlist:
        imarr=numpy.array(Image.open(im),dtype=numpy.float)
        arr=arr+imarr/N

      # Round values in array and cast as 8-bit integer
      arr=numpy.array(numpy.round(arr),dtype=numpy.uint8)

      # Generate, save and preview final image
      out=Image.fromarray(arr,mode="RGB")
      out.save(os.path.join(outdir,re.sub(r"(\.png|\.jpg)","",this_img)) + args.suffix + "Average.png")
      # out.show()

    # count how many img been done
    counter = counter + 1 


  # ! print counter just to back track ??
  print ('num img done', counter)


