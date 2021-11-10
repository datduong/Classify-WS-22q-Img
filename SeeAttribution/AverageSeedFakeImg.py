import re,sys,os,pickle
import pandas as pd 
import numpy as np 

from tqdm import tqdm

from PIL import Image

import pandas as pd 
import matplotlib.pyplot as plt

# ! take seed1-cond1-mix1 average over "seed"

imgdir = '/data/duongdb/WS22qOther_08102021/Classify/b4ns448wl10ss10lr1e-05dp0.2b64ntest1WS+22q11DS+Control+Normal+Whole/AverageAttr_Occlusion2'
condition = ''
mixtype = ['M1','M0','M.2','M.4','M.6','M.8'] # ['M.2','M.4','M.6','M.8']
class1class2_list = ['0,42,4','3,42,4','1,42,4','0,52,5','3,52,5','1,52,5','0,72,7','3,72,7','1,72,7']

img_list = os.listdir(imgdir)
os.chdir(imgdir)

for mix in tqdm(mixtype): 
  for class1class2 in class1class2_list: 
    this_img_list = [i for i in img_list if (mix in i) and (class1class2 in i) and ('bysideSign' in i)] # and (cond in i)    
    #
    if len(this_img_list) == 0: 
      continue
    this_img_list = [os.path.join(imgdir,i) for i in this_img_list]
    w,h=Image.open(this_img_list[0]).size
    N=len(this_img_list)
    arr=np.zeros((h,w,3),np.float)
    #
    for im in this_img_list:
      imarr=np.array(Image.open(im),dtype=np.float)
      arr=arr+imarr
    # Round values in array and cast as 8-bit integer
    arr=np.array(np.round(arr/N),dtype=np.uint8) # ! average
    # Generate, save and preview final image
    out=Image.fromarray(arr,mode="RGB")
    foutname = os.path.join(imgdir,'Average'+class1class2+mix+'.png')
    print ('img count ', N, ' save ', foutname)
    out.save(foutname)


