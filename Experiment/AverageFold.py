
import re,sys,os,pickle
import seaborn as sn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, balanced_accuracy_score


# ! take average of confusion matrix. 
labels = 'Other HMI IP MA ML NF1 TSC'.split() # SPRED1
num_labels = len(labels)

method = 'b4ns448cropWl5ss10lr3e-05dp0.2b64ntest1' # b4ns448recropWl10ss10lr1e-05dp0.2 'b4ns448cropWl5ss10lr3e-05dp0.2'

path = '/data/duongdb/SkinConditionImages01112021/'+method 
os.chdir(path)

#
files = [f for f in os.listdir(path) if '.np' in f]
folds = np.zeros((5,num_labels,num_labels)) # 5 folds
for index,f in enumerate(files): 
  folds [index] = pickle.load(open(f,'rb'))
  
#
average = np.round (np.mean(folds,axis=0),1) ## over the folds
std = np.round (np.std(folds,axis=0), 1)


annotation = np.repeat('a',num_labels**2).reshape((num_labels,num_labels)).tolist()
for i in range(num_labels): 
  for j in range(num_labels): 
    annotation[i][j] = str(average[i,j]) + '\n\u00B1' + str(std[i,j])
    
#
df_cm = pd.DataFrame( average, 
                      index = [i for i in labels],
                      columns = [i for i in labels]).astype(float).round(3)
plt.figure(figsize=(7,6))
sn.set(font_scale=1.5) # for label size
sn.heatmap(df_cm, annot=annotation, annot_kws={"size": 16}, fmt='s', cbar=False) # font size fmt=".1f"
plt.savefig('ave_'+method+'.png') # +str(weight)

