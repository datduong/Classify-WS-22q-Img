
import os,sys,re,pickle 
import numpy as np 
import pandas as pd 

# ! read in prediction trained on "whole"
# ! see accuracy of each group. 

maindir = '/data/duongdb/WS22qOther_08102021/Classify/'

modeltype = ['b4ns448wl10ss10lr3e-05dp0.2b64ntest1WS+22q11DS+Control+Normal+Whole+blankcenter','b4ns448Wl10ss10lr3e-05dp0.2b64ntest1T0.6WS+22q11DS+Control+Normal+kimg10+target0.6+TransA+blankcenter']

for model in modeltype: 
  print ('\n{}'.format(model))
  fin = 'final_prediction.csv'
  df = pd.read_csv(os.path.join(maindir,model,fin)) # index,name,path,label,fold,is_ext,softlabel,target,target_soft,0,1,2,3,true_label_index,predict_label_index
  labelset = sorted(['22q11DS2y', '22q11DSadolescence', '22q11DSolderadult', '22q11DSyoungadult', '22q11DSyoungchild', 'Controls2y', 'Controlsadolescence', 'Controlsolderadult', 'Controlsyoungadult', 'Controlsyoungchild', 'WS2y', 'WSadolescence', 'WSolderadult', 'WSyoungadult', 'WSyoungchild'])
  fout = open(os.path.join(maindir,model,'final_prediction_acc_per_group.csv'),'w')
  for label in labelset: 
    df_ = df[df.name.str.contains(label)] 
    trueval = np.array ( df_.target.values ) 
    predictval = np.array ( df_.predict_label_index.values ) 
    print (label, len(predictval))
    fractcorrect = np.sum(trueval == predictval) / len(trueval)
    fout.write ('{},{}\n'.format(label,fractcorrect))
    print ('{}\t{}'.format(label,fractcorrect))
  # 
  fout.close() 

  


