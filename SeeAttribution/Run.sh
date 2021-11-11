
source /data/$USER/conda/etc/profile.d/conda.sh
conda activate py37

cd /data/duongdb/ClassifyFaceConditions/SeeAttribution
modeldir=/data/duongdb/WS22qOther_08102021/Classify/b4ns448wl10ss10lr3e-05dp0.2b64ntest1WS+22q11DS+Control+Normal+Whole+blankcenter/EvalTrainTest
for condition_name in WS 22q11DS 
do 
python3 AverageAttrAllFoldPerImg.py --model-dir $modeldir --fold 0,1,2,3,4 --folder_name '_Occlusion2' --condition_name $condition_name --keyword '_attr_np' --use_attr_np --positive_np
done 

# ! average all ATTRIBUTIONS INTO 1 SINGLE IMAGE FOR EACH DISEASE

cd /data/duongdb/ClassifyFaceConditions/SeeAttribution

image_path=/data/duongdb/WS22qOther_08102021/Classify/b4ns448wl10ss10lr3e-05dp0.2b64ntest1WS+22q11DS+Control+Normal+Whole+blankcenter/EvalTrainTest/AverageAttr_Occlusion2

from_csv=/data/duongdb/WS22qOther_08102021/Classify/test+WS+22q11DS+Control+Normal+Split.csv

for keyword in 22q11DS2y 22q11DSadolescence 22q11DSolderadult 22q11DSyoungadult 22q11DSyoungchild Controls2y Controlsadolescence Controlsolderadult Controlsyoungadult Controlsyoungchild WS2y WSadolescence WSolderadult WSyoungadult WSyoungchild 
do
  output_name='average_'$keyword
  python3 AverageAllAttrAsOneImg.py --image_path $image_path --output_name $output_name --keyword $keyword --from_csv $from_csv 
done 


# -------------------------------------------------------------------------------------------

# ! PERMUTATION compare attributions based on age 

# ! python code below. 

import re,sys,os,pickle
from datetime import datetime
import time

basescript = """#!/bin/bash

source /data/$USER/conda/etc/profile.d/conda.sh
conda activate py37


col_label_name='label'

csv_input=/data/duongdb/WS22qOther_08102021/Classify/test+blankcenter+WS+22q11DS+Control+Normal+Split.csv
img_dir=/data/duongdb/WS22qOther_08102021/Classify/b4ns448wl10ss10lr3e-05dp0.2b64ntest1WS+22q11DS+Control+Normal+Whole+blankcenter/EvalTrainTest/AverageAttr_Occlusion2

# csv_input=/data/duongdb/WS22qOther_08102021/Classify/test+WS+22q11DS+Control+Normal+Split.csv
# img_dir=/data/duongdb/WS22qOther_08102021/Classify/b4ns448wl10ss10lr1e-05dp0.2b64ntest1WS+22q11DS+Control+Normal+Whole/EvalTestImg/AverageAttr_test_Occlusion2

labels=LABELS

num_permu=1000

cd /data/duongdb/ClassifyFaceConditions/SeeAttribution


python3 TestHeatMapPermu.py --disease_name CONDITIONNAME --num_permu $num_permu --csv_input $csv_input --img_dir $img_dir --col_label_name $col_label_name --labels $labels --suffix _scale_np_01 --positive_np

"""

label_dict = {}
for i in ['22q11DS','WS']: # ,'Normal' 'Controls' WS
  labels = []
  for j in ['2y','youngchild','adolescence','youngadult','olderadult']:
    labels.append(i+j)
  # 
  label_dict[i] = ','.join(labels)


label_dict['22q11DS+WS'] = label_dict['22q11DS'] + ',' + label_dict['WS']

os.chdir('/data/duongdb/WS22qOther_08102021/Classify')
for index,condition in enumerate(label_dict.keys()): 
  script = re.sub('CONDITIONNAME',condition,basescript)
  script = re.sub('LABELS',label_dict[condition],script)
  now = datetime.now() # current date and time
  scriptname = 'script'+str(index)+'-'+now.strftime("%m-%d-%H-%M-%S")+'.sh'
  fout = open(scriptname,'w')
  fout.write(script)
  fout.close() 
  os.system('sbatch --time=9:00:00 --mem=8g --cpus-per-task=8 ' + scriptname)


