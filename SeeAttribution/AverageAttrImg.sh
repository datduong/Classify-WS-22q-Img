
source /data/$USER/conda/etc/profile.d/conda.sh
conda activate py37

cd /data/duongdb/ClassifyFaceConditions/SeeAttribution
modeldir=/data/duongdb/WS22qOther_08102021/Classify/b4ns448wl10ss10lr3e-05dp0.2b64ntest1WS+22q11DS+Control+Normal+Whole+blankcenter/EvalTrainTest
for condition_name in WS 22q11DS # WS 
do 
python3 AverageAttrImg.py --model-dir $modeldir --fold 0,1,2,3,4 --folder_name '_Occlusion2' --condition_name $condition_name --keyword '_attr_np' --use_attr_np --positive_np
done 

--keyword 'heatmappositive' # ! may have to skip fold


# ! average all ATTRIBUTIONS INTO 1 SINGLE IMAGE FOR EACH DISEASE

cd /data/duongdb/ClassifyFaceConditions/SeeAttribution

image_path=/data/duongdb/WS22qOther_08102021/Classify/b4ns448wl10ss10lr3e-05dp0.2b64ntest1WS+22q11DS+Control+Normal+Whole+blankcenter/EvalTrainTest/AverageAttr_Occlusion2

from_csv=/data/duongdb/WS22qOther_08102021/Classify/test+WS+22q11DS+Control+Normal+Split.csv

for keyword in 22q11DS2y 22q11DSadolescence 22q11DSolderadult 22q11DSyoungadult 22q11DSyoungchild Controls2y Controlsadolescence Controlsolderadult Controlsyoungadult Controlsyoungchild WS2y WSadolescence WSolderadult WSyoungadult WSyoungchild # Normal2y Normaladolescence Normalolderadult Normalyoungadult Normalyoungchild 
do
  output_name='average_'$keyword
  python3 AverageAllAttr.py --image_path $image_path --output_name $output_name --keyword $keyword --from_csv $from_csv 
  # --maptype heatmappositive
done 


# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------


# ! ! can run quite slowly if we have too many images ??? so we should write a script. 

basescript = """#!/bin/bash

source /data/$USER/conda/etc/profile.d/conda.sh
conda activate py37
module load CUDA/11.0 # ! newest version at the time
module load cuDNN/8.0.3/CUDA-11.0
module load gcc/8.3.0

for keyword in heatmappositive bysideSign
do
  cd /data/duongdb/ClassifyFaceConditions/SeeAttribution

  modeldir=/data/duongdb/WS22qOther_08102021/Classify/b4ns448wl10ss10lr1e-05dp0.2b64ntest1WS+22q11DS+Control+Normal+Whole/EvalTestImg

  for condition_name in CONDITIONNAME
  do 
    python3 AverageAttrImg.py --model-dir $modeldir --fold 0,1,2,3,4 --folder_name '_test_Occlusion2' --condition_name $condition_name --keyword $keyword
  done 
done 

"""

import os,sys,re 
os.chdir('/data/duongdb/ClassifyFaceConditions')
for index,condition in enumerate(['22q11DS','WS','Controls']): # ['Controls','22q11DS','WS','Normal']
  script = re.sub('CONDITIONNAME',condition,basescript)
  fout = open('attribute'+str(index)+'.sh','w')
  fout.write(script)
  fout.close() 
  os.system('sbatch --time=2:00:00 --mem=6g --cpus-per-task=2 ' + 'attribute'+str(index)+'.sh')


# -------------------------------------------------------------------------------------------

# ! run t-test on attribution 
cd /data/duongdb/ClassifyFaceConditions/SeeAttribution
python3 TTestHeatMap_race.py


# ! average all ATTRIBUTIONS on RACE

cd /data/duongdb/ClassifyFaceConditions/SeeAttribution

image_path=/data/duongdb/WS22qOther_08102021/Classify/b4ns448wl10ss10lr3e-05dp0.2b64ntest1WS+22q11DS+Control+Normal+Whole+blankcenter/EvalTrainTest/AverageAttr_Occlusion2

# ['WS', 'WSafrican-africanamerican', 'WSasian', 'WSasian-eastern', 'WSasian-hawaiian', 'WSasian-southeastern', 'WSasian-southern', 'WScaucasian', "WSdon'tuseintestset", 'WSlatinamerican', 'WSmiddleeastern', 'WSno']


filter_col='Disease+Race'
# filter_by_word='WScaucasian'
# disease='22q11DS'

output_name='ave_'
rm -rf $image_path/$output_name'*png'

for disease in WS 22q11DS
do 
  for keyword in heatmappositive SignAverage
  do
    for filter_by_word in 'african-africanamerican' 'asian' 'asian-eastern' 'asian-hawaiian' 'asian-southeastern' 'asian-southern' 'caucasian' 'latinamerican' 'middleeastern'
    do
      from_csv='/data/duongdb/WS22qOther_08102021/Classify/'$disease'_race_blankcenter_train_test.csv'
      filter_by_word=$disease$filter_by_word
      suffix=$filter_by_word
      python3 AverageAllAttr.py --image_path $image_path --output_name $output_name --filter_col $filter_col --filter_by_word $filter_by_word --from_csv $from_csv --suffix $suffix --maptype $keyword
    done
  done
done 

# -------------------------------------------------------------------------------------------

# ! PERMUTATION is super slow. 
import re,sys,os,pickle
from datetime import datetime
import time

basescript = """#!/bin/bash

source /data/$USER/conda/etc/profile.d/conda.sh
conda activate py37

cd /data/duongdb/ClassifyFaceConditions/SeeAttribution
python3 TTestHeatMap_race_permu.py --disease_name CONDITIONNAME --num_permu 100 --sample_replace --use_sample_wt --suffix 'samplereplaceweighted'

"""
os.chdir('/data/duongdb/WS22qOther_08102021/Classify')
for index,condition in enumerate(['22q11DS','WS']): # ['Controls','22q11DS','WS','Normal']
  script = re.sub('CONDITIONNAME',condition,basescript)
  now = datetime.now() # current date and time
  scriptname = 'script'+str(index)+'-'+now.strftime("%m-%d-%H-%M-%S")+'.sh'
  fout = open(scriptname,'w')
  fout.write(script)
  fout.close() 
  os.system('sbatch --time=24:00:00 --mem=8g --cpus-per-task=4 ' + scriptname)




# -------------------------------------------------------------------------------------------

# ! PERMUTATION on age  
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

# ! --positive_np --sample_replace --use_sample_wt --suffix _age_samplereplaceweighted_$num_permu

python3 TTestHeatMap_age_permu_cosine_singlepermu.py --disease_name CONDITIONNAME --num_permu $num_permu --csv_input $csv_input --img_dir $img_dir --col_label_name $col_label_name --labels $labels --suffix _scale_np_01 --scale_np_01 --positive_np

python3 TTestHeatMap_age_permu_cosine_singlepermu.py --disease_name CONDITIONNAME --num_permu $num_permu --csv_input $csv_input --img_dir $img_dir --col_label_name $col_label_name --labels $labels --suffix _notscale_np_01 --positive_np

python3 TTestHeatMap_age_permu_cosine.py --disease_name CONDITIONNAME --num_permu $num_permu --csv_input $csv_input --img_dir $img_dir --col_label_name $col_label_name --labels $labels --suffix _scale_np_01 --scale_np_01 --positive_np

python3 TTestHeatMap_age_permu_cosine.py --disease_name CONDITIONNAME --num_permu $num_permu --csv_input $csv_input --img_dir $img_dir --col_label_name $col_label_name --labels $labels --suffix _notscale_np_01 --positive_np

# python3 TTestHeatMap_age_permu_cosine_from_img.py --disease_name CONDITIONNAME --num_permu $num_permu --csv_input $csv_input --img_dir $img_dir --col_label_name $col_label_name --labels $labels --what_image '_heatmappositiveAverage.png' --suffix '_heatmappos'


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


