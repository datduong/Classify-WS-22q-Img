import re,sys,os,pickle
from datetime import datetime
import time

# sbatch --partition=gpu --time=1-00:00:00 --gres=gpu:p100:2 --mem=12g --cpus-per-task=24
# sbatch --partition=gpu --time=4:00:00 --gres=gpu:p100:1 --mem=16g --cpus-per-task=24
# sbatch --partition=gpu --time=1-00:00:00 --gres=gpu:p100:2 --mem=10g --cpus-per-task=20
# sbatch --time=12:00:00 --mem=100g --cpus-per-task=24
# sinteractive --time=1:00:00 --gres=gpu:p100:1 --mem=12g --cpus-per-task=12

# ! ! LET'S TEST ON THE GAN IMAGES BEFORE WE USE THEM


script = """#!/bin/bash

source /data/$USER/conda/etc/profile.d/conda.sh
conda activate py37
module load CUDA/11.0 # ! newest version at the time
module load cuDNN/8.0.3/CUDA-11.0
module load gcc/8.3.0

# ! check model name
weight=WEIGHT
learningrate=LEARNRATE
imagesize=IMAGESIZE
schedulerscaler=ScheduleScaler 
dropout=DROPOUT

batchsize=64 # 64 ... 64 doesn't work with new pytorch 1.7 ?? why ?? we were using 1.6 

ntest=1 # ! we tested 1, and it looks fine at 1, don't need data aug during testing

kernel_type=9c_b4ns_$imagesize'_30ep' # ! this is experiment name

suffix=SUFFIX

# ! check if we use 60k or not
model_folder_name=b4ns$imagesize$imagetype'wl'$weight'ss'$schedulerscaler'lr'$learningrate'dp'$dropout'b'$batchsize'ntest'$ntest$suffix

maindir=/data/duongdb/WS22qOther_08102021/Classify

modeldir=$maindir/$model_folder_name 
mkdir $modeldir

logdir=$maindir/$model_folder_name 
oofdir=$maindir/$model_folder_name/EvalDev 

cd /data/duongdb/ClassifyFaceConditions

# ! eval

imagecsv=$maindir/train+F0X1M0.7T0.7+22q11DS-Normal+Whole.csv # ! GAN AS OUR TEST IMAGES 

python evaluate.py --image-csv $imagecsv --kernel-type $kernel_type --model-dir $modeldir --log-dir $logdir --image-size $imagesize --enet-type tf_efficientnet_b4_ns --oof-dir $oofdir --batch-size 64 --num-workers 4 --fold 'FOLD' --out-dim 4 --CUDA_VISIBLE_DEVICES 0 --dropout $dropout --do_test_manual_name '22q_Norm0.7' --n-test $ntest --ret_vec_rep 

"""

path = '/data/duongdb/WS22qOther_08102021'
os.chdir(path)


SUFFIX='WS+22q11DS+Control+Normal+Whole'
# if 'Whole' in SUFFIX: 
#   LABELUP = '22q11DS,Controls,WS'
# else: 
#   LABELUP = '22q11DS2y,22q11DSadolescence,22q11DSolderadult,22q11DSyoungadult,22q11DSyoungchild,Controls2y,Controlsadolescence,Controlsolderadult,Controlsyoungadult,Controlsyoungchild,WS2y,WSadolescence,WSolderadult,WSyoungadult,WSyoungchild' # 20 labels with normal

LABELUP = '22q11DS,WS,Controls'

counter=0
for fold in [0]: # 0,1,2,3,4,5 '0','1','2','3','4'
  for imagesize in [448]: # 448 512 768 640
    for weight in ['10']: # 5,10,
      for schedulerscaler in [10]:
        for learn_rate in [0.00001]:  # 0.00001,0.00003  # we used this too, 0.0001
          for dropout in [0.2]:
            script2 = re.sub('WEIGHT',str(weight),script)
            script2 = re.sub('LABELUP',str(LABELUP),script2)
            script2 = re.sub('IMAGESIZE',str(imagesize),script2)
            script2 = re.sub('SUFFIX',str(SUFFIX),script2)
            script2 = re.sub('LEARNRATE',str(learn_rate),script2)
            script2 = re.sub('ScheduleScaler',str(schedulerscaler),script2)
            script2 = re.sub('FOLD',str(fold),script2)
            script2 = re.sub('DROPOUT',str(dropout),script2)
            now = datetime.now() # current date and time
            scriptname = 'script'+str(counter)+'-'+now.strftime("%m-%d-%H-%M-%S")+'.sh'
            fout = open(scriptname,'w')
            fout.write(script2)
            fout.close()
            # 
            time.sleep( 1 )
            # os.system('sbatch --partition=gpu --time=30:00:00 --gres=gpu:v100x:1 --mem=12g --cpus-per-task=16 ' + scriptname )
            os.system('sbatch --partition=gpu --time=00:20:00 --gres=gpu:p100:1 --mem=6g --cpus-per-task=8 ' + scriptname )
            # os.system('sbatch --time=24:00:00 --mem=64g --cpus-per-task=20 ' + scriptname )
            counter = counter + 1 

#
exit()


