import re,sys,os,pickle
from datetime import datetime
import time

# sbatch --partition=gpu --time=1-00:00:00 --gres=gpu:p100:2 --mem=12g --cpus-per-task=24
# sbatch --partition=gpu --time=4:00:00 --gres=gpu:p100:1 --mem=16g --cpus-per-task=24
# sbatch --partition=gpu --time=1-00:00:00 --gres=gpu:p100:2 --mem=10g --cpus-per-task=20
# sbatch --time=12:00:00 --mem=100g --cpus-per-task=24
# sinteractive --time=1:00:00 --gres=gpu:p100:1 --mem=12g --cpus-per-task=12

script = """#!/bin/bash

source /data/$USER/conda/etc/profile.d/conda.sh
conda activate py37
module load CUDA/11.0 # ! newest version at the time
module load cuDNN/8.0.3/CUDA-11.0
module load gcc/8.3.0

# ! check model name

weight=WEIGHT # ! don't need to write out weight, the X1 X10 will give us idea of size ??

learningrate=LEARNRATE
imagesize=IMAGESIZE
schedulerscaler=ScheduleScaler 
dropout=DROPOUT

batchsize=32 # 64 ... 64 doesn't work with new pytorch 1.7 ?? why ?? we were using 1.6 

ntest=1 # ! we tested 1, and it looks fine at 1, don't need data aug during testing

kernel_type=9c_b4ns_$imagesize'_30ep' # ! this is experiment name

# ! check if we use 60k or not

suffix=NUM_TIME_MIX # ! now many times we do style mix

labeltype=WholeSet

suffix2=AddGenAddNormalWholeSet # ! use full name

# ! skip using $weight in the name, because $weight has "comma" ??
model_folder_name=b4ns$imagesize$imagetype'Wl'$weight'ss'$schedulerscaler'lr'$learningrate'dp'$dropout'b'$batchsize'ntest'$ntest$suffix2$suffix

cd /data/duongdb/ClassifyFaceConditions

maindir=/data/duongdb/WS22qOther_08102021/Classify

modeldir=$maindir/$model_folder_name 
mkdir $modeldir

logdir=$maindir/$model_folder_name 
oofdir=$maindir/$model_folder_name/EvalDev 

fold=FOLD

# ! train

imagecsv=$maindir/train-oursplit'F'$fold$suffix$suffix2.csv # ! train input with more normal faces + GAN, note, VALID SET HAS NO GAN

# python train.py --image-csv $imagecsv --kernel-type $kernel_type --image-size $imagesize --enet-type tf_efficientnet_b4_ns --use-amp --CUDA_VISIBLE_DEVICES 0 --model-dir $modeldir --log-dir $logdir --num-workers 8 --fold 'FOLD' --out-dim 4 --weighted-loss 'WEIGHT' --n-epochs 30 --batch-size $batchsize --init-lr $learningrate --scheduler-scaler $schedulerscaler --dropout $dropout --n-test $ntest --label-upweigh 'LABELUP' 

# ! eval

imagecsv=$maindir/test-oursplitAddNormalWholeSet.csv # ! test input with more normal faces, ALWAYS TEST ON SAME TEST SET

# python evaluate.py --image-csv $imagecsv --kernel-type $kernel_type --model-dir $modeldir --log-dir $logdir --image-size $imagesize --enet-type tf_efficientnet_b4_ns --oof-dir $oofdir --batch-size 64 --num-workers 4 --fold 'FOLD' --out-dim 12 --CUDA_VISIBLE_DEVICES 0 --dropout $dropout --n-test $ntest

# python evaluate.py --image-csv $imagecsv --kernel-type $kernel_type --model-dir $modeldir --log-dir $logdir --image-size $imagesize --enet-type tf_efficientnet_b4_ns --oof-dir $oofdir --batch-size 32 --num-workers 4 --fold 'FOLD' --out-dim 4 --CUDA_VISIBLE_DEVICES 0 --dropout $dropout --do_test --n-test $ntest 


# ! look at pixels
# for condition in 22q11DS_early 22q11DS_inter 22q11DS_late Controls_early Controls_inter Controls_late WS_early WS_inter WS_late # Normal_early Normal_inter Normal_late
# do
# python evaluate.py --image-csv $imagecsv --kernel-type $kernel_type --model-dir $modeldir --log-dir $logdir --image-size $imagesize --enet-type tf_efficientnet_b4_ns --oof-dir $oofdir --batch-size 1 --num-workers 8 --fold 'FOLD' --out-dim 12 --dropout $dropout --do_test --n-test $ntest --attribution_keyword $condition --outlier_perc 5
# done

# for condition in MA ML HMI IP NF1 TSC EverythingElse
# do
# python evaluate.py --image-csv $imagecsv --kernel-type $kernel_type --model-dir $modeldir --log-dir $logdir --image-size $imagesize --enet-type tf_efficientnet_b4_ns --oof-dir $oofdir --batch-size 1 --num-workers 8 --fold 'FOLD' --out-dim 12 --dropout $dropout --n-test $ntest --attribution_keyword $condition --outlier_perc 1
# done

"""

path = '/data/duongdb/WS22qOther_08102021'
os.chdir(path)

# b4ns448cropWl5ss10lr3e-05dp0.2 # ! best

# b4ns448recropWl10ss10lr1e-05dp0.2 # ! best

# b4ns448Wl1ss10lr0.0001dp0.2b48ntest1

# b4ns448Wl10ss10lr1e-05dp0.2b48ntest1N60k

NUM_TIME_MIX='X1'

# b4ns448Wl1ss10lr0.0001dp0.2b32ntest1N60kGanX1

# LABELUP = 'WS_early,WS_late,WS_inter,22q11DS_early,22q11DS_late,22q11DS_inter,Controls_early,Controls_inter,Controls_late'

# LABELUP = 'WS_early,WS_late,WS_inter,22q11DS_early,22q11DS_late,22q11DS_inter;Controls_early,Controls_inter,Controls_late'

LABELUP = '22q11DS,WS,Controls'

counter=0
for fold in [1,2,3,4,5]: #  
  for imagesize in [448]: # 448 512 768 640
    for weight in ['5','10']: # 5,10, '5;10' # ! always use weight 1 ? # can take string a;b
      for schedulerscaler in [10]:
        for learn_rate in [0.00001]:  # 0.00001,0.00003  # we used this too, 0.0001
          for dropout in [0.2]:
            script2 = re.sub('WEIGHT',str(weight),script)
            script2 = re.sub('IMAGESIZE',str(imagesize),script2)
            script2 = re.sub('LABELUP',str(LABELUP),script2)
            script2 = re.sub('LEARNRATE',str(learn_rate),script2)
            script2 = re.sub('ScheduleScaler',str(schedulerscaler),script2)
            script2 = re.sub('FOLD',str(fold),script2)
            script2 = re.sub('DROPOUT',str(dropout),script2)
            script2 = re.sub('NUM_TIME_MIX',str(NUM_TIME_MIX),script2)
            now = datetime.now() # current date and time
            scriptname = 'script'+str(counter)+'-'+now.strftime("%m-%d-%H-%M-%S")+'.sh'
            fout = open(scriptname,'w')
            fout.write(script2)
            fout.close()
            # 
            time.sleep(4)
            # os.system('sbatch --partition=gpu --time=27:00:00 --gres=gpu:v100x:1 --mem=12g --cpus-per-task=16 ' + scriptname )
            os.system('sbatch --partition=gpu --time=00:30:00 --gres=gpu:p100:1 --mem=4g --cpus-per-task=4 ' + scriptname )
            # os.system('sbatch --time=24:00:00 --mem=64g --cpus-per-task=20 ' + scriptname )
            counter = counter + 1 

#

exit()



# # ! power
# from scipy.stats import norm
# import numpy as np
# percent_difference = 10
# standard_dev = 15
# people = 30
# 1 - norm.cdf ( 1.96, percent_difference/(standard_dev/np.sqrt(people)), 1) 


# # We used a one-sample t test to compare human readers and algorithms and determine whether the difference in the number of correct diagnoses in batches of 30 cases was different from 0. # ! https://www.sciencedirect.com/science/article/pii/S147020451930333X
# percent_difference = 1.9 
# standard_dev = 15
# people = 500
# 1 - norm.cdf ( 1.96, percent_difference/(standard_dev/np.sqrt(people)), 1) 
