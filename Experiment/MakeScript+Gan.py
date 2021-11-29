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

batchsize=64 # 64 ... 64 doesn't work with new pytorch 1.7 ?? why ?? we were using 1.6 

ntest=1 # ! we tested 1, and it looks fine at 1, don't need data aug during testing

kernel_type=9c_b4ns_$imagesize'_30ep' # ! this is experiment name

suffix=NUM_TIME_MIX # ! now many times we do style mix

outdim=4

NUMFC=numberoflayers # ! final layer, should it be complicated or simple linear?? 

suffix2=M0.75T0.6AveWS+22q11DS+Control+Normal+kimg10+target0.6+blankcenter

# ! skip using $weight in the name, because $weight has "comma" ??
model_folder_name=b4ns$imagesize$imagetype'WlWEIGHTss'$schedulerscaler'lr'$learningrate'dp'$dropout'b'$batchsize'ntest'$ntest$suffix2 # ! '+Hard' 'fc'$NUMFC

cd /data/duongdb/DeployOnline/Classify-WS-22q-Img

maindir=/data/duongdb/WS22qOther_08102021/Classify

modeldir=$maindir/$model_folder_name 
mkdir $modeldir

logdir=$maindir/$model_folder_name 
oofdir=$maindir/$model_folder_name/EvalDev 

fold=FOLD

# ! train

imagecsv=$maindir/train+'F'$fold$suffix$suffix2.csv # ! train input with more normal faces + GAN, note, VALID SET HAS NO GAN

python train.py --image-csv $imagecsv --kernel-type $kernel_type --image-size $imagesize --enet-type tf_efficientnet_b4_ns --use-amp --CUDA_VISIBLE_DEVICES 0,1 --model-dir $modeldir --log-dir $logdir --num-workers 8 --fold 'FOLD' --out-dim $outdim --weighted-loss 'WEIGHT' --n-epochs 30 --batch-size $batchsize --init-lr $learningrate --scheduler-scaler $schedulerscaler --dropout $dropout --n-test $ntest --label-upweigh 'LABELUP' --num_fc $NUMFC --soft_label

# ! eval

imagecsv=$maindir/test+blankcenter+WS+22q11DS+Control+Normal+Whole+Soft.csv # ! test input ALWAYS TEST ON SAME TEST SET

python evaluate.py --image-csv $imagecsv --kernel-type $kernel_type --model-dir $modeldir --log-dir $logdir --image-size $imagesize --enet-type tf_efficientnet_b4_ns --oof-dir $oofdir --batch-size 64 --num-workers 4 --fold 'FOLD' --out-dim $outdim --CUDA_VISIBLE_DEVICES 0 --dropout $dropout --do_test --n-test $ntest --num_fc $NUMFC --soft_label # --ret_vec_rep

# ! eval on GAN

# imagecsv=$maindir/train+'F'$fold$suffix$suffix2.csv # ! not use test. use train input with more normal faces + GAN

# python evaluate.py --image-csv $imagecsv --kernel-type $kernel_type --model-dir $modeldir --log-dir $logdir --image-size $imagesize --enet-type tf_efficientnet_b4_ns --oof-dir $oofdir --batch-size 64 --num-workers 4 --fold 'FOLD' --out-dim $outdim --CUDA_VISIBLE_DEVICES 0 --dropout $dropout --n-test $ntest --do_test_manual_name '22q_Norm0.7' --ret_vec_rep --soft_label 



"""

path = '/data/duongdb/WS22qOther_08102021'
os.chdir(path)

NUM_TIME_MIX='X1'

LABELUP = '22q11DS,WS,Controls'

# LABELUP = '22q11DS2y,22q11DSadolescence,22q11DSolderadult,22q11DSyoungadult,22q11DSyoungchild,Controls2y,Controlsadolescence,Controlsolderadult,Controlsyoungadult,Controlsyoungchild,WS2y,WSadolescence,WSolderadult,WSyoungadult,WSyoungchild'

counter=0

numberoflayers=0

# for ATTRIBUTELABEL in '22q11DS,WS,Controls,Normal2y,Normaladolescence,Normalolderadult,Normalyoungadult,Normalyoungchild'.split(','): 
for fold in [0,1,2,3,4]: #  
  for imagesize in [448]: # 448 512 768 640
    for weight in ['10']: # 5,10, '5;10' # ! always use weight 1 ? # can take string a;b
      for schedulerscaler in [10]:
        for learn_rate in [0.00003]:  # 0.00001,0.00003  # we used this too, 0.0001
          for dropout in [0.2]:
            script2 = re.sub('WEIGHT',str(weight),script)
            script2 = re.sub('IMAGESIZE',str(imagesize),script2)
            script2 = re.sub('numberoflayers',str(numberoflayers),script2)
            script2 = re.sub('LABELUP',str(LABELUP),script2)
            script2 = re.sub('LEARNRATE',str(learn_rate),script2)
            script2 = re.sub('ScheduleScaler',str(schedulerscaler),script2)
            script2 = re.sub('FOLD',str(fold),script2)
            script2 = re.sub('DROPOUT',str(dropout),script2)
            script2 = re.sub('NUM_TIME_MIX',str(NUM_TIME_MIX),script2)
            # script2 = re.sub('ATTRIBUTELABEL',str(ATTRIBUTELABEL),script2)
            now = datetime.now() # current date and time
            scriptname = 'script'+str(counter)+'-'+now.strftime("%m-%d-%H-%M-%S")+'.sh'
            fout = open(scriptname,'w')
            fout.write(script2)
            fout.close()
            # 
            time.sleep(2)
            os.system('sbatch --partition=gpu --time=30:00:00 --gres=gpu:p100:2 --mem=12g --cpus-per-task=16 ' + scriptname )
            # os.system('sbatch --partition=gpu --time=00:40:00 --gres=gpu:p100:1 --mem=8g --cpus-per-task=8 ' + scriptname )
            # os.system('sbatch --time=24:00:00 --mem=8g --cpus-per-task=20 ' + scriptname )
            counter = counter + 1 


#

exit()

