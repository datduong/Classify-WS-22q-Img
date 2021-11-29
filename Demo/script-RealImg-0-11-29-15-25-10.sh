#!/bin/bash

source /data/$USER/conda/etc/profile.d/conda.sh
conda activate py37
module load CUDA/11.0 # ! newest version at the time
module load cuDNN/8.0.3/CUDA-11.0
module load gcc/8.3.0

# ! check model name
weight=10
learningrate=1e-05
imagesize=448
schedulerscaler=10 
dropout=0.2

batchsize=64 # 64 ... 64 doesn't work with new pytorch 1.7 ?? why ?? we were using 1.6 

ntest=1 # ! we tested 1, and it looks fine at 1, don't need data aug during testing

kernel_type=9c_b4ns_$imagesize'_30ep' # ! this is experiment name

suffix=WS+22q11DS+Control+Normal+Whole

NUMFC=0 # ! final layer, should it be complicated or simple linear?? 

model_folder_name=Demo-b4ns$imagesize$imagetype'wl'$weight'ss'$schedulerscaler'lr'$learningrate'dp'$dropout'b'$batchsize'ntest'$ntest$suffix # ! name your model

codedir=/data/duongdb/DeployOnline/Classify-WS-22q-Img # ! change to your own path on your machine

cd $codedir 

workdir=$codedir/Demo # ! change to your own path on your machine

modeldir=$workdir/$model_folder_name 
mkdir $modeldir

logdir=$workdir/$model_folder_name 

oofdir=$workdir/$model_folder_name/EvalTestImg # ! change if needed

# ! train
imagecsv=$codedir/Experiment/TrainTestCsv/RealImg/train+blankcenter+WS+22q11DS+Control+Normal.csv # ! train input 

python train.py --image-csv $imagecsv --kernel-type $kernel_type --image-size $imagesize --enet-type tf_efficientnet_b4_ns --use-amp --CUDA_VISIBLE_DEVICES 0,1 --model-dir $modeldir --log-dir $logdir --num-workers 8 --fold '0' --out-dim 4 --weighted-loss $weight --n-epochs 30 --batch-size $batchsize --init-lr $learningrate --scheduler-scaler $schedulerscaler --dropout $dropout --n-test $ntest --label-upweigh '22q11DS,Controls,WS' --num_fc $NUMFC


# ! eval

# imagecsv=$codedir/Experiment/TrainTestCsv/RealImg/test+blankcenter+WS+22q11DS+Control+Normal.csv # ! test input 

# python evaluate.py --image-csv $imagecsv --kernel-type $kernel_type --model-dir $modeldir --log-dir $logdir --image-size $imagesize --enet-type tf_efficientnet_b4_ns --oof-dir $oofdir --batch-size 64 --num-workers 4 --fold '0' --out-dim 4 --CUDA_VISIBLE_DEVICES 0 --dropout $dropout --do_test --n-test $ntest --num_fc $NUMFC # --ret_vec_rep  # ! actual test set


# ! look at pixels # # ! --do_test --do_test_manual_name $do_test_manual_name

# for condition in ATTRIBUTELABEL 
# do

# python evaluate.py --image-csv $imagecsv --kernel-type $kernel_type --model-dir $modeldir --log-dir $logdir --image-size $imagesize --enet-type tf_efficientnet_b4_ns --oof-dir $oofdir --batch-size 1 --num-workers 8 --fold '0' --out-dim 4 --dropout $dropout --n-test $ntest --attribution_keyword $condition --outlier_perc 2 --attribution_model Occlusion --do_test --attr_np_as_vec 

# done

