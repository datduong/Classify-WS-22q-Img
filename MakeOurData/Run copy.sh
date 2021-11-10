
#!/bin/bash

# sinteractive --time=2:00:00 --gres=gpu:p100:1 --mem=8g --cpus-per-task=6
# sbatch --partition=gpu --time=12:00:00 --gres=gpu:p100:1 --mem=8g --cpus-per-task=8 


source /data/$USER/conda/etc/profile.d/conda.sh
conda activate py37

module load CUDA/11.0
module load cuDNN/8.0.3/CUDA-11.0
module load gcc/8.3.0


# ------------------------------------------------------------------------------------------------------

# ! add soft label images to train/test !!! 2 WAYS ?

suffix2=X1
wholegroup=1

normal_in_gan=1

mix_ratio='0.75' # '1' # '0.75'
mixtype=M$mix_ratio'T0.6Ave'  # ! what is the mix ratio and the truncation for clear images

# mixtype='T0.6'  # ! what is the mix ratio and the truncation for clear images

full_fold=0

suffix1=WS+22q11DS+Control+Normal+kimg10+target0.6+blankcenter

# WS+22q11DS+Control+Normal+kimg10+target0.6 # 
# suffix1=WS+22q11DS+Control+Normal+kimg10+target0.6+TransA
# suffix1=WS+22q11DS+Control+Normal+kimg10+target0.6+DiscA

suffix3='blankcenter+WS+22q11DS+Control+Normal+Whole+Soft' # train+WS+22q11DS+Control+Normal+Whole+Soft.csv

cd /data/duongdb/ClassifyFaceConditions/MakeOurData

for fold in 0 1 2 3 4 # exclude valid fold, so GAN has to be adjusted for each fold 1 2 3 4
do

img_data_path=/data/duongdb/WS22qOther_08102021/Classify/$suffix1$mixtype/F$fold$suffix2

fout_path=/data/duongdb/WS22qOther_08102021/Classify/

# ! 2 ways
python3 MakeCsvGanImgSoft_FixSeed.py --fout_path $fout_path --img_data_path $img_data_path --datatype 'train' --fold $fold --wholegroup $wholegroup --suffix F$fold$suffix2$mixtype$suffix1 --soft_label 1 --normal_in_gan $normal_in_gan --mix_ratio $mix_ratio --original_train_csv /data/duongdb/WS22qOther_08102021/Classify/train+$suffix3.csv --full_fold $full_fold 


# ! not mix
# python3 MakeCsvGanImgSoft_NotMix.py --fout_path $fout_path --img_data_path $img_data_path --datatype 'train' --fold $fold --wholegroup $wholegroup --suffix F$fold$suffix2$mixtype$suffix1 --soft_label 1 --normal_in_gan $normal_in_gan --original_train_csv /data/duongdb/WS22qOther_08102021/Classify/train+$suffix3.csv --full_fold $full_fold 

# ! age progress, all age in same fold , can be also used when we mix age|fixed_label
# python3 MakeCsvGanImgSoft_TransA.py --fout_path $fout_path --img_data_path $img_data_path --datatype 'train' --fold $fold --wholegroup $wholegroup --suffix F$fold$suffix2$mixtype$suffix1 --soft_label 1 --normal_in_gan $normal_in_gan --original_train_csv /data/duongdb/WS22qOther_08102021/Classify/train+$suffix3.csv --full_fold $full_fold 


done
cd $fout_path



# j = """24762148
# 24762231
# 24762232
# 24762233
# 24762234
# 24762388
# 24762434
# 24762435
# 24762516
# 24762517""".split()
# import os 
# for i in j: 
#   os.system ('scancel '+i)

  