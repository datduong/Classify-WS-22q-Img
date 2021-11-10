
#!/bin/bash

# sinteractive --time=2:00:00 --gres=gpu:p100:1 --mem=12g
# sinteractive --time=2:00:00 --gres=gpu:p100:1 --mem=8g --cpus-per-task=6
# sinteractive --time=1:00:00 --gres=gpu:v100x:1 --mem=20g --cpus-per-task=32 
# sbatch --partition=gpu --time=12:00:00 --gres=gpu:p100:1 --mem=8g --cpus-per-task=8 
# sbatch --partition=gpu --time=1-00:00:00 --gres=gpu:v100x:2 --mem=20g --cpus-per-task=20 

source /data/$USER/conda/etc/profile.d/conda.sh
conda activate py37

module load CUDA/11.0
module load cuDNN/8.0.3/CUDA-11.0
module load gcc/8.3.0


# ------------------------------------------------------------------------------------------------------

# ! add in normal faces, let's try just WS + Controls 

normal_data_path=/data/duongdb/FairFace/FairFace-aligned-60k-agegroup-06012021 # ! normal faces aligned as in stylegan

soft=0
wholegroup=1

root_data_path=/data/duongdb/WS22qOther_08102021

# ! specify the test set 
qualtric_test_img_csv=$root_data_path/'WS_Control_test_img_list_survey_WS.csv' 
normal_test_img_csv=/data/duongdb/WS22qOther_06012021/Classify/test-oursplitAddNormalWholeSet.csv # ! previous test set for normal faces... not matter because we don't care about normal, we just need a big 2nd dataset

if [ $soft = 0 ] 
then 
  suffix=WS+Control+Normal+Whole
else
  suffix=WS+Control+Normal+Whole+Soft
fi  

cd /data/duongdb/ClassifyFaceConditions/MakeOurData

python3 MakeCsvFaceCondition.py --datatype 'train' --root_data_path $root_data_path --source_folder 'Align512' --suffix $suffix --soft_label $soft --wholegroup $wholegroup --normal_data_path $normal_data_path --qualtric_test_img_csv $qualtric_test_img_csv --only_these_labels WS,Controls --normal_test_img_csv $normal_test_img_csv

python3 MakeCsvFaceCondition.py --datatype 'test' --root_data_path $root_data_path --source_folder 'Align512' --suffix $suffix --soft_label $soft --wholegroup $wholegroup --normal_data_path $normal_data_path --qualtric_test_img_csv $qualtric_test_img_csv --only_these_labels WS,Controls --normal_test_img_csv $normal_test_img_csv


# ------------------------------------------------------------------------------------------------------

# ! add in normal faces, let's try just 22q + Controls ... not use WS

normal_data_path=/data/duongdb/FairFace/FairFace-aligned-60k-agegroup-06012021 # ! normal faces aligned as in stylegan

soft=0
wholegroup=1

root_data_path=/data/duongdb/WS22qOther_08102021

# ! specify the test set 
qualtric_test_img_csv=$root_data_path/'22q11DS_Controls_test_img_list_survey_22q11DS.csv' 
normal_test_img_csv=/data/duongdb/WS22qOther_06012021/Classify/test-oursplitAddNormalWholeSet.csv # ! previous test set for normal faces... not matter because we don't care about normal, we just need a big 2nd dataset

if [ $soft = 0 ] 
then 
  suffix=22q11DS+Control+Normal+Whole
else
  suffix=22q11DS+Control+Normal+Whole+Soft
fi  

cd /data/duongdb/ClassifyFaceConditions/MakeOurData

python3 MakeCsvFaceCondition.py --datatype 'train' --root_data_path $root_data_path --source_folder 'Align512' --suffix $suffix --soft_label $soft --wholegroup $wholegroup --normal_data_path $normal_data_path --qualtric_test_img_csv $qualtric_test_img_csv --only_these_labels 22q11DS,Controls --normal_test_img_csv $normal_test_img_csv

python3 MakeCsvFaceCondition.py --datatype 'test' --root_data_path $root_data_path --source_folder 'Align512' --suffix $suffix --soft_label $soft --wholegroup $wholegroup --normal_data_path $normal_data_path --qualtric_test_img_csv $qualtric_test_img_csv --only_these_labels 22q11DS,Controls --normal_test_img_csv $normal_test_img_csv




# ------------------------------------------------------------------------------------------------------

# ! add in normal faces, use all WS + 22q + Controls 

only_these_labels='WS,22q11DS,Controls' # ! add in normal faces, use all WS + 22q + Controls 

normal_data_path=/data/duongdb/FairFace/FairFace-aligned-60k-agegroup-06012021 # ! normal faces aligned as in stylegan

soft=1
wholegroup=1

root_data_path=/data/duongdb/WS22qOther_08102021

# ! specify the test set 
qualtric_test_img_csv=$root_data_path/'WS_22q11DS_controls_test_img_list.csv' 

normal_test_img_csv=/data/duongdb/WS22qOther_06012021/Classify/test-oursplitAddNormalWholeSet.csv # ! previous test set for normal faces... not matter because we don't care about normal, we just need a big 2nd dataset

if [ $soft = 0 ] 
then 
  suffix=WS+22q11DS+Control+Normal+Whole # ! name
else
  suffix=WS+22q11DS+Control+Normal+Whole+Soft
fi  

cd /data/duongdb/ClassifyFaceConditions/MakeOurData

python3 MakeCsvFaceCondition.py --datatype 'train' --root_data_path $root_data_path --source_folder 'Align512' --suffix $suffix --soft_label $soft --wholegroup $wholegroup --normal_data_path $normal_data_path --qualtric_test_img_csv $qualtric_test_img_csv --only_these_labels $only_these_labels --normal_test_img_csv $normal_test_img_csv 

python3 MakeCsvFaceCondition.py --datatype 'test' --root_data_path $root_data_path --source_folder 'Align512' --suffix $suffix --soft_label $soft --wholegroup $wholegroup --normal_data_path $normal_data_path --qualtric_test_img_csv $qualtric_test_img_csv --only_these_labels $only_these_labels --normal_test_img_csv $normal_test_img_csv
cd $root_data_path

# ------------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------------


# ! add soft label images to train/test !!! 4 WAYS MIX or 2 WAYS ?

suffix2=X1
wholegroup=1

normal_in_gan=1

mix_ratio='0.7'
mixtype=M$mix_ratio'T0.7'  # ! what is the mix ratio and the truncation for clear images

full_fold=0

if [ $wholegroup = 1 ] 
then 
  suffix1=blankcenter+WS+22q11DS+Control+Normal # +InEval
  suffix3=blankcenter+WS+22q11DS+Control+Normal+Whole+Soft
else
  suffix1=blankcenter+WS+22q11DS+Control+Normal # +InEval # WS+22q11DS+Control+Normal+
  suffix3=blankcenter+WS+22q11DS+Control+Normal+Whole+Soft # train+WS+22q11DS+Control+Normal+Whole+Soft.csv
fi  

cd /data/duongdb/ClassifyFaceConditions/MakeOurData

for fold in 0 1 2 3 4  # exclude valid fold, so GAN has to be adjusted for each fold
do

img_data_path=/data/duongdb/WS22qOther_08102021/Classify/blankcenter+WS+22q11DS+Control+Normal+Split$mixtype/F$fold$suffix2

fout_path=/data/duongdb/WS22qOther_08102021/Classify/

# ! 4 ways
# python3 MakeCsvGanImgSoft4way.py --fout_path $fout_path --img_data_path $img_data_path --datatype 'train' --fold $fold --wholegroup $wholegroup --suffix F$fold$suffix2$mixtype'+SoftImg+'$suffix1 --original_train_csv /data/duongdb/WS22qOther_08102021/Classify/train+$suffix3.csv --soft_label 1 --normal_in_gan $normal_in_gan 

# ! 2 ways
python3 MakeCsvGanImgSoft.py --fout_path $fout_path --img_data_path $img_data_path --datatype 'train' --fold $fold --wholegroup $wholegroup --suffix F$fold$suffix2$mixtype$suffix1 --soft_label 1 --normal_in_gan $normal_in_gan --mix_ratio $mix_ratio --original_train_csv /data/duongdb/WS22qOther_08102021/Classify/train+$suffix3.csv --full_fold $full_fold

done
cd $fout_path



## !!


# ! add soft label images to train/test !!! 4 WAYS MIX or 2 WAYS ?

suffix2=X1
wholegroup=1

normal_in_gan=1

mix_ratio='0.7'
mixtype=M$mix_ratio'T0.5'  # ! what is the mix ratio and the truncation for clear images

full_fold=0

if [ $wholegroup = 1 ] 
then 
  suffix1=blankcenter+WS+22q11DS+Control+Normal+JustFake # +InEval
  suffix3=blankcenter+WS+22q11DS+Control+Normal+Whole+Soft
else
  suffix1=blankcenter+WS+22q11DS+Control+Normal+JustFake # +InEval # WS+22q11DS+Control+Normal+
  suffix3=blankcenter+WS+22q11DS+Control+Normal+Whole+Soft # train+WS+22q11DS+Control+Normal+Whole+Soft.csv
fi  

cd /data/duongdb/ClassifyFaceConditions/MakeOurData

for fold in 0 1 2 3 4  # exclude valid fold, so GAN has to be adjusted for each fold
do

img_data_path=/data/duongdb/WS22qOther_08102021/Classify/blankcenter+WS+22q11DS+Control+Normal+Split$mixtype/F$fold$suffix2

fout_path=/data/duongdb/WS22qOther_08102021/Classify/

# ! 4 ways
# python3 MakeCsvGanImgSoft4way.py --fout_path $fout_path --img_data_path $img_data_path --datatype 'train' --fold $fold --wholegroup $wholegroup --suffix F$fold$suffix2$mixtype'+SoftImg+'$suffix1 --original_train_csv /data/duongdb/WS22qOther_08102021/Classify/train+$suffix3.csv --soft_label 1 --normal_in_gan $normal_in_gan 

# ! 2 ways
python3 MakeCsvGanImgSoft.py --fout_path $fout_path --img_data_path $img_data_path --datatype 'train' --fold $fold --wholegroup $wholegroup --suffix F$fold$suffix2$mixtype$suffix1 --soft_label 1 --normal_in_gan $normal_in_gan --mix_ratio $mix_ratio --full_fold $full_fold

done
cd $fout_path



# DEBUG PROB OF FAKE IMAGES

# ! add soft label images to train/test !!! 4 WAYS MIX or 2 WAYS ?

suffix2=X1
wholegroup=1

normal_in_gan=1

mixtype=M0.1T0.7  # ! what is the mix ratio and the truncation for clear images
mix_ratio=0.1

if [ $wholegroup = 1 ] 
then 
  suffix1=Whole
  suffix3=Whole+Soft
else
  suffix1=Whole # WS+22q11DS+Control+Normal+
  suffix3=Whole+Soft
fi  

cd /data/duongdb/ClassifyFaceConditions/MakeOurData

for fold in 0  # exclude valid fold, so GAN has to be adjusted for each fold
do

img_data_path=/data/duongdb/WS22qOther_08102021/Classify/WS+22q11DS+Control+Normal+Split+GenSoft22q11DS-Normal$mixtype/F$fold$suffix2

fout_path=/data/duongdb/WS22qOther_08102021/Classify/

# ! 4 ways
# python3 MakeCsvGanImgSoft4way.py --fout_path $fout_path --img_data_path $img_data_path --datatype 'train' --fold $fold --wholegroup $wholegroup --suffix F$fold$suffix2$mixtype'+SoftImg+'$suffix1 --original_train_csv /data/duongdb/WS22qOther_08102021/Classify/train+$suffix3.csv --soft_label 1 --normal_in_gan $normal_in_gan 

# ! 2 ways
# --original_train_csv /data/duongdb/WS22qOther_08102021/Classify/train+$suffix3.csv # ! add GAN to original train/test dataset
python3 MakeCsvGanImgSoft.py --fout_path $fout_path --img_data_path $img_data_path --datatype 'train' --fold $fold --wholegroup $wholegroup --suffix F$fold$suffix2$mixtype'+22q11DS-Normal+'$suffix1 --soft_label 1 --normal_in_gan $normal_in_gan --mix_ratio $mix_ratio

done
cd $fout_path


# ------------------------------------------------------------------------------------------------------

# ! REMOVE EXTREME GAN IMAGES 
# ! add soft label images to train/test !!! 4 WAYS MIX or 2 WAYS ?

suffix2=X1
wholegroup=1

normal_in_gan=1

mixtype=M0.7T0.7  # ! what is the mix ratio and the truncation for clear images
mix_ratio=0.7

if [ $wholegroup = 1 ] 
then 
  suffix1=Filter+Whole
  suffix3=WS+22q11DS+Control+Normal+Whole+Soft
else
  suffix1=Filter+Whole # WS+22q11DS+Control+Normal+
  suffix3=WS+22q11DS+Control+Normal+Whole+Soft
fi  

cd /data/duongdb/ClassifyFaceConditions/MakeOurData

for fold in 0 1 2 3 4  # exclude valid fold, so GAN has to be adjusted for each fold
do

img_data_path=/data/duongdb/WS22qOther_08102021/Classify/WS+22q11DS+Control+Normal+Split+GenSoftImg22q11DS-Controls$mixtype/F$fold$suffix2

fout_path=/data/duongdb/WS22qOther_08102021/Classify/

gan_csv=/data/duongdb/WS22qOther_08102021/Classify/b4ns448wl10ss10lr1e-05dp0.2b64ntest1WS+22q11DS+Control+Normal+Whole/test_on_gan_from_fold$fold.csv

python3 KeepHighProbFakeImg.py --fout_path $fout_path --img_data_path $img_data_path --datatype 'train' --fold $fold --wholegroup $wholegroup --suffix F$fold$suffix2$mixtype'+SoftImg22q11DS-Controls+'$suffix1 --soft_label 1 --normal_in_gan $normal_in_gan --mix_ratio $mix_ratio --original_train_csv /data/duongdb/WS22qOther_08102021/Classify/train+$suffix3.csv --high_prob 1 --low_prob 0 --gan_csv $gan_csv

done
cd $fout_path


# ------------------------------------------------------------------------------------------------------

