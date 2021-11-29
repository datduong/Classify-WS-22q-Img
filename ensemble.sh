

# ----------------------------------------------------------------------------------------------------------------


# ! model only disease label, not consider age label

source /data/$USER/conda/etc/profile.d/conda.sh
conda activate py37

cd /data/duongdb/ClassifyFaceConditions

for modelname in b4ns448wl10ss10lr3e-05dp0.2b64ntest1WS+22q11DS+Control+Normal+Whole+blankcenter # ! change model name if needed
do
cd /data/duongdb/ClassifyFaceConditions
modeldir="/data/duongdb/WS22qOther_08102021/Classify/"$modelname
labels='22q11DS,Controls,Normal,WS' 
python3 ensemble_our_classifier.py --model-dir $modeldir --labels $labels
done 
cd $modeldir
