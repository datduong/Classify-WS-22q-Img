
cd /data/duongdb/ClassifyFaceConditions/MakeOurData

maindir=/data/duongdb/WS22qOther_08102021

col_name_for_group='AgeGrouping'

new_img_data_path=$maindir/TrimImg/
mkdir $new_img_data_path

new_qualtric_data_path=$maindir/QualtricImg
mkdir $new_qualtric_data_path

which_survey=WS # ! pick the survey 
for disease in WS Controls # 22q11DS WS 
do 

  test_csv_name=$maindir/$disease'_test_img_list_survey_'$which_survey'.csv'
  rm -f $test_csv_name # ! start from fresh

  for label in 2y adolescence youngchild olderadult youngadult
  do

    suffix=$disease$label

    for csv in $disease'_early' $disease'_inter' $disease'_late'
    do 
    src_img_data_path=$maindir/$csv/TrimWhiteSpaceNoBorder
    csv_path=$maindir/$csv.csv
    prefix=$csv
    python3 ExtractImgInGroupFromCsv.py --src_img_data_path $src_img_data_path --new_img_data_path $new_img_data_path --csv_path $csv_path --col_name_for_group $col_name_for_group --prefix $prefix --suffix $suffix --new_qualtric_data_path $new_qualtric_data_path --which_survey $which_survey --test_csv_name $test_csv_name
    done

  done 
  
done 
cd $maindir


# ! concat WS and 22q test csv if we want to train joint model 
WS=$maindir/'WS_test_img_list_survey_WS.csv'
q22=$maindir/'22q11DS_test_img_list_survey_22q11DS.csv'
c1=$maindir/'Controls_test_img_list_survey_WS.csv'
c2=$maindir/'Controls_test_img_list_survey_22q11DS.csv'
cat $WS $q22 $c1 $c2 > $maindir/WS_22q11DS_controls_test_img_list.csv 
cd $maindir


# ! remove duplicate IN PYTHON
import pandas as pd 
df = pd.read_csv('WS_22q11DS_controls_test_img_list.csv',header=None)
df[df.duplicated()]
df = df.drop_duplicates()
df.to_csv('WS_22q11DS_controls_test_img_list.csv',index=None)

