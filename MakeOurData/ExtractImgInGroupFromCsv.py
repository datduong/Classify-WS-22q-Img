
# ! use csv to pull images. 

import os,sys,re,pickle
import pandas as pd 
import numpy as np 

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--src_img_data_path", type=str, default=None)
parser.add_argument("--new_img_data_path", type=str, default=None)
parser.add_argument("--new_qualtric_data_path", type=str, default=None)
parser.add_argument("--csv_path", type=str, default=None)
parser.add_argument("--col_name_for_group", type=str, default=None)
parser.add_argument("--prefix", type=str, default=None)
parser.add_argument("--suffix", type=str, default=None)
parser.add_argument("--which_survey", type=str, default=None)
parser.add_argument("--test_csv_name", type=str, default=None)

args = parser.parse_args()

if args.new_qualtric_data_path is not None: 
  try: 
    os.mkdir(os.path.join(args.new_qualtric_data_path,args.which_survey))
  except:
    pass

# !  
print ('\n\n'+args.csv_path+'\n\n')
       
csv = pd.read_csv (args.csv_path,dtype=str) # ! excel converted to csv 
csv = csv.fillna('')

# ! handle some variation in spelling
temp_ = [re.sub(' ','',l) for l in csv[args.col_name_for_group].values]
csv[args.col_name_for_group] = temp_
# ['<2y', 'adolecence', 'adolescence', 'olderadult', 'youngadult', 'youngchid', 'youngchild']
map_misspell = {'<2y':'2y', 'adolecence':'adolescence', 'youngchid':'youngchild' }
csv = csv.replace({args.col_name_for_group: map_misspell})

print ( set ( csv[args.col_name_for_group].values ) )
print (csv)

fout = open(args.test_csv_name,'a') # ! write+append out the test images names 

for index,row in csv.iterrows():
  r = row['Slide'] 
  if len(r) == 0: 
    print (row)
    continue # skip if no slide
  l = row[args.col_name_for_group]
  if len(l) == 0: 
    continue # skip blank age group
  # get test images
  t = row[args.which_survey+'Test'].strip() # ! can have space ? 
  #
  if l in args.suffix: # if label is found in suffix input
    new_name = args.prefix + r + '_' + args.suffix + '.png'
    os.system( 'scp ' + os.path.join(args.src_img_data_path,'Slide'+r+'.png') + ' ' + os.path.join(args.new_img_data_path,new_name) ) 
    # ! scp the test set to its own qualtric folder 
    if args.new_qualtric_data_path is not None: 
      if (t == 'x') or (t=='X') : # empty is in train set 
        os.system( 'scp ' + os.path.join(args.src_img_data_path,'Slide'+r+'.png') + ' ' + os.path.join(args.new_qualtric_data_path,args.which_survey,new_name) ) 
        fout.write( new_name + '\n' )


# ! 
fout.close()     



# ! make labels 

# labelset = []
# for d in ['WS','Normal','Controls','22q11DS']: 
#   for a in ['2y', 'adolescence', 'olderadult', 'youngadult', 'youngchild']:
#     labelset = labelset + [d+a]

# # 
# labelset = sorted(list(set(labelset)))
# print( ",".join(labelset) )


# '22q11DS2y,22q11DSadolescence,22q11DSolderadult,22q11DSyoungadult,22q11DSyoungchild,Controls2y,Controlsadolescence,Controlsolderadult,Controlsyoungadult,Controlsyoungchild,Normal2y,Normaladolescence,Normalolderadult,Normalyoungadult,Normalyoungchild,WS2y,WSadolescence,WSolderadult,WSyoungadult,WSyoungchild'

# '22q11DS2y,22q11DSadolescence,22q11DSolderadult,22q11DSyoungadult,22q11DSyoungchild,Controls2y,Controlsadolescence,Controlsolderadult,Controlsyoungadult,Controlsyoungchild,WS2y,WSadolescence,WSolderadult,WSyoungadult,WSyoungchild'
