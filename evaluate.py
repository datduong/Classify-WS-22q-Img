import os
import time
import random
import argparse
import numpy as np
import pandas as pd
import cv2
import PIL.Image
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from util import GradualWarmupSchedulerV2

import apex 
from apex import amp
from dataset import get_df, get_transforms, DatasetFromCsv
from models import Effnet_Face_Conditions, Resnest_Face_Conditions, Seresnext_Face_Conditions
from train import get_trans

from captum.attr import IntegratedGradients
from captum.attr import GradientShap
from captum.attr import Occlusion
from captum.attr import NoiseTunnel

from copy import deepcopy 

import geffnet

import OtherMetrics
from SeeAttribution import GetAttribution

from SoftLabelLoss import cross_entropy_with_probs

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kernel-type', type=str, required=True)
    parser.add_argument('--image-csv', type=str, default=None)
    parser.add_argument('--label-upweigh', type=str, default=None)
    parser.add_argument('--data-folder', type=int)
    parser.add_argument('--image-size', type=int, required=True)
    parser.add_argument('--enet-type', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=32)
    parser.add_argument('--out-dim', type=int, default=9)
    parser.add_argument('--use-amp', action='store_true')
    parser.add_argument('--use-meta', action='store_true')
    parser.add_argument('--model-dir', type=str, default='./weights')
    parser.add_argument('--log-dir', type=str, default='./logs')
    parser.add_argument('--oof-dir', type=str, default='./oofs')
    parser.add_argument('--eval', type=str, choices=['best', 'best_20', 'final','best_all','ourlabel'], default="ourlabel") # "best")
    parser.add_argument('--CUDA_VISIBLE_DEVICES', type=str, default=None) # '0'
    parser.add_argument('--n-meta-dim', type=str, default='512,128')
    parser.add_argument('--fold', type=str, default='0,1,2,3,4')
    
    # ! added
    parser.add_argument('--dropout', type=float, default=0.5) # doesn't get used
    parser.add_argument('--n-test', type=int, default=1, help='how many times do we flip images, 1=>no_flip, max=8')
    parser.add_argument('--attribution_keyword', type=str, default=None) 
    parser.add_argument('--outlier_perc', type=int, default=10, help='show fraction of high contributing pixel, default 10%')
    parser.add_argument('--img-map-file', type=str, default='train.csv')
    parser.add_argument('--do_test', action='store_true', default=False)
    parser.add_argument('--do_test_manual_name', type=str, default=None)
    parser.add_argument('--soft_label', action='store_true', default=False)
    parser.add_argument('--ret_vec_rep', action='store_true', default=False)
    parser.add_argument('--num_fc', type=int, default=0)
    parser.add_argument('--attribution_model', type=str, default='integrated_gradient')
    parser.add_argument('--noise_tunnel', action='store_true', default=False)
    parser.add_argument('--attr_np_as_vec', action='store_true', default=False)
    
    args = parser.parse_args()
    return args



def val_epoch(model, loader, our_label_index, diagnosis2idx, n_test=1, get_output=True, fold=None, args=None, nn_model_on_attr_vec=None):

    model.eval()
    val_loss = []
    LOGITS = []
    PROBS = []
    TARGETS = []
    VEC_REP = []

    if args.attribution_keyword is not None: # ! do attribution
        n_test = 1 # ! test original image, not flipping
        print ('\nwill do attribution_model\n')
        if args.attribution_model == 'Occlusion': 
            attribution_model = Occlusion(model)
        else: 
            attribution_model = IntegratedGradients(model) 
            if args.noise_tunnel: 
                attribution_model = NoiseTunnel(attribution_model)

    with torch.no_grad():
        for (data, target, target_soft, path, data_resize) in tqdm(loader): # @path is needed to select our labels

            if args.attribution_keyword is not None: 
                our_label_index = [path.index(j) for j in path if args.attribution_keyword in j] # ! get images with @attribution_keyword
                if len (our_label_index) == 0 :
                    continue # skip

            if args.use_meta:
                data, meta = data
                data, meta, target = data.to(device), meta.to(device), target.to(device)    
            else:
                data, target = data.to(device), target.to(device)
                meta = None

            #
            logits = torch.zeros((data.shape[0], args.out_dim)).to(device)
            probs = torch.zeros((data.shape[0], args.out_dim)).to(device)
            for I in range(n_test):
                l = model(get_trans(data, I), x_meta=meta)
                if isinstance(l, tuple) : 
                    l, vec_rep = l
                logits += l
                probs += l.softmax(1)
            logits /= n_test
            probs /= n_test

            LOGITS.append(logits.detach().cpu())
            PROBS.append(probs.detach().cpu()) # ! @PROBS is shape=(1, obs, labelsize)
            TARGETS.append(target.detach().cpu())

            if args.soft_label: 
                loss = criterion(logits, target_soft.to(device))
            else:
                loss = criterion(logits, target)
            
            val_loss.append(loss.detach().cpu().numpy())

            # ! keep vec if we want to plot t-sne
            if args.ret_vec_rep: 
                VEC_REP.append(vec_rep.detach().cpu()) 
            
            # ! do attribution here. call IntegratedGradient, or some other approaches
            if args.attribution_keyword is not None: 
                our_label_index = [path.index(j) for j in path if args.attribution_keyword in j.split('/')[-1]] # @attribution_keyword in file name
                if len (our_label_index) > 0 :
                    temp = GetAttribution.GetAttributionPlot (  data[our_label_index].detach().cpu(), 
                                                                probs[our_label_index].detach().cpu(), 
                                                                np.array(path)[our_label_index], 
                                                                data_resize[our_label_index], 
                                                                attribution_model, 
                                                                fold=fold,
                                                                true_label_index=target[our_label_index].detach().cpu(),
                                                                args=args, 
                                                                nn_model=nn_model_on_attr_vec)

    # ! end eval loop
    if args.attribution_keyword is not None: 
        exit() # ! just do attribution
        
    val_loss = np.mean(val_loss)
    LOGITS = torch.cat(LOGITS).numpy()
    PROBS = torch.cat(PROBS).numpy() # ! @PROBS is shape=(1, test_size, num_label)
    TARGETS = torch.cat(TARGETS).numpy()

    # ! compute acc for this fold. 
    
    acc = (PROBS.argmax(1) == TARGETS).mean() * 100.
    bal_acc = OtherMetrics.compute_balanced_accuracy_score(PROBS, TARGETS) # ! weighted accuracy
    
    if args.do_test_manual_name is None: 
        for key,idx in diagnosis2idx.items(): # ! AUC
            if idx in our_label_index: 
                auc = roc_auc_score((TARGETS == idx).astype(float), PROBS[:, idx]) 
                print (time.ctime() + ' ' + f'Fold {fold}, {key} auc: {auc:.5f}')
        # ! global confusion matrix
        OtherMetrics.plot_confusion_matrix( PROBS, TARGETS, diagnosis2idx, os.path.join(args.log_dir,'confusion_matrix_fold'+str(fold) ), our_label_index )

    # ! return vec rep
    if args.ret_vec_rep: 
       VEC_REP = torch.cat(VEC_REP).numpy() # ! @VEC_REP is shape=(1, test_size, num_label) should be same as @PROB
        
    return LOGITS, PROBS, VEC_REP, val_loss, acc, bal_acc



def main():

    # ! load pretrained to get embedding of attribution np.matrix ? 
    nn_model_on_attr_vec = None
    if args.attr_np_as_vec: 
        print ('load pre-trained imagenet model for attribution')
        nn_model_on_attr_vec = geffnet.create_model(args.enet_type, pretrained=True)
        nn_model_on_attr_vec.classifier = nn.Identity()
        nn_model_on_attr_vec.eval() # ! turn on eval mode to avoid random dropout during eval

    df, diagnosis2idx, our_label_index = get_df(args.image_csv, soft_label=args.soft_label)

    _, transforms_val, transforms_resize = get_transforms(args.image_size)

    print ('our_label_index {}'.format(our_label_index))
   
    LOGITS = []
    PROBS = []
    VEC = []
    
    for fold in [int(i) for i in args.fold.split(',')]:

        if args.do_test and (args.do_test_manual_name is None): 
            print ('\ntesting on fold id=5 using data trained without fold {}\n'.format(fold))
            df_valid = df[df['fold'] == 5] # ! eval on our own test set
        elif args.do_test_manual_name is not None: 
            print ('\nmanual testing')
            df_valid = df # ! test on whole set of GAN images ?? or test back on train set ???
        else: 
            df_valid = df[df['fold'] == fold] # ! eval on the left-out fold

        print ('eval data size {}'.format(df_valid.shape))

        dataset_valid = DatasetFromCsv(df_valid, 'valid', None, transform=transforms_val, transform_resize=transforms_resize, soft_label=args.soft_label)
        valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

        print ('len of valid pytorch dataset {}'.format(len(dataset_valid)))
        
        # ! load model         
        model_file = os.path.join(args.model_dir, f'{args.kernel_type}_{args.eval}_fold{fold}.pth')
        print ('\nmodel_file {}\n'.format(model_file))
        
        model = ModelClass(
            args.enet_type,
            n_meta_features=0,
            n_meta_dim=[int(nd) for nd in args.n_meta_dim.split(',')],
            out_dim=args.out_dim,
            pretrained=True, 
            args=args
        )
        model = model.to(device)

        try:  # single GPU model_file
            if args.CUDA_VISIBLE_DEVICES is None:
                model.load_state_dict(torch.load(model_file), strict=True, map_location=torch.device('cpu')) # ! avoid error in loading model trained on GPU
            else: 
                model.load_state_dict(torch.load(model_file), strict=True) 
        except:  # multi GPU model_file
            if args.CUDA_VISIBLE_DEVICES is None:
                state_dict = torch.load(model_file, map_location=torch.device('cpu'))
            else: 
                state_dict = torch.load(model_file)
            state_dict = {k[7:] if k.startswith('module.') else k: state_dict[k] for k in state_dict.keys()}
            model.load_state_dict(state_dict, strict=True)
        
        if DP : 
            model = torch.nn.DataParallel(model)

        model.eval()

        this_LOGITS, this_PROBS, this_VEC, val_loss, acc, bal_acc = val_epoch(model, valid_loader, our_label_index, diagnosis2idx, n_test=args.n_test, get_output=True, fold=fold, args=args, nn_model_on_attr_vec=nn_model_on_attr_vec)
        
        LOGITS.append(this_LOGITS)
        PROBS.append(this_PROBS)
        if args.ret_vec_rep: 
            VEC.append(this_VEC)

        # print 
        content = time.ctime() + ' ' + f'Fold {fold}, valid loss: {(val_loss):.5f}, acc: {(acc):.4f}, bal_acc {(bal_acc):.6f}'
        print(content)
        with open(os.path.join(args.log_dir, f'log_{args.kernel_type}_eval.txt'), 'a') as appender:
            appender.write(content + '\n')

        # ! merge data frame
        print ('PROB output size {}'.format(PROBS[0].shape))
        prob_df = pd.DataFrame( PROBS[0], columns=np.arange(args.out_dim) ) # @PROBS is shape=(1, obs, labelsize)
        prob_df = prob_df.reset_index(drop=True) # has to do this to concat right
        df_valid_temp = df_valid.copy()
        df_valid_temp = df_valid_temp.reset_index(drop=True)
        assert df_valid_temp.shape[0] == prob_df.shape[0]
        df_valid_prob = pd.concat([df_valid_temp, prob_df], axis=1) # ! append col wise
        
        # ! add in vec rep 
        if args.ret_vec_rep: 
            VEC = pd.DataFrame( VEC[0], columns=['X'+str(i) for i in np.arange(VEC[0].shape[1])] )
            VEC = VEC.reset_index(drop=True)
            df_valid_prob = pd.concat([df_valid_prob, VEC], axis=1)
            
        log_file_name = 'eval_fold_'+str(fold)+'.csv'
        if args.do_test:
            log_file_name = 'test_on_fold_5_from_fold'+str(fold)+'.csv' # ! special fold 5, which is our test set
        if args.do_test_manual_name is not None:
            log_file_name = args.do_test_manual_name+'_from_fold'+str(fold)+'.csv' # ! testing on some other test set 
        df_valid_prob.to_csv(os.path.join(args.log_dir, log_file_name),index=False)

    # end folds  



if __name__ == '__main__':

    args = parse_args()
    os.makedirs(args.oof_dir, exist_ok=True)

    if args.enet_type == 'resnest101':
        ModelClass = Resnest_Face_Conditions
    elif args.enet_type == 'seresnext101':
        ModelClass = Seresnext_Face_Conditions
    elif 'efficientnet' in args.enet_type:
        ModelClass = Effnet_Face_Conditions
    else:
        raise NotImplementedError()

    if args.CUDA_VISIBLE_DEVICES is not None: 
        os.environ['CUDA_VISIBLE_DEVICES'] = args.CUDA_VISIBLE_DEVICES
        DP = len(os.environ['CUDA_VISIBLE_DEVICES']) > 1
        device = torch.device('cuda')
    else: 
        DP = False  
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not args.soft_label:     
        criterion = nn.CrossEntropyLoss()
    else: 
        criterion = cross_entropy_with_probs

    main()

