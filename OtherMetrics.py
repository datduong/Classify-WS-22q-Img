import os,sys,re,pickle 
import seaborn as sn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, balanced_accuracy_score

import torch
from ignite.metrics import Accuracy, Precision, Recall, Fbeta


def plot_confusion_matrix (prediction, y_true, diagnosis2idx, title, our_index, just_our_label=False): 

    labels = sorted ( list ( diagnosis2idx.keys() ) )
    output = confusion_matrix(y_true, prediction.argmax(axis=1)) ## ! take max as our best prediction
    sum_of_rows = output.sum(axis=1)
    normalized_array = output / sum_of_rows[:, np.newaxis] * 100 ## put back on 100 scale
    normalized_array = np.round (normalized_array,5)

    print (normalized_array.shape)
    print (normalized_array)

    if not just_our_label: 
        df_cm = pd.DataFrame(normalized_array, 
                            index = [i for i in labels],
                            columns = [i for i in labels]).astype(float).round(3)
        plt.figure(figsize=(12,12))
        sn.set(font_scale=1.4) # for label size
        sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt=".1f", cbar=False) # font size
        plt.savefig(title+'.png')

    # ! do only our conditions 
    labels = [ lab if lab!='EverythingElse' else 'Other' for lab in labels]
    labels = [ labels[i] for i in our_index ]
    normalized_array = normalized_array[our_index,:][:,our_index] ## get our conditions
    df_cm = pd.DataFrame(normalized_array, 
                         index = [i for i in labels],
                         columns = [i for i in labels]).astype(float).round(3)
    plt.figure(figsize=(12,12))
    sn.set(font_scale=1.4) # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt=".1f", cbar=False) # font size
    plt.savefig(title+'_ourlabels.png')
    pickle.dump(normalized_array,open(title+'_ourlabels.np','wb')) # later we will average over many folds


def plot_confusion_matrix_manual (prediction, y_true, diagnosis2idx, title, our_index, figsize=16): 

    output = confusion_matrix(y_true, prediction.argmax(axis=1)) ## ! take max as our best prediction
    sum_of_rows = output.sum(axis=1)
    normalized_array = output / sum_of_rows[:, np.newaxis] * 100 ## put back on 100 scale
    normalized_array = np.round (normalized_array,5)
    print (normalized_array)

    # ! do only our conditions 
    temp = list ( set(prediction.argmax(axis=1)) )
    our_index = list ( set ( our_index + temp ) ) # combine the label index
    labels = sorted ( [ k for k,val in diagnosis2idx.items() if val in our_index ] ) # ! we misclassify something in our dataset as ISIC condition
    labels = [ lab if lab!='EverythingElse' else 'Other' for lab in labels]
    print (labels)
    
    df_cm = pd.DataFrame(normalized_array, 
                         index = [i for i in labels],
                         columns = [i for i in labels]).astype(float).round(3)
    if len(labels) <= 6: 
        figsize = 6
    elif len(labels) >= 15: 
        figsize = 20
    plt.figure(figsize=(figsize,figsize))
    sn.set_context("talk", font_scale=1)
    # sn.set(font_scale=1.4) # for label size
    ax = plt.axes()
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt=".1f", cbar=False, ax=ax) # font size
    ax.set_title('Confusion matrix')
    plt.xticks(rotation=45)
    plt.savefig(title+'_ourlabels.png')
    pickle.dump(normalized_array,open(title+'_ourlabels.np','wb')) # later we will average over many folds


def compute_balanced_accuracy_score (prediction,target): 
    return balanced_accuracy_score (target, prediction.argmax(axis=1))


def recall_at_k(yhat_raw, y, k):
    #num true labels in top k predictions / num true labels
    sortd = np.argsort(yhat_raw)[:,::-1]
    topk = sortd[:,:k]

    #get recall at k for each example
    vals = []
    for i, tk in enumerate(topk):
        num_true_in_top_k = y[i,tk].sum()
        denom = y[i,:].sum()
        vals.append(num_true_in_top_k / float(denom))

    vals = np.array(vals)
    vals[np.isnan(vals)] = 0.

    return np.mean(vals)


def precision_at_k(yhat_raw, y, k):
    #num true labels in top k predictions / k
    sortd = np.argsort(yhat_raw)[:,::-1]
    topk = sortd[:,:k]

    #get precision at k for each example
    vals = []
    for i, tk in enumerate(topk):
        if len(tk) > 0:
            num_true_in_top_k = y[i,tk].sum()
            denom = len(tk)
            vals.append(num_true_in_top_k / float(denom))

    return np.mean(vals)


def topk_accuracy(yhat_raw, target, topk=(1,)):
    metrics = {} # dict
    for k_i in topk:
        rec_at_k = recall_at_k(yhat_raw, target, k_i)
        metrics['rec_at_%d' % k_i] = rec_at_k
        prec_at_k = precision_at_k(yhat_raw, target, k_i)
        metrics['prec_at_%d' % k_i] = prec_at_k
        metrics['f1_at_%d' % k_i] = 2*(prec_at_k*rec_at_k)/(prec_at_k+rec_at_k)
    return metrics


# ! 
class MetricsGroup: # https://discuss.pytorch.org/t/combining-metrics-in-ignite-metrics/81688/2
        
    def __init__(self, metrics_dict):
        self.metrics = metrics_dict
        
    def update(self, output):
        for name, metric in self.metrics.items():
            metric.update(output)
            
    def compute(self):
        output = {}
        for name, metric in self.metrics.items():
            output[name] = metric.compute()
        return output


class LabelwiseAccuracy(Accuracy):
    def __init__(self, output_transform=lambda x: x):
        self._num_correct = None
        self._num_examples = None
        super(LabelwiseAccuracy, self).__init__(output_transform=output_transform)

    def reset(self):
        self._num_correct = None
        self._num_examples = 0
        super(LabelwiseAccuracy, self).reset()

    def update(self, output):

        # y_pred, y = self._check_shape(output)
        self._check_shape(output)
        y_pred, y = output
        
        self._check_type((y_pred, y))

        num_classes = y_pred.size(1)
        last_dim = y_pred.ndimension()
        y_pred = torch.transpose(y_pred, 1, last_dim - 1).reshape(-1, num_classes)
        y = torch.transpose(y, 1, last_dim - 1).reshape(-1, num_classes)
        correct_exact = torch.all(y == y_pred.type_as(y), dim=-1)  # Sample-wise
        correct_elementwise = torch.sum(y == y_pred.type_as(y), dim=0)

        if self._num_correct is not None:
            self._num_correct = torch.add(self._num_correct,
                                                    correct_elementwise)
        else:
            self._num_correct = correct_elementwise
        self._num_examples += correct_exact.shape[0]

    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError('Accuracy must have at least one example before it can be computed.')
        return self._num_correct.type(torch.float) / self._num_examples

