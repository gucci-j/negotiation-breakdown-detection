import numpy as np
import torch
import torch.nn as nn
from scipy import interp

from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, fbeta_score, precision_score, recall_score, accuracy_score

import matplotlib.pyplot as plt
import seaborn as sns

def compute_accuracy(preds: np.array, y: np.array, sigmoid:bool=False):
    if sigmoid:
        preds = (preds >= 0.5).astype(int)
    return accuracy_score(y, preds)


def compute_ap(preds: np.array, y: np.array):
    return average_precision_score(y, preds)


def compute_f1_prec_rec(preds: np.array, y: np.array, 
                        data_type='binary', pos_label=True,
                        sigmoid=False):
    if data_type == 'binary':
        if sigmoid is True:
            if pos_label is False:
                preds = (preds < 0.5).astype(int)
                y = (y == 0).astype(int)
            else:
                preds = (preds >= 0.5).astype(int)
        elif pos_label is False:
            y = (y == 0).astype(int)
            preds = (preds == 0).astype(int)

        f1 = f1_score(y, preds, zero_division=0)
        fbeta = fbeta_score(y, preds, beta=2.0, zero_division=0)
        prec = precision_score(y, preds, zero_division=0)
        rec = recall_score(y, preds, zero_division=0)
        return f1, fbeta, prec, rec

    else: # multi-class
        preds = preds.astype(int)
        acc = accuracy_score(y, preds)
        f1 = f1_score(y, preds, average='macro')
        fbeta = fbeta_score(y, preds, beta=2.0, average='macro')
        prec = precision_score(y, preds, average='macro')
        rec = recall_score(y, preds, average='macro')
        return acc, f1, fbeta, prec, rec


class ConfusionMatrix(object):
    """Draw & Compute Confusion Matrix"""
    def __init__(self, data_type='binary'):
        plt.figure(1)
        sns.set_style()
        sns.set_context("paper")
        self.cm_data_list = []
        self.data_type = data_type
        
    
    def draw(self, preds: np.array, y: np.array, num_fold:int, run_start_time: str, base_path: str):
        # comppute cm
        if self.data_type == 'binary':
            rounded_preds = (preds >= 0.5).astype(int)
            cm_data = confusion_matrix(y, rounded_preds) # In the case of binary: (tn, fp, fn, tp)
        else:
            cm_data = confusion_matrix(y, preds.astype(int))
        cm_data = cm_data.astype(float) / cm_data.sum(axis=1)[:, np.newaxis]
        self.cm_data_list.append(cm_data.flatten())

        # plot
        plt.figure(1)
        sns.heatmap(cm_data, annot=True, cmap='Blues')
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Normalised confusion matrix")
        plt.savefig('{}/fig/{}/cm_{}.png'.format(base_path, run_start_time, num_fold), bbox_inches="tight", pad_inches=0.1)
        plt.clf()


    def finishing(self, logger):
        self.cm_data_list = np.array(self.cm_data_list)
        logger.info(f'cm breakdown: {self.cm_data_list}')
        std_data = [np.std(self.cm_data_list[:, index]) for index in range(self.cm_data_list.shape[1])]
        cm_data = [np.mean(self.cm_data_list[:, index]) for index in range(self.cm_data_list.shape[1])]
        logger.info(f'std: {std_data} | cm: {cm_data}')


class ROC(object):
    """Draw & Compute ROC-AUC"""
    def __init__(self, data_type='binary'):
        self.tpr_list = []
        self.roc_auc_list = []
        self.mean_fpr = np.linspace(0, 1, 100)
        self.data_type = data_type
        plt.figure(2)
        sns.set_style()
        sns.set_context("paper")
        

    def draw(self, preds: np.array, y: np.array, num_fold:int):
        if self.data_type == 'multi-class':
            # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score
            auc = roc_auc_score(y, preds, average='macro', multi_class='ovo')
            self.roc_auc_list.append(auc)
            return None

        fpr, tpr, _ = roc_curve(y, preds)
        auc = roc_auc_score(y, preds)

        self.tpr_list.append(interp(self.mean_fpr, fpr, tpr))
        self.tpr_list[-1][0] = 0.0
        self.roc_auc_list.append(auc)

        # plot
        plt.figure(2)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
             label='ROC fold %d (AUC = %0.2f)' % (num_fold, auc))
        
        
    def finishing(self, run_start_time: str, base_path: str, logger):
        if self.data_type == 'multi-class':
            mean_auc = np.std(self.roc_auc_list)
            std_auc = np.std(self.roc_auc_list)
            logger.info(f'Mean AUC: {mean_auc} | Std AUC: {std_auc}')
            return None

        plt.figure(2)
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, label='Chance', color='r', alpha=.8)
        
        mean_tpr = np.mean(self.tpr_list, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(self.mean_fpr, mean_tpr)
        std_auc = np.std(self.roc_auc_list)
        plt.plot(self.mean_fpr, mean_tpr, color='b',
            label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), lw=2, alpha=.8)
        
        std_tpr = np.std(self.tpr_list, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(self.mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
            label=r'$\pm$ 1 std. dev.')

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.grid(True)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title("ROC curve")
        plt.legend(loc="lower right")
        plt.savefig('{}/fig/{}/roc.png'.format(base_path, run_start_time), bbox_inches="tight", pad_inches=0.1)
    
        
    def get_auc_list(self):
        return self.roc_auc_list


class PrecRecall(object):
    """Draw & Compute PR-AUC"""
    def __init__(self):
        self.prec_list = [] # y-coordinates
        self.pr_auc_list = []
        self.mean_rec = np.linspace(0.0, 1.0, 200) # x-coordinates
        plt.figure(3)
        sns.set_style()
        sns.set_context("paper")
        

    def draw(self, preds: np.array, y: np.array, num_fold:int):
        prec, rec, _ = precision_recall_curve(y, preds)
        # pr_auc = auc(rec, prec)
        ap = average_precision_score(y, preds)

        self.prec_list.append(interp(self.mean_rec, rec[::-1], prec[::-1]))
        self.prec_list[-1][0] = 1.0 # init
        # self.pr_auc_list.append(pr_auc)
        self.pr_auc_list.append(ap)

        # plot
        plt.figure(3)
        plt.plot(rec, prec, lw=1, alpha=0.3,
             label='Fold %d (AP = %0.2f)' % (num_fold, ap))
        
        
    def finishing(self, run_start_time: str, base_path: str):
        plt.figure(3)
        mean_prec = np.mean(self.prec_list, axis=0)
        mean_prec[-1] = 0.0
        # mean_pr_auc = auc(self.mean_rec, mean_prec)
        mean_pr_auc = np.mean(self.pr_auc_list)
        std_auc = np.std(self.pr_auc_list)
        plt.plot(self.mean_rec, mean_prec, color='b',
            label=r'Mean AP (AP = %0.2f $\pm$ %0.2f)' % (mean_pr_auc, std_auc), lw=2, alpha=.8)
        
        std_prec = np.std(self.prec_list, axis=0)
        prec_upper = np.minimum(mean_prec + std_prec, 1)
        prec_lower = np.maximum(mean_prec - std_prec, 0)
        plt.fill_between(self.mean_rec, prec_lower, prec_upper, color='grey', alpha=.2,
            label=r'$\pm$ 1 std. dev.')

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.grid(True)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("PR curve")
        plt.legend(loc="lower left")
        plt.savefig('{}/fig/{}/pr.png'.format(base_path, run_start_time), bbox_inches="tight", pad_inches=0.1)
    
        
    def get_auc_list(self):
        return self.pr_auc_list