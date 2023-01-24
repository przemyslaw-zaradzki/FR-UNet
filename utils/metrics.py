import numpy as np
import torch
import cv2
from matplotlib import pyplot as plt
from PIL import Image

from sklearn.metrics import (confusion_matrix, f1_score,
                             jaccard_similarity_score, precision_recall_curve,
                             roc_auc_score, roc_curve)


class AverageMeter(object):
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = np.multiply(val, weight)
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum = np.add(self.sum, np.multiply(val, weight))
        self.count = self.count + weight
        self.avg = self.sum / self.count

    @property
    def value(self):
        return np.round(self.val, 4)

    @property
    def average(self):
        return np.round(self.avg, 4)


def get_metrics(predict, target, threshold=None, predict_b=None):
    predict = torch.sigmoid(predict).cpu().detach().numpy().flatten()
    if predict_b is not None:
        predict_b = predict_b.flatten()
    else:
        predict_b = np.where(predict >= threshold, 1, 0)
    if torch.is_tensor(target):
        target = target.cpu().detach().numpy().flatten()
    else:
        target = target.flatten()
    tp = (predict_b * target).sum()
    tn = ((1 - predict_b) * (1 - target)).sum()
    fp = ((1 - target) * predict_b).sum()
    fn = ((1 - predict_b) * target).sum()
    target = np.where(target >= 0.5, 1, 0)
    try:
        auc = roc_auc_score(target, predict)
    except:
        auc = 0.0
    acc = (tp + tn) / (tp + fp + fn + tn)
    pre = tp / (tp + fp)
    sen = tp / (tp + fn)
    spe = tn / (tn + fp)
    iou = tp / (tp + fp + fn)
    f1 = 2 * pre * sen / (pre + sen)
    return {
        "AUC": np.round(auc, 4),
        "F1": np.round(f1, 4),
        "Acc": np.round(acc, 4),
        "Sen": np.round(sen, 4),
        "Spe": np.round(spe, 4),
        "pre": np.round(pre, 4),
        "IOU": np.round(iou, 4),
    }


def count_connect_component(predict, target, threshold=None, connectivity=8):
    if threshold != None:
        predict = torch.sigmoid(predict).cpu().detach().numpy()
        predict = np.where(predict >= threshold, 1, 0)
    if torch.is_tensor(target):
        target = target.cpu().detach().numpy()
    pre_n, _, _, _ = cv2.connectedComponentsWithStats(np.asarray(
        predict, dtype=np.uint8)*255, connectivity=connectivity)
    gt_n, _, _, _ = cv2.connectedComponentsWithStats(np.asarray(
        target, dtype=np.uint8)*255, connectivity=connectivity)
    return pre_n/gt_n


def evaluate(gt, pre, dataset_name, experiment_date,  DTI=False):
    if DTI:
        predictions = pre.cpu().detach().numpy()
    else:
        predictions = torch.sigmoid(pre).cpu().detach().numpy()

    for idx in range(predictions.shape[0]):
        prediction = 255*predictions[idx,:,:]
        prediction = prediction.astype(np.uint8)
        prediction = Image.fromarray(prediction)
        if DTI:
            prediction.save(f"saved/FR_UNet/{dataset_name}/{experiment_date}/{experiment_date}_test_DTI_{idx}.png")
        else:
            prediction.save(f"saved/FR_UNet/{dataset_name}/{experiment_date}/{experiment_date}_test_{idx}.png")

    #====== Evaluate the results
    print("\n\n========  Evaluate the results =======================")
    y_scores = predictions.flatten()
    y_true = gt.cpu().detach().numpy()
    y_true = y_true.flatten()

    #Area under the ROC curve
    y_true = np.round(y_true).astype('uint8')
    fpr, tpr, thresholds = roc_curve((y_true), y_scores)
    AUC_ROC = roc_auc_score(y_true, y_scores)
    # test_integral = np.trapz(tpr,fpr) #trapz is numpy integration
    print("\nArea under the ROC curve: " +str(AUC_ROC))
    roc_curve_plt = plt.figure()
    plt.plot(fpr,tpr,'-',label='Area Under the Curve (AUC = %0.4f)' % AUC_ROC)
    plt.title('ROC curve')
    plt.xlabel("FPR (False Positive Rate)")
    plt.ylabel("TPR (True Positive Rate)")
    plt.legend(loc="lower right")
    plt.savefig(f"saved/FR_UNet/{dataset_name}/{experiment_date}/ROC.png")

    #Precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    precision = np.fliplr([precision])[0]  #so the array is increasing (you won't get negative AUC)
    recall = np.fliplr([recall])[0]  #so the array is increasing (you won't get negative AUC)
    AUC_prec_rec = np.trapz(precision,recall)
    print("\nArea under Precision-Recall curve: " +str(AUC_prec_rec))
    prec_rec_curve = plt.figure()
    plt.plot(recall,precision,'-',label='Area Under the Curve (AUC = %0.4f)' % AUC_prec_rec)
    plt.title('Precision - Recall curve')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="lower right")
    plt.savefig(f"saved/FR_UNet/{dataset_name}/{experiment_date}/Precision_recall.png")

    #Confusion matrix
    threshold_confusion = 0.5
    print("\nConfusion matrix:  Custom threshold (for positive) of " +str(threshold_confusion))
    y_pred = np.empty((y_scores.shape[0]))
    for i in range(y_scores.shape[0]):
        if y_scores[i]>=threshold_confusion:
            y_pred[i]=1
        else:
            y_pred[i]=0
    confusion = confusion_matrix(y_true, y_pred)
    print(confusion)
    accuracy = 0
    if float(np.sum(confusion))!=0:
        accuracy = float(confusion[0,0]+confusion[1,1])/float(np.sum(confusion))
    print("Global Accuracy: " +str(accuracy))
    specificity = 0
    if float(confusion[0,0]+confusion[0,1])!=0:
        specificity = float(confusion[0,0])/float(confusion[0,0]+confusion[0,1])
    print("Specificity: " +str(specificity))
    sensitivity = 0
    if float(confusion[1,1]+confusion[1,0])!=0:
        sensitivity = float(confusion[1,1])/float(confusion[1,1]+confusion[1,0])
    print("Sensitivity: " +str(sensitivity))
    precision = 0
    if float(confusion[1,1]+confusion[0,1])!=0:
        precision = float(confusion[1,1])/float(confusion[1,1]+confusion[0,1])
    print("Precision: " +str(precision))

    #Jaccard similarity index
    jaccard_index = jaccard_similarity_score(y_true, y_pred, normalize=True)
    print("\nJaccard similarity score: " +str(jaccard_index))

    #F1 score
    F1_score = f1_score(y_true, y_pred, labels=None, average='binary', sample_weight=None)
    print("\nF1 score (F-measure): " +str(F1_score))

    #Save the results
    if DTI:
        file_perf = open(f"saved/FR_UNet/{dataset_name}/{experiment_date}/performances_DTI.txt", 'w')
    else:
        file_perf = open(f"saved/FR_UNet/{dataset_name}/{experiment_date}/performances.txt", 'w')
    file_perf.write("Area under the ROC curve: "+str(AUC_ROC)
                    + "\nArea under Precision-Recall curve: " +str(AUC_prec_rec)
                    + "\nJaccard similarity score: " +str(jaccard_index)
                    + "\nF1 score (F-measure): " +str(F1_score)
                    +"\n\nConfusion matrix:"
                    +str(confusion)
                    +"\nACCURACY: " +str(accuracy)
                    +"\nSENSITIVITY: " +str(sensitivity)
                    +"\nSPECIFICITY: " +str(specificity)
                    +"\nPRECISION: " +str(precision)
                    )
    file_perf.close()
