"""Tests used to evaluate networks."""
import os

import numpy as np

import matplotlib
if "DISPLAY" not in os.environ:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils import get_class
from utils import get_confusion_matrix
from utils import round_list
from utils import get_one_hot

from sklearn import metrics


def run_test(network, test_x, test_y, figure_path='figures'):
    """
    Will conduct the test suite to determine model strength.

    Args:
        test_x: data the model has not yet seen to predict
        test_y: corresponding truth vectors
    """
    if network.num_classes is None or network.num_classes == 0:
        print ('Cannot conduct test: there\'s no classification layer')
        return

    if test_y.shape[1] > 1:
        test_y = get_class(test_y)  # Y is in one hot representation

    raw_prediction = network.forward_pass(input_data=test_x,
                                          convert_to_class=False)
    class_prediction = get_class(raw_prediction)

    confusion_matrix = get_confusion_matrix(
        prediction=class_prediction,
        truth=test_y
    )

    tp = np.diagonal(confusion_matrix).astype('float32')
    tn = (np.array([np.sum(confusion_matrix)] *
                   confusion_matrix.shape[0]) -
          confusion_matrix.sum(axis=0) -
          confusion_matrix.sum(axis=1) + tp).astype('float32')
    # sum each column and remove diagonal
    fp = (confusion_matrix.sum(axis=0) - tp).astype('float32')
    # sum each row and remove diagonal
    fn = (confusion_matrix.sum(axis=1) - tp).astype('float32')

    sens = np.nan_to_num(tp / (tp + fn))  # recall
    spec = np.nan_to_num(tn / (tn + fp))
    sens_macro = np.nan_to_num(sum(tp) / (sum(tp) + sum(fn)))
    spec_macro = np.nan_to_num(sum(tn) / (sum(tn) + sum(fp)))
    dice = 2 * tp / (2 * tp + fp + fn)
    ppv = np.nan_to_num(tp / (tp + fp))  # precision
    ppv_macro = np.nan_to_num(sum(tp) / (sum(tp) + sum(fp)))
    npv = np.nan_to_num(tn / (tn + fn))
    npv_macro = np.nan_to_num(sum(tn) / (sum(tn) + sum(fn)))
    accuracy = np.sum(tp) / np.sum(confusion_matrix)
    f1 = np.nan_to_num(2 * (ppv * sens) / (ppv + sens))
    f1_macro = np.average(np.nan_to_num(2 * sens * ppv / (sens + ppv)))

    print ('{} test\'s results'.format(network.name))

    print ('TP:'),
    print (tp)
    print ('FP:'),
    print (fp)
    print ('TN:'),
    print (tn)
    print ('FN:'),
    print (fn)

    print ('\nAccuracy: {}'.format(accuracy))

    print ('Sensitivity:'),
    print(round_list(sens, decimals=3))
    print ('\tMacro Sensitivity: {:.4f}'.format(sens_macro))

    print ('Specificity:'),
    print(round_list(spec, decimals=3))
    print ('\tMacro Specificity: {:.4f}'.format(spec_macro))

    print ('DICE:'),
    print(round_list(dice, decimals=3))
    print ('\tAvg. DICE: {:.4f}'.format(np.average(dice)))

    print ('Positive Predictive Value:'),
    print(round_list(ppv, decimals=3))
    print ('\tMacro Positive Predictive Value: {:.4f}'.format
           (ppv_macro))

    print ('Negative Predictive Value:'),
    print(round_list(npv, decimals=3))
    print ('\tMacro Negative Predictive Value: {:.4f}'.format
           (npv_macro))

    print ('f1-score:'),
    print(round_list(f1, decimals=3))
    print ('\tMacro f1-score: {:.4f}'.format(f1_macro))
    print('')

    if not os.path.exists(figure_path):
        print ('Creating figures folder')
        os.makedirs(figure_path)

    if not os.path.exists('{}/{}{}'.format(figure_path, network.timestamp,
                                           network.name)):
        print ('Creating {}/{}{} folder'.format(figure_path,
                                                network.timestamp,
                                                network.name))
        os.makedirs('{}/{}{}'.format(
            figure_path,
            network.timestamp,
            network.name)
        )
    print ('Saving ROC figures to folder: {}{}'.format(
        network.timestamp,
        network.name)
    )

    plt.figure(3)
    plt.title("Confusion matrix for {}".format(network.name))
    plt.xticks(range(confusion_matrix.shape[0]))
    plt.yticks(range(confusion_matrix.shape[0]))
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.imshow(confusion_matrix, origin='lower', cmap='hot',
               interpolation='nearest')
    plt.colorbar()

    plt.savefig('{}/{}{}/confusion_matrix.png'.format(
        figure_path,
        network.timestamp,
        network.name))

    all_class_auc = metrics.roc_auc_score(
        get_one_hot(test_y),
        get_one_hot(class_prediction),
        average=None
    )

    for i in range(network.num_classes):

        fpr, tpr, thresholds = metrics.roc_curve(test_y,
                                                 raw_prediction[:, i],
                                                 pos_label=i)
        auc = all_class_auc[i]
        print ('AUC: {:.4f}'.format(auc))
        print ('\tGenerating ROC_{} ...'.format(i))

        plt.figure(2)
        plt.ion()
        plt.clf()
        plt.plot(fpr, tpr, label=("AUC: {:.4f}".format(auc)))
        plt.title("ROC Curve for {}_{}".format(network.name, i))
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.legend(loc='lower right')
        plt.ylim(0.0, 1.0)
        plt.xlim(0.0, 1.0)
        plt.show()

        plt.savefig('{}/{}{}/{}.png'.format(figure_path,
                                            network.timestamp,
                                            network.name, i))
    return {
        'accuracy': accuracy,
        'macro_sensitivity': sens_macro,
        'macro_specificity': spec_macro,
        'avg_dice': np.average(dice),
        'macro_ppv': ppv_macro,
        'macro_npv': npv_macro,
        'macro_f1': f1_macro,
        'macro_auc': np.average(all_class_auc)
    }
