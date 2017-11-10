"""Tests used to evaluate networks."""
import os

import numpy as np

from utils import get_class
from utils import get_confusion_matrix
from utils import round_list
from utils import get_one_hot
from utils import get_timestamp

from sklearn import metrics

from copy import deepcopy

from collections import Counter

import matplotlib
if "DISPLAY" not in os.environ:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt


def run_test(network, test_x, test_y, figure_path='figures', plot=True):
    """
    Will conduct the test suite to determine model strength.

    Args:
        test_x: data the model has not yet seen to predict
        test_y: corresponding truth vectors
        figure_path: string, folder to place images in.
        plot: bool, determines if graphs should be plotted when ran.
    """
    if network.num_classes is None or network.num_classes == 0:
        raise ValueError('There\'s no classification layer')

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
    print ('Saving ROC figures to folder: {}/{}{}'.format(
        figure_path,
        network.timestamp,
        network.name)
    )

    plt.figure()
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
    if not plot:
        plt.close()

    fig = plt.figure()
    all_class_auc = []
    for i in range(network.num_classes):
        if network.num_classes == 1:
            fpr, tpr, thresholds = metrics.roc_curve(test_y,
                                                     raw_prediction,
                                                     pos_label=1)
        else:
            fpr, tpr, thresholds = metrics.roc_curve(test_y,
                                                     raw_prediction[:, i],
                                                     pos_label=i)

        auc = metrics.auc(fpr, tpr)
        all_class_auc += [auc]
        # print ('AUC: {:.4f}'.format(auc))
        # print ('\tGenerating ROC {}/{}{}/{}.png ...'.format(figure_path,
        #                                                     network.timestamp,
        #                                                     network.name, i))
        plt.clf()
        plt.plot(fpr, tpr, label=("AUC: {:.4f}".format(auc)))
        plt.title("ROC Curve for {}_{}".format(network.name, i))
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.legend(loc='lower right')
        plt.ylim(0.0, 1.0)
        plt.xlim(0.0, 1.0)

        plt.savefig('{}/{}{}/{}.png'.format(figure_path,
                                            network.timestamp,
                                            network.name, i))
        if plot:
            plt.show(False)

    if not plot:
        plt.close(fig.number)
    print ('Average AUC: : {:.4f}'.format(np.average(all_class_auc)))
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


def k_fold_validation(network, train_x, train_y, k=5, epochs=10,
                      plot=False):
    """
    Conduct k fold cross validation on a network.

    Args:
        network: Network object you want to cross validate
        train_x: ndarray of shape (batch, features), train samples
        train_y: ndarray of shape(batch, classes), train labels
        k: int, how many folds to run
        epochs: int, number of epochs to train each fold

    Returns final metric dictionary
    """
    try:
        network.save_name
    except:
        network.save_model()
    chunk_size = int((train_x.shape[0]) / k)
    results = []
    timestamp = get_timestamp()
    for i in range(k):
        val_x = train_x[i * chunk_size:(i + 1) * chunk_size]
        val_y = train_y[i * chunk_size:(i + 1) * chunk_size]
        tra_x = np.concatenate(
            (train_x[:i * chunk_size], train_x[(i + 1) * chunk_size:]),
            axis=0
        )
        tra_y = np.concatenate(
            (train_y[:i * chunk_size], train_y[(i + 1) * chunk_size:]),
            axis=0
        )
        net = deepcopy(network)
        net.train(
            epochs=epochs,
            train_x=tra_x,
            train_y=tra_y,
            val_x=val_x,
            val_y=val_y,
            batch_ratio=0.05,
            plot=plot
        )
        results += [Counter(run_test(
            net,
            val_x,
            val_y,
            figure_path='figures/kfold_{}{}'.format(timestamp, network.name),
            plot=plot))]
    aggregate_results = reduce(lambda x, y: x + y, results)

    print ('\nFinal Cross validated results')
    print ('-----------------------------')
    for metric_key in aggregate_results.keys():
        aggregate_results[metric_key] /= float(k)
        print ('{}: {:.4f}'.format(metric_key, aggregate_results[metric_key]))

    return aggregate_results
