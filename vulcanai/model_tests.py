"""Tests used to evaluate networks."""
import os

import numpy as np
import pandas as pd
import collections
import scipy

from utils import get_class
from utils import get_confusion_matrix
from utils import round_list
from utils import get_one_hot
from utils import get_timestamp

from sklearn import metrics

from sklearn.utils import shuffle

from copy import deepcopy

from collections import Counter

import matplotlib
if os.name is not "posix":
    if "DISPLAY" not in os.environ:
        matplotlib.use('Agg')
else:
    matplotlib.use('TkAgg')
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
        plt.xlabel('1 - Specificity')
        plt.ylabel('Sensitivity')
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


def k_fold_validation(network, train_x, train_y, df_test_set, prediction, k=5, epochs=10,
                      batch_ratio=1.0, plot=False):
    """
    Conduct k fold cross validation on a network.

    Args:
        network: Network object you want to cross validate
        train_x: ndarray of shape (batch, features), train samples
        train_y: ndarray of shape(batch, classes), train labels
        k: int, how many folds to run
        batch_ratio: float, 0-1 for % of total to allocate for a batch
        epochs: int, number of epochs to train each fold

    Returns final metric dictionary

    TO-DO:
        When k=1 issues arise because chunk sizes aren't created properly
    """

    try:
        network.save_name
    except:
        network.save_model()
    chunk_size = int((train_x.shape[0]) / k)
    results = []
    ls_improvementScores = []
    timestamp = get_timestamp()
    print 'Second features', train_x.shape, train_y.shape
    for i in range(k):
        val_x = train_x[i * chunk_size:(i + 1) * chunk_size]
        val_y = train_y[i * chunk_size:(i + 1) * chunk_size]
        print 'vals: ', val_x.shape, val_y.shape
        tra_x = np.concatenate(
            (train_x[:i * chunk_size], train_x[(i + 1) * chunk_size:]),
            axis=0
        )
        tra_y = np.concatenate(
            (train_y[:i * chunk_size], train_y[(i + 1) * chunk_size:]),
            axis=0
        )
        test_set = np.hstack((val_x, val_y))
        df_test_set = pd.DataFrame(test_set, columns=list(df_test_set))
        df_pred = pd.DataFrame(tra_y)
        tra_y = np.array(pd.get_dummies((df_pred[0])), dtype=np.float32)
        val_y = pd.DataFrame(val_y)
        val_y = np.array((pd.get_dummies(val_y[0])), dtype=np.float32)
        print tra_x.shape
        print tra_y.shape
        print val_x.shape
        print val_y.shape
        if tra_y.shape[1] != 28 and prediction == 'Score':
            x,y = tra_y.shape
            addCol = np.ones((x,(28-y)))
            tra_y = np.hstack((tra_y, addCol))
            print '\t', tra_x.shape
            print '\t', tra_y.shape
        if val_y.shape[1] != 28 and prediction == 'Score':
            x,y = val_y.shape
            addCol = np.ones((x, (28-y)))
            val_y = np.hstack((val_y, addCol))
            print '\t', val_y.shape
        net = deepcopy(network)
        net.train(
            epochs=epochs,
            train_x=tra_x,
            train_y=tra_y,
            val_x=val_x,
            val_y=val_y,
            batch_ratio=0.1,
            plot=plot
        )
        #foldScore = calculate_improvement_score(network=network, df_test=df_test_set, val_x=val_x, val_y=val_y, prediction=prediction)
        foldScore = calculate_improvement_score(network=net, df_base_target=df_test_set, prediction=prediction)
        print 'Fold score of: ', foldScore
        ls_improvementScores.append(foldScore)
        results += [Counter(run_test(
            net,
            val_x,
            val_y,
            figure_path='figures/kfold_{}{}'.format(timestamp, network.name),
            plot=plot))]
        del net
    aggregate_results = reduce(lambda x, y: x + y, results)
    n = len(ls_improvementScores)
    arr_improvementScores = np.array(ls_improvementScores, np.float32)
    sigma = np.std(arr_improvementScores, axis=0)
    ybar = np.mean(arr_improvementScores, axis=0)
    sigma = float(sigma)
    print ls_improvementScores, sigma, ybar
    z_score = (ybar - 1.0)/np.sqrt((sigma**2)/n)
    #z_score = (0.0 - ybar)/(sigma)
    p_value = scipy.stats.norm.sf(abs(z_score))

    print ('\nFinal Cross validated results')
    print ('-----------------------------')
    for metric_key in aggregate_results.keys():
        aggregate_results[metric_key] /= float(k)
        print ('{}: {:.4f}'.format(metric_key, aggregate_results[metric_key]))
    print '***** Z-Score: ', z_score
    #Return p_value
    return aggregate_results, p_value, z_score

def get_drug_scores(network, df):
    print '***** In drug score *****'
    ls_drugs = [1.0, 2.0, 3.0, 4.0]
    dct_armTest = collections.defaultdict()
    for index, row in df.iterrows():
        dct_armTest[index] = {}
    for index, row in df.iterrows():
        subjData = df.loc[[index]]
        subjFeat = subjData.drop(['qids_final_score'], axis=1)
        nn_subjFeat = np.array(subjFeat, dtype=np.float32)
        subjScoreClass = network.forward_pass(input_data=nn_subjFeat, convert_to_class=True)
        dct_armTest[index][row.drug] = subjScoreClass[0]
        for newDrug in ls_drugs:
            if newDrug != row.drug:
                subjData['drug'] = newDrug
                subjFeat = subjData.drop('qids_final_score', axis=1)
                nn_subjFeat = np.array(subjFeat, dtype=np.float32)

                subjScoreClass = network.forward_pass(input_data=nn_subjFeat, convert_to_class=True)
                dct_armTest[index][newDrug] = subjScoreClass[0]
    print '**** Exiting drug score ********'
    return dct_armTest

def get_remission_probs(network, df):
    print '***** In drug score *****'
    ls_drugs = [1.0, 2.0, 3.0, 4.0]
    dct_armTest = collections.defaultdict()
    for index, row in df.iterrows():
        dct_armTest[index] = {}
    for index, row in df.iterrows():
        subjData = df.loc[[index]]
        subjFeat = subjData.drop(['remsn'], axis=1)
        nn_subjFeat = np.array(subjFeat, dtype=np.float32)
        subjScore = network.forward_pass(input_data=nn_subjFeat, convert_to_class=False)
        subjScore = subjScore[0][1] * 100
        remsProb = round(subjScore, 2)
        dct_armTest[index][row.drug] = remsProb
        for newDrug in ls_drugs:
            if newDrug != row.drug:
                subjData['drug'] = newDrug
                subjFeat = subjData.drop('remsn', axis=1)
                nn_subjFeat = np.array(subjFeat, dtype=np.float32)
                subjScore = network.forward_pass(input_data=nn_subjFeat, convert_to_class=False)
                subjScore = subjScore[0][1] * 100
                remsProb = round(subjScore, 2)
                dct_armTest[index][newDrug] = remsProb
    print '**** Exiting drug score ********'
    return dct_armTest
def filter_matched_medications(dct_scores, df_testAns):
    print '***** In filtered subjects ****** '
    dct_DrugScore = {}
    filteredSubj = {}
    for key in dct_scores.keys():
        drugOne = dct_scores[key][1.0]
        drugTwo = dct_scores[key][2.0]
        drugThree = dct_scores[key][3.0]
        drugFour = dct_scores[key][4.0]
        ls_scores = [drugOne, drugTwo, drugThree, drugFour]
        minScoreDrug = min(drugOne, drugTwo, drugThree, drugFour)
        minScoreInd = ls_scores.index(min(ls_scores))
        minScoreInd = minScoreInd + 1.0
        if all(ls_scores[0] == item for item in ls_scores):
            filteredSubj[key] = dct_scores[key][minScoreInd]
            continue
        array_scores = np.array(ls_scores)
        ls_dupIndex = np.where(array_scores == minScoreDrug)
        ls_dupIndex = list(ls_dupIndex[0])
        if df_testAns.loc[key, 'drug'] in ls_dupIndex:
            filteredSubj[key] = dct_scores[key][minScoreInd]

        else:
            #minScoreInd = ls_scores.index(min(ls_scores))
            #minScoreDrug = min(drugOne, drugTwo, drugThree, drugFour)
            minScoreInd = ls_scores.index(min(ls_scores))
            minScoreInd = minScoreInd + 1.0
            if minScoreInd == df_testAns.loc[key, 'drug']:
                filteredSubj[key] = dct_scores[key][minScoreInd]

    print '****** Exiting filter subjects ********'
    return filteredSubj

def check_dup_index(scores, item):
    return [ind for ind, x in enumerate(scores) if x[0] == item]

def calculate_improvement_score(network, df_base_target, prediction):
    #Should be passed the test predictions and the test targets (truth)

    #Calculate drop in QIDS score with Targets. avg across all: V_t
        # (baseline_score - target).mean()

    #Find subset of predictions where medication predicted is matched with
    #On the subset, calculate drop in QIDS score. avg across all: V_p

    #Find improvement score = V_t - V_p

    #Return improvement score
    print '******** In calculate improvement score *********'
    df_testAns = deepcopy(df_base_target)

    if prediction == 'Score':
        dct_scores = get_drug_scores(network, df_base_target)
    elif prediction == 'Remission':
        dct_scores = get_remission_probs(network, df_base_target)
    dct_filteredSubj = filter_matched_medications(dct_scores, df_testAns)
    dct_filteredSubjBase = {}

    if prediction == 'Score':
        v_t_baseQids = np.array(df_base_target.qids_base, np.float32)
        v_t_finalQids = np.array(df_base_target.qids_final_score, np.float32)
        for index, row in df_base_target.iterrows():
            if index in dct_filteredSubj.keys():
                dct_filteredSubjBase[index] = row.qids_base
        V_t = (v_t_baseQids - v_t_finalQids).mean()


    elif prediction == 'Remission':
        #No one is in remission at base
        remsnCount = 0
        nonRemsnCount = 0
        remsnTotal = 0
        ls_base = []
        ls_final = []
        for index, row, in df_base_target.iterrows():
            #if index in dct_filteredSubj.keys():
            if row.remsn == 1.0:
                remsnCount = remsnCount + 1.0
                remsnTotal = remsnTotal + 1.0
                    # ls_filtSubjFinal.append(1.0)
            else:
                nonRemsnCount = nonRemsnCount + 1.0
                remsnTotal = remsnTotal + 1.0
        remsnRate = float(remsnCount/remsnTotal) * 100
        v_t_baseQids = 0
        for index, row in df_base_target.iterrows():
            ls_base.append(0.0)
            ls_final.append(remsnRate)
        v_t_baseQids = np.array(ls_base, np.float32)
        v_t_finalQids = np.array(ls_final, np.float32)
        V_t = remsnRate
        #V_t = (v_t_baseQids - v_t_finalQids).mean()
        print V_t

    ls_filtSubjBase = []
    ls_filtSubjFinal = []

    if prediction == 'Score':
        for key in dct_filteredSubj.keys():
            ls_filtSubjBase.append(dct_filteredSubjBase[key])
            ls_filtSubjFinal.append(dct_filteredSubj[key])
        v_p_baseQids = np.array(ls_filtSubjBase, np.float32)
        v_p_finalQids = np.array(ls_filtSubjFinal, np.float32)
        V_p = (v_p_baseQids - v_p_finalQids).mean()

    elif prediction == 'Remission':
        remsnCount = 0
        nonRemsnCount = 0
        remsnTotal = 0
        for index, row, in df_base_target.iterrows():
            #if index in dct_filteredSubj.keys():
            if row.remsn == 1.0:
                remsnCount = remsnCount + 1.0
                remsnTotal = remsnTotal + 1.0
                # ls_filtSubjFinal.append(1.0)
            else:
                nonRemsnCount = nonRemsnCount + 1.0
                remsnTotal = remsnTotal + 1.0
        remsnRate = float(remsnCount / remsnTotal) * 100

        for key in dct_filteredSubj.keys():
            dct_filteredSubjBase[index] = remsnRate
        for key in dct_filteredSubj.keys():
            ls_filtSubjBase.append(0.0)
            ls_filtSubjFinal.append(dct_filteredSubj[key])
        v_p_baseQids = np.array(ls_filtSubjBase, np.float32)
        v_p_finalQids = np.array(ls_filtSubjFinal, np.float32)
        #V_p = (v_p_baseQids/v_p_finalQids).mean()
        #V_p = (v_p_baseQids - v_p_finalQids).mean()
        V_p = v_p_finalQids.mean()
        print 'V_t & V_p scores: ', V_t, V_p
        ## We want improvement_score to be greater than 1
        improvement_score = V_p/V_t
        #improvement_score = V_t/V_p
        print '\t Improvement Scores: ', improvement_score
        count = 0.0
        remSum = 0.0
        for key in dct_filteredSubj.keys():
            count += 1.0
            remSum += dct_filteredSubj[key]
        meanRem = remSum/count
    #Negative implied prediction is better
    #improvement_score = V_t - V_p
    print improvement_score
    print '******** Exiting improvment score ******'
    return improvement_score

def bootfold_p_estimate(network, data_matrix, test_matrix, to_predict, n_samples=10, k_folds=10):
    """
    Bootstrap k-fold CV p-value estimation for improvement scores.
    Try some sort of stratified bootstrapping for class imbalance

    Args:
        network: a Network object
        data_matrix: ndarray with index on 0th dimension
        n_samples: how many bootstrap sampls to generate
        k_folds: how many cross validated folds per bootstrap sample
    """
    data_matrix = np.array(data_matrix, dtype=np.float32)

    sample_size = data_matrix.shape[0]
    ls_p_values = []
    ls_z_scores = []
    for samp in range(n_samples):
        #sample = data_matrix[np.random.choice(sample_size, size=sample_size, replace=True), :]
        #print 'Data Matrix Shape before: ', (data_matrix.shape)
        sample = data_matrix[np.random.choice(sample_size, size=sample_size, replace=True)]
        #feat = np.delete(sample,[-1],1)
        feat = sample[:, :-1]
        df_feat = pd.DataFrame(feat)
        train_x = np.array(df_feat, dtype=np.float32)
        train_y = sample[:,-1]
        df_pred = pd.DataFrame(train_y)
        train_y = np.array(df_pred, dtype=np.float32)
        #train_y = np.array(pd.get_dummies(df_pred[0]))
        print 'Train size', train_y.shape, train_x.shape
        _, p_value, z_score = k_fold_validation(network=network, train_x=train_x, train_y=train_y, df_test_set=test_matrix, prediction=to_predict, k=k_folds, epochs=500)
        ls_p_values.append(p_value)
        ls_z_scores.append(z_score)
        print p_value, z_score
    print 'z-scores: ', ls_z_scores
    print 'P-values: ', ls_p_values
    print 'Average of p-values: ', np.average(ls_p_values)