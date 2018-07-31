import numpy as np
import pandas as pd
import os
import random
import math
import csv
import collections
from collections import OrderedDict
import sys
import copy
import timeit
from operator import itemgetter

from pandas.core.common import array_equivalent

from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.utils import shuffle

from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, VotingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.externals.six import StringIO
from sklearn.feature_selection import mutual_info_classif, VarianceThreshold, RFECV, RFE
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, RandomizedLasso, LassoCV, RandomizedLogisticRegression

import theano.tensor as T
from vulcanai.net import Network
from vulcanai.snapshot_ensemble import Snapshot
from vulcanai.utils import *
from vulcanai.model_tests import run_test, k_fold_validation, bootfold_p_estimate

df_stard = pd.read_csv("~/training_scripts/Notebooks/STARD/STARD_CSV/STARD07_13.csv")
df_comed = pd.read_csv("~/training_scripts/Notebooks/COMED/Custom_CSV/COMED07_05.csv")
#df_stard = pd.read_csv("~/training_scripts/Notebooks/STARD/STARD_CSV/STARDBaseQIDSRAll07_16.csv")
#df_comed = pd.read_csv("~/training_scripts/Notebooks/COMED/Custom_CSV/COMEDBASEAll07_05.csv")
df_stard = df_stard.apply(pd.to_numeric, errors='ignore')
df_comed = df_comed.apply(pd.to_numeric, errors='ignore')

df_stard.loc[df_stard['qscur_r'] == -5.0, 'qscur_r'] = np.nan
df_stard.loc[df_stard['qscur_r'] == -4.0, 'qscur_r'] = np.nan
df_stard.loc[df_stard['qscur_r'] == -3.0, 'qscur_r'] = np.nan

df_stard['qids_final_score'] = np.nan
df_comed['qids_final_score'] = np.nan

df_stardLevel1 = df_stard.loc[(df_stard['level'] == 'Level 1')]
df_stardBase = df_stard.loc[(df_stard['level'] == 'Level 1') & (df_stard['days_baseline'] == 0)]

df_stardBase['drug'] = 'Citalopram'
df_stardBase.drug.value_counts()

level1BySubj = df_stardLevel1.groupby('subjectkey')
#dct_columnInfo['hamd'] = 'Hamd score'

pd.options.mode.chained_assignment = None
for subj, df in level1BySubj:
    maxDayTot = df.loc[df['days_baseline'].idxmax()]
    maxDayTotInd = df.loc[df['days_baseline'] == maxDayTot.days_baseline].index[0]

    if maxDayTot.days_baseline > 6:
        if not math.isnan(maxDayTot.qscur_r):
            df_stardBase.loc[df_stardBase['subjectkey'] == maxDayTot.subjectkey, 'qids_final_score'] = maxDayTot.qscur_r
        else:
            df = df.drop(maxDayTotInd)
            while len(df) > 0:
                maxDayTot = df.loc[df['days_baseline'].idxmax()]
                maxDayTotInd = df.loc[df['days_baseline'] == maxDayTot.days_baseline].index[0]
                if math.isnan(maxDayTot.qscur_r):
                    df = df.drop(maxDayTotInd)
                elif not math.isnan(maxDayTot.qscur_r):
                    break
                else:
                    df = df.drop(maxDayTotInd)

            if math.isnan(maxDayTot.qscur_r):
                df_stardBase.loc[df_stardBase['subjectkey'] == maxDayTot.subjectkey, 'qids_final_score'] = np.nan
            elif not math.isnan(maxDayTot.qscur_r):
                df_stardBase.loc[df_stardBase['subjectkey'] == maxDayTot.subjectkey, 'qids_final_score'] = maxDayTot.qscur_r
            else:
                df_stardBase.loc[df_stardBase['subjectkey'] == maxDayTot.subjectkey, 'qids_final_score'] = np.nan

df_comedBase = df_comed.loc[df_comed['week'] == 0]
comedBySubj = df_comed.groupby('subjectkey')
#dct_columnInfo['hamd'] = 'Hamd score'

pd.options.mode.chained_assignment = None
for subj, df in comedBySubj:
    maxDayTot = df.loc[df['days_baseline'].idxmax()]
    maxDayTotInd = df.loc[df['days_baseline'] == maxDayTot.days_baseline].index[0]

    if maxDayTot.days_baseline > 6:
        if not math.isnan(maxDayTot.qstot):
            df_comedBase.loc[df_comedBase['subjectkey'] == maxDayTot.subjectkey, 'qids_final_score'] = maxDayTot.qstot
        else:
            df = df.drop(maxDayTotInd)
            while len(df) > 0:
                maxDayTot = df.loc[df['days_baseline'].idxmax()]
                maxDayTotInd = df.loc[df['days_baseline'] == maxDayTot.days_baseline].index[0]
                if math.isnan(maxDayTot.qstot):
                    df = df.drop(maxDayTotInd)
                elif not math.isnan(maxDayTot.qstot):
                    break
                else:
                    df = df.drop(maxDayTotInd)

            if math.isnan(maxDayTot.qstot):
                df_comedBase.loc[df_comedBase['subjectkey'] == maxDayTot.subjectkey, 'qids_final_score'] = np.nan
            elif not math.isnan(maxDayTot.qstot):
                df_comedBase.loc[df_comedBase['subjectkey'] == maxDayTot.subjectkey, 'qids_final_score'] = maxDayTot.qstot
            else:
                df_comedBase.loc[df_comedBase['subjectkey'] == maxDayTot.subjectkey, 'qids_final_score'] = np.nan


df_stardBase = pd.read_csv("~/training_scripts/Notebooks/STARD/STARD_CSV/STARDBaseQIDSRAll07_16.csv")
df_comedBase = pd.read_csv("~/training_scripts/Notebooks/COMED/Custom_CSV/COMEDBaseAll07_05.csv")

df_stardBaseModel = copy.deepcopy(df_stardBase)
df_comedBaseModel = copy.deepcopy(df_comedBase)
df_comedBaseModel['qids_base'] = df_comedBaseModel.qstot
df_stardBaseModel['qids_base'] = df_stardBaseModel.qscur_r
if 'subjectkey' in list(df_stardBaseModel):
    del df_stardBaseModel['subjectkey']
if 'src_subject_id' in list(df_stardBaseModel):
    del df_stardBaseModel['src_subject_id']
if 'dataset_id' in list(df_stardBaseModel):
    del df_stardBaseModel['dataset_id']
if 'days_baseline' in list(df_stardBaseModel):
    del df_stardBaseModel['days_baseline']
if 'promoted_subjectkey' in list(df_stardBaseModel):
    del df_stardBaseModel['promoted_subjectkey']
if 'clinicid' in list(df_stardBaseModel):
    del df_stardBaseModel['clinicid']
if 'clinid' in list(df_stardBaseModel):
    del df_stardBaseModel['clinid']
if 'medication1_dosage' in list(df_stardBaseModel):
    del df_stardBaseModel['medication1_dosage']
if 'stsd1' in list(df_stardBaseModel):
    del df_stardBaseModel['stsd1']
if 'medication2_name' in list(df_stardBaseModel):
    del df_stardBaseModel['medication2_name']
if 'medication_name' in list(df_stardBaseModel):
    del df_stardBaseModel['medication_name']
if 'crcid' in list(df_stardBaseModel):
    del df_stardBaseModel['crcid']
print len(list(df_stardBaseModel))

if 'subjectkey' in list(df_comedBaseModel):
    del df_comedBaseModel['subjectkey']
if 'src_subject_id' in list(df_comedBaseModel):
    del df_comedBaseModel['src_subject_id']
if 'dataset_id' in list(df_comedBaseModel):
    del df_comedBaseModel['dataset_id']
if 'days_baseline' in list(df_comedBaseModel):
    del df_comedBaseModel['days_baseline']
if 'promoted_subjectkey' in list(df_comedBaseModel):
    del df_comedBaseModel['promoted_subjectkey']
if 'clinicid' in list(df_comedBaseModel):
    del df_comedBaseModel['clinicid']
if 'clinid' in list(df_comedBaseModel):
    del df_comedBaseModel['clinid']
if 'medication1_dosage' in list(df_comedBaseModel):
    del df_comedBaseModel['medication1_dosage']
if 'stsd1' in list(df_comedBaseModel):
    del df_comedBaseModel['stsd1']
if 'medication2_name' in list(df_comedBaseModel):
    del df_comedBaseModel['medication2_name']
if 'medication_name' in list(df_comedBaseModel):
    del df_comedBaseModel['medication_name']
if 'crcid' in list(df_comedBaseModel):
    del df_comedBaseModel['crcid']
print len(list(df_comedBaseModel))

df_stardBaseModel['drug'] = 4.0
df_comedBaseModel.rename(columns={'arm': 'drug'}, inplace=True)
print len(list(df_comedBaseModel)), len(list(df_stardBaseModel))
for comedColumn in list(df_comedBaseModel):
    if comedColumn not in list(df_stardBaseModel):
        del df_comedBaseModel[comedColumn]
for stardColumn in list(df_stardBaseModel):
    if stardColumn not in list(df_comedBaseModel):
        del df_stardBaseModel[stardColumn]
print len(list(df_comedBaseModel)), len(list(df_stardBaseModel))
print len(df_comedBaseModel), len(df_stardBaseModel)


df_stardBaseModel = df_stardBaseModel.apply(pd.to_numeric, errors='ignore')
df_comedBaseModel = df_comedBaseModel.apply(pd.to_numeric, errors='ignore')

dct_stardCodes = {}
dct_comedCodes = {}

print "Number of features before encoding: %d " % len(list(df_stardBaseModel))
labeler = LabelEncoder()
for column in list(df_stardBaseModel):
    if column == 'remsn':
        continue
    elif df_stardBaseModel[column].dtype == 'object':
        print column
        newCol = str(column) + '_codes'
        df_stardBaseModel[column] = df_stardBaseModel[column].astype('category')
        df_stardBaseModel[newCol] = df_stardBaseModel[column].cat.codes
        dct_stardCodes[column] = df_stardBaseModel[[column, newCol]]
        del df_stardBaseModel[column]
print "Number of features after encoding: %d " % len(list(df_stardBaseModel))

print "Number of features before encoding: %d " % len(list(df_comedBaseModel))
labeler = LabelEncoder()
for column in list(df_comedBaseModel):
    if column == 'remsn':
        continue
    elif df_comedBaseModel[column].dtype == 'object':
        newCol = str(column) + '_codes'
        df_comedBaseModel[column] = df_comedBaseModel[column].astype('category')
        df_comedBaseModel[newCol] = df_comedBaseModel[column].cat.codes
        dct_comedCodes[column] = df_comedBaseModel[[column, newCol]]
        del df_comedBaseModel[column]

print "Number of features after encoding: %d " % len(list(df_comedBaseModel))

for column in list(df_comedBaseModel):
    df_comedBaseModel[column].fillna(-1, inplace=True)

for column in list(df_stardBaseModel):
    df_stardBaseModel[column].fillna(-1, inplace=True)

features = df_comedBaseModel.drop(['remsn'], axis=1)
for column in list(features):
    if features[column].dtypes == 'object':
        features[column] = features[column].astype(np.float32)

features['remsn'] = df_comedBaseModel.remsn
df_comedBaseModel = copy.deepcopy(features)

features = df_stardBaseModel.drop(['remsn'], axis=1)
for column in list(features):
    if features[column].dtypes == 'object':
        features[column] = features[column].astype(np.float32)

features['remsn'] = df_stardBaseModel.remsn
df_stardBaseModel = copy.deepcopy(features)

print len(list(df_comedBaseModel)), len(list(df_stardBaseModel))
for comedColumn in list(df_comedBaseModel):
    if comedColumn not in list(df_stardBaseModel):
        del df_comedBaseModel[comedColumn]
for stardColumn in list(df_stardBaseModel):
    if stardColumn not in list(df_comedBaseModel):
        del df_stardBaseModel[stardColumn]
print len(list(df_comedBaseModel)), len(list(df_stardBaseModel))
print len(df_comedBaseModel), len(df_stardBaseModel)

df_combined = df_comedBaseModel.append(df_stardBaseModel)
if 'qids_final_score' in list(df_combined):
    df_combined = df_combined.loc[df_combined['qids_final_score'] != 99.0]

print 'Before known features'
print len(list(df_combined))

currFeat = ['anshk', 'dage', 'ebdsg', 'educat', 'emspk', 'emwry', 'frsit', \
            'gender_codes', 'happt', 'hengy', 'hhypc', 'hintr', 'hmdsd',\
            'hmnin', 'hpanx', 'hsanx', 'hsex', 'hslow', 'hsuic', 'ildn', \
            'interview_age', 'ivrtn', 'ivwfr', 'obfgt', 'phstm', 'qids_base', \
            'qstot', 'teblk', 'tetht', 'totincom', 'trwit', 'vagit', 'vapin', \
            'vcntr', 'vemin', 'vengy', 'vhysm', 'vmdsd', 'vmnin', 'vslow', \
            'vsoin', 'vvwsf', 'vwtdc', 'vwtin', 'wiser', 'wpai02', 'wpai04', \
            'wpai05', 'wsas01', 'wynrv', 'drug', 'qids_final_score']
remsnCurrFeat = ['ebhgy', 'educat', 'frcwd', 'hengy', 'hsuic', 'phach', \
                 'qstot', 'tejmp', 'teshk', 'totincom', 'trwit', 'vengy', \
                 'vmdsd', 'vsoin', 'vwtin', 'wpai04', 'drug', 'remsn']
#remsnCurrFeat = ['anbrt', 'dkmge', 'dkpbm', 'ebaln', 'ebcrl', 'ebhgy', 'ebups', \
#                 'emqst', 'emstu', 'emupr', 'emwrt', 'frcar', 'frfar', 'frlne', \
#                 'fropn', 'hinsg', 'hintr', 'iplsr', 'islow', 'ismtc', 'isuic', \
#                 'isymp', 'obcln', 'obcnt', 'obgrm', 'phach', 'phsck', 'tejmp', \
#                 'visday', 'vsuic', 'wistp', 'drug', 'remsn']
df_combined = df_combined[remsnCurrFeat]
print 'After known features'
print len(list(df_combined))

#df_stardBaseModelNN = df_combined.loc[df_combined.drug == 4.0]
#df_comedBaseModelNN = df_combined.loc[df_combined.drug != 4.0]
#print len(df_stardBaseModelNN), len(df_comedBaseModelNN)
#df_stardBaseModelNN = df_stardBaseModelNN.sample(frac=1).reset_index(drop=True)
#df_stardTestNN = df_stardBaseModelNN[0:308]

#df_stardTestNN.drug.value_counts()
#df_stardBaseModelNN = df_stardBaseModelNN.drop(df_stardBaseModelNN.index[0:308])

#df_comedBaseModelNN = df_comedBaseModelNN.sample(frac=1).reset_index(drop=True)
#df_comedTestNN = df_comedBaseModelNN[0:140]

#df_comedTestNN.drug.value_counts()

#df_combinedTestNN = df_comedTestNN.append(df_stardTestNN)
#print len(df_combinedTestNN)
#df_comedBaseModelNN = df_comedBaseModelNN.drop(df_comedBaseModelNN.index[0:140])

#df_combinedNN = df_comedBaseModelNN.append(df_stardBaseModelNN)
df_combinedNN = copy.deepcopy(df_combined)
len(df_combinedNN)

#df_combinedTestNN = df_combinedTestNN[list(df_combinedNN)]
df_combinedTestNN = df_combinedNN.iloc[[1]]
print "*****", len(list(df_combinedTestNN))
len(list(df_combinedNN)), len(list(df_combinedTestNN))
if 'qids_final_score' in list(df_combinedNN):
    df_combinedNN = df_combinedNN.loc[df_combinedNN['qids_final_score'] != -1.0]
    #df_combinedTestNN = df_combinedTestNN.loc[df_combinedTestNN['qids_final_score'] != -1.0]
    #df_combinedTestNN = df_combinedTestNN.reset_index()

    reserve = 0.8
    features = df_combinedNN.drop(['qids_final_score'], axis=1)

    for column in list(features):
        if features[column].dtypes == 'object':
            features[column] = features[column].astype(np.float32)

    nn_features = np.array(features, dtype=np.float32)
    nn_pred = np.array(pd.get_dummies(df_combinedNN.qids_final_score), dtype=np.float32)
    nn_features, nn_pred = shuffle(nn_features, nn_pred, random_state=0)



else:
    reserve = 0.8
    features = df_combinedNN.drop(['remsn'], axis=1)

    for column in list(features):
        if features[column].dtypes == 'object':
            features[column] = features[column].astype(np.float32)

    nn_features = np.array(features, dtype=np.float32)
    nn_pred = np.array(pd.get_dummies(df_combinedNN.remsn), dtype=np.float32)
    nn_features, nn_pred = shuffle(nn_features, nn_pred, random_state=0)

sizeOfFeatures = int(nn_features.shape[1])

input_var = T.fmatrix('input')
output = T.fmatrix('truth')

num_classes = nn_pred.shape[-1]
print sizeOfFeatures, num_classes
network_dense_config = {
    'mode': 'dense',
    'units': [sizeOfFeatures],
    'dropouts': [0.5],
}


dense_netComStarQIDSScore = Network(
    name='STARD_qidsScore5_classifier',
    dimensions=(None, sizeOfFeatures),
    input_var=input_var,
    y=output,
    num_classes=num_classes,
    config=network_dense_config,
    #input_network={'network': autoencoder, 'layer': 10, 'get_params': True},
    activation='selu',
    pred_activation='softmax',
    optimizer='adam',
    learning_rate=0.00001,
    stopping_rule=None
)

#dense_net.create_classification_layer(dense_net, arms_classes, None)
#qids_score5k = k_fold_validation(dense_netComStarQIDSScore, nn_features, nn_qids_range, k=5, epochs=150)
#dense_netComStarQIDSScore.train(epochs=150, train_x=nn_features, train_y=nn_qids_score, val_x=nn_featuresTest, val_y=nn_qids_scoreTest, batch_ratio=0.1, plot=True, change_rate=None)
colOrder = ['anshk', 'dage', 'ebdsg', 'educat', 'emspk', 'emwry', 'frsit', \
            'gender_codes', 'happt', 'hengy', 'hhypc', 'hintr', 'hmdsd',\
            'hmnin', 'hpanx', 'hsanx', 'hsex', 'hslow', 'hsuic', 'ildn', \
            'interview_age', 'ivrtn', 'ivwfr', 'obfgt', 'phstm', 'qids_base', \
            'qstot', 'teblk', 'tetht', 'totincom', 'trwit', 'vagit', 'vapin', \
            'vcntr', 'vemin', 'vengy', 'vhysm', 'vmdsd', 'vmnin', 'vslow', \
            'vsoin', 'vvwsf', 'vwtdc', 'vwtin', 'wiser', 'wpai02', 'wpai04', \
            'wpai05', 'wsas01', 'wynrv', 'drug', 'qids_final_score']

remsnColOrder = ['ebhgy', 'educat', 'frcwd', 'hengy', 'hsuic', 'phach', \
                 'qstot', 'tejmp', 'teshk', 'totincom', 'trwit', 'vengy', \
                 'vmdsd', 'vsoin', 'vwtin', 'wpai04', 'drug', 'remsn']
#remsnColOrder = ['anbrt', 'dkmge', 'dkpbm', 'ebaln', 'ebcrl', 'ebhgy', 'ebups', \
#                 'emqst', 'emstu', 'emupr', 'emwrt', 'frcar', 'frfar', 'frlne', \
#                 'fropn', 'hinsg', 'hintr', 'iplsr', 'islow', 'ismtc', 'isuic', \
#                 'isymp', 'obcln', 'obcnt', 'obgrm', 'phach', 'phsck', 'tejmp', \
#                 'visday', 'vsuic', 'wistp', 'drug', 'remsn']
df_combinedNN = df_combinedNN[remsnColOrder]
df_combinedTestNN = df_combinedTestNN[list(df_combinedNN)]
df_combinedTestNN = df_combinedTestNN[remsnColOrder]
df_combinedNN = df_combinedNN.fillna(-1)
df_combinedTestNN = df_combinedTestNN.fillna(-1)
df_combinedTestNN = df_combinedTestNN.reset_index()
ls_cols = list(df_combinedNN)
if 'index' in list(df_combinedTestNN):
    del df_combinedTestNN['index']
#print len(list(df_combinedTestNN)), list(df_combinedTestNN)
bootfold_p_estimate(network=dense_netComStarQIDSScore, data_matrix=df_combinedNN, test_matrix= df_combinedTestNN, to_predict='Remission', n_samples=15, k_folds=10)
