# nohup python web_interface_file.py --test_file_dir > log/20240120.log 2>&1 &

import os
import lightgbm as lgb
import numpy as np

import pandas as pd

import argparse
import os

import rpf
from cutoff import get_y_pred_labels


parser = argparse.ArgumentParser()
parser.add_argument('--test_file_dir', default='data/CREAT_random.csv')
parser.add_argument('--model_result_name', default='CREAT_random')
args = parser.parse_args()

test_file_dir = args.test_file_dir
model_result_name = args.model_result_name

output_dir = 'output/'+model_result_name  
predict_label_file = output_dir+'/predict_label_file_'+model_result_name+'.xlsx'
summary_file_dir = output_dir+'/evaluate_'+model_result_name+'.csv'
fig_dir = 'output/'+model_result_name+'/'+model_result_name+'_auc.png'
ap_fig_dir = 'output/'+model_result_name+'/'+model_result_name+'_ap.png'


if os.path.exists(output_dir) == False:
    os.system('mkdir ' + output_dir)

model_configs_dir = 'bert_config'
models = 'biobert_v1.1_pubmed,NCBI_BERT_pubmed_uncased_L-12_H-768_A-12,NCBI_BERT_pubmed_mimic_uncased_L-12_H-768_A-12,uncased_L-12_H-768_A-12'
models = models.split(',')
model_names = {'Sci-BBUP':'NCBI_BERT_pubmed_uncased_L-12_H-768_A-12', 'Sci-BBUPC':'NCBI_BERT_pubmed_mimic_uncased_L-12_H-768_A-12', 
                        'BIO-BBU':'biobert_v1.1_pubmed', 'BBU':'uncased_L-12_H-768_A-12','lightGBM':'lightgbm'}
  
  
model_label_names = ['Sci-BBUP_predict_label','Sci-BBUPC_predict_label',
'BIO-BBU_predict_label','BBU_predict_label','lightGBM_balance_predict_label',
'svm','replace_first_result','replace_second_result','first_label','second_label','primary_screen_result',
'lightGBM_predict_label']
        
def get_folder_dir(folders_dir):
    for base_path, folders, files in os.walk(folders_dir):
        break
    return folders

def run_bert(model_config,test_file_dir): 
    os.system('python BERT_classify.py\
        --task_name="Classify"  \
        --do_lower_case=True \
        --do_test=True \
        --vocab_file=bert_config/%s/vocab.txt  \
        --bert_config_file=bert_config/%s/bert_config.json \
        --init_checkpoint=bert_config/%s/bert_model.ckpt   \
        --max_seq_length=317  \
        --train_batch_size=2   \
        --learning_rate=2e-5   \
        --num_train_epochs=4.0   \
        --output_dir=output/%s/%s\
        --test_file_dir=%s \
        ' %
        (model_config,model_config,model_config,model_result_name,model_config,test_file_dir)
        )

def get_model_input(output_dir):
    models_output = {}
    for model in models:
        train_data_dir = '%s/%s/%s' % (output_dir,model,'test_results.tsv')
        model_output = []
        ids = []
        with open(train_data_dir,'r') as file:
            for line in file:
                line=line.strip().split('\t')
                ids.append(line[0])
                model_output.append(float(line[-1]))
        models_output[model] = model_output
    predict_data = np.zeros(shape=(len(model_output),len(models)))
    for i in range(len(models)):
        predict_data[:,i]=models_output[models[i]]
    return ids,predict_data
    
def get_lightgbm(): 
    gbm = lgb.Booster(model_file='lightgbm_model.txt')# 模型预测
    ids,predict_data =  get_model_input(output_dir)
    y_preds = gbm.predict(predict_data, num_iteration=gbm.best_iteration)
    # y_pred_labels = get_y_pred_labels(y_preds,'sensitive')
    y_pred_labels = get_y_pred_labels(y_preds,'balanced')
    return ids,y_preds,y_pred_labels
    


def write_predict_label():
    data = pd.read_csv(test_file_dir)
    predict_probabilities = {}
    #--------------sub model-------------
    for model in model_names.keys():
        if model != 'lightGBM':
            predict_labels, predict_scores = [],[]
            file = 'output/'+model_result_name+'/'+model_names[model]+'/test_results.tsv'
            with open(file,'r') as file_data:
                for line in file_data:
                    line = line.split('\t')
                    predict_label = line[-2]
                    predict_score = line[-1].strip()
                    predict_labels.append(predict_label)
                    predict_scores.append(predict_score)
        data[model+'_predict_label'] =  predict_labels
        data[model+'_predict_score'] =  predict_scores
        predict_probabilities[model] =  predict_scores
    #--------------lightgbm-------------
    ids,y_preds,predict_labels = get_lightgbm()
    data['lightGBM_predict_label'] =  predict_labels
    data['vpredict_score'] =  y_preds
    predict_probabilities['lightGBM'] =  y_preds
    df = pd.DataFrame(data)
    df.to_excel(predict_label_file) 


def combine_evaluate(predict_label_file):
    all_label_df = pd.read_excel(predict_label_file)
    for model_label_name in model_label_names:
        labels_all = list(all_label_df['rct_label'])
        predlabels_all = list(all_label_df[model_label_name])
        ml_compare = rpf.ML_compare(model_label_name,labels_all, predlabels_all,None,
                    fig_dir = fig_dir,
                    ap_fig_dir = ap_fig_dir,
                    pos_label='1',
                    summary_file_dir =summary_file_dir)
                        
                    
                   
combine_evaluate(predict_label_file)