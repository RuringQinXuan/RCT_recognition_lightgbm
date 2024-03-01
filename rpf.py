import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
from utils import accuracy
from sklearn.metrics import precision_recall_curve, average_precision_score
import os

class ML_compare:
    def __init__(self,model_name,true_labels, predict_labels, predict_scores,fig_dir,ap_fig_dir,pos_label='1',summary_file_dir = 'summary.csv'):
    # def __init__(self,model_name, true_labels, predict_labels , fig_dir,pos_label='1',summary_file_dir = 'summary.csv',predict_scores=None,threshold=None):
    # def __init__(self,model_name,true_labels, predict_labels,fig_dir,pos_label='1',summary_file_dir = 'summary.csv'):
        self.model_name = model_name
        self.true_labels = true_labels
        self.predict_labels = predict_labels 
        self.predict_scores = predict_scores
        self.summary_file_dir= summary_file_dir
        self.fig_dir = fig_dir
        self.ap_fig_dir = ap_fig_dir
        self.pos_label = pos_label
        self.dic={}
        self.dic['model_name'] = self.model_name
        self.tp,self.fp,self.tn,self.fn = self.get_tp_fp_tn_fp()
        self.get_r_p_f_a()
        self.get_creat_interest()
        # self.draw_predict()
        self.writ_result()
    def get_tp_fp_tn_fp(self):
        [tp,fp,tn,fn] = [0,0,0,0]
        for i in range(len(self.true_labels)):
            if str(int(self.true_labels[i])) == self.pos_label and str(int(self.predict_labels[i])) == self.pos_label:
                tp += 1
            if str(int(self.true_labels[i])) != self.pos_label and str(int(self.predict_labels[i])) == self.pos_label:
                fp += 1
            if str(int(self.true_labels[i])) != self.pos_label and str(int(self.predict_labels[i])) != self.pos_label:
                tn += 1
            if str(int(self.true_labels[i])) == self.pos_label and str(int(self.predict_labels[i])) != self.pos_label:
                fn += 1
        return tp,fp,tn,fn
    def get_r_p_f_a(self):
        eval_acc, eval_prec, eval_recall, eval_f1, p, r, f = accuracy(self.predict_labels, self.true_labels)
        if self.tp+self.fp > 0:
            precision_score =  (self.tp/(self.tp+self.fp))
        else:
            precision_score = 0
        if self.tp+self.fn > 0:
            recall_score = self.tp/(self.tp+self.fn)
        else:
            recall_score = 0
        if recall_score+precision_score > 0 :
            f1 = 2*precision_score*recall_score/(recall_score+precision_score)
        else:
            f1 = 0
        self.accuracy = '%.3f' % ((self.tp+self.tn)/len(self.true_labels))
        self.dic['tp'] = self.tp
        self.dic['fp'] = self.fp
        self.dic['tn'] = self.tn
        self.dic['fn'] = self.fn
        self.dic['all'] = self.tp+self.fp+self.tn+self.fn
        self.dic['recall_score'] = '%.3f' % (recall_score)
        self.dic['precision_score'] = '%.3f' % (precision_score)
        self.dic['f1'] = '%.3f' % (f1)
        self.dic['eval_recall'] = '%.3f' % (eval_recall)
        self.dic['eval_prec'] = '%.3f' % (eval_prec)
        self.dic['eval_f1'] = '%.3f' % (eval_f1)
    def get_creat_interest(self):
        self.manual = self.tp+self.fn
        self.correctly_predicted = self.tp
        self.missed = self.fn
        self.total_predicted_positive = self.tp+self.fp
        self.Ineligible_spared_to_screen = self.tn+self.fn
        if self.manual > 0:
            self.Sensitivity = (self.tp/self.manual)
        else:
            self.Sensitivity =  0
        if self.fp+self.tn > 0:
            self.specificity = (self.tn/(self.fp+self.tn))
        else:
            self.specificity = 0
    def draw_predict(self):
        plt.switch_backend('agg')
        fig = plt.figure(figsize=(12, 6))
        plt.figure()
        fpr, tpr, threshold = metrics.roc_curve(self.true_labels, self.predict_scores)
        y = tpr - fpr
        Youden_index = np.argmax(y)  # Only the first occurrence is returned.
        self.optimal_threshold = threshold[Youden_index]
        self.roc_auc = metrics.auc(fpr, tpr)
        plt.plot(fpr, tpr, 'b', label = 'Val AUC = %0.3f' % self.roc_auc)
        plt.legend(loc = 'lower right')
        plt.title('Validation Receiver Operating Characteristic Curve')
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.savefig(self.fig_dir)
        
        precision, recall, thresholds = precision_recall_curve(self.true_labels, self.predict_scores)
        self.AP = average_precision_score(self.true_labels, self.predict_scores, average='macro', pos_label=1, sample_weight=None)
        plt.figure("P-R Curve")
        plt.title('Precision/Recall Curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.plot(recall,precision,label = 'Val AP = %0.3f' % self.AP)
        plt.savefig(self.ap_fig_dir)
        
        self.dic['roc_auc'] = self.roc_auc
        self.dic['ap'] = self.AP
    def writ_result(self):
        self.dic['manual'] = self.manual
        self.dic['correctly_predicted'] = self.correctly_predicted
        self.dic['missed'] = self.missed
        self.dic['total_predicted_positive'] = self.total_predicted_positive
        self.dic['Ineligible_spared_to_screen'] = self.Ineligible_spared_to_screen
        self.dic['Ineligible_spared_to_screen'] = self.Ineligible_spared_to_screen
        self.dic['Sensitivity'] = '%.3f' % (self.Sensitivity)
        self.dic['specificity'] = '%.3f' % (self.specificity)
        self.dic['work_load_saving'] = '%.1f%%' % (100*(self.tn+self.fn)/(self.fp+self.tn+self.fn+self.tp))
        df = pd.DataFrame(self.dic,index=[0])
        if os.path.exists(self.summary_file_dir) == False: 
            df.to_csv(self.summary_file_dir, encoding='utf_8_sig', mode='a', index=False, header = self.dic.keys())
        else:
            df.to_csv(self.summary_file_dir, encoding='utf_8_sig', mode='a', index=False, header =False)
        