# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 14:28:26 2018

@author: Administrator
"""

import numpy as np

sample = [(0,1),(1,1),(2,1),(3,-1),(4,-1),(5,-1),(6,1),(7,1),(8,1),(9,-1)]

class Class2X1Ada(object):
    def __init__(self,data):
        self.data = np.array([item[0] for item in data])
        self.label = set([item[1] for item in data])
        self.ground_truth = np.array([item[1] for item in data])
        self.data_num = len(data)
        cut_set_forw = list(set(self.data))[1:] 
        cut_set_back = list(set(self.data))[0:-1]        
        self.cut_set = [(x+y)/2 for (x,y) in zip(cut_set_forw,cut_set_back)]
        
    def init_weights(self):
        weights = np.ones((self.data_num))/self.data_num
        return weights
    
    @staticmethod
    def _cut_off_lossfunc(ground_truth,predict_truth):
        return np.sum(ground_truth != predict_truth)
    
    @staticmethod
    def _cut_off_loss_weights(weights,ground_truth,predict_truth):
        return np.sum((ground_truth!=predict_truth)*weights)
    
    def _cut_off_value(self,weights):
        cut_value,iter_loc = 0,0
        loss_min = 99999
        while iter_loc< self.data_num-1:
            cut_value_iter = self.cut_set[iter_loc]
            predict_truth_less = np.where(self.data<cut_value_iter,1,-1)
            predict_truth_more = np.where(self.data>cut_value_iter,1,-1)
            loss_less = self._cut_off_loss_weights(weights,self.ground_truth,predict_truth_less)
            loss_more = self._cut_off_loss_weights(weights,self.ground_truth,predict_truth_more)
            loss = np.min((loss_less,loss_more))
            symbol = ['<','>'][np.argmin((loss_less,loss_more))]
            #print('-------------------------')
            #print('Step {}'.format(iter_loc))
            #print('Loss is {}.'.format(loss/self.data_num))
            #print('Cut value is {}.'.format(cut_value_iter))
            if loss<loss_min:
                loss_min = loss
                cut_value = cut_value_iter
                print('Update cut value to {}.'.format(cut_value))
            iter_loc+=1
                #print('-------------------------')
        return cut_value,loss_min,symbol 
    
    @staticmethod    
    def _compute_classfier_para(loss):
        paras = 0.5*np.log((1-loss)/loss)
        return paras
    
    def _update_weights_on_data(self,pre_weights,paras,cut_value,symbol):
        if symbol == '<':
            predict_truth = np.where(self.data<cut_value,1,-1)
        elif symbol == '>':
            predict_truth = np.where(self.data>cut_value,1,-1)
        
        transi = np.exp(-self.ground_truth*predict_truth*paras)
        Zm = np.sum(pre_weights*transi)
        
        post_weights = pre_weights/Zm*transi
        return post_weights
    
    def TrainAdaboost(self,NumOfClassifer = 3):
        print('Init weights')
        weights = self.init_weights()
        model = []
        train = []
        for i in range(NumOfClassifer):
            print('Step {}'.format(i+1))
            cut_value,error,symbol = self._cut_off_value(weights)
            print('Cut off is {},error is {}.'.format(cut_value,error))
            am = self._compute_classfier_para(error)
            print("Sub-model-{}'s para is {}.".format(i+1,am))
            model_info = {'cut_off':cut_value,'am':am,'symbol':symbol}
            train_info = {'weights':weights,'loss':error}
            model.append(model_info)
            train.append(train_info)
            weights = self._update_weights_on_data(weights,am,cut_value,symbol)
        print('Train succeed.')
        self.model = model
      
    def ValidateOnTrainset(self):
        predict = np.zeros(self.data_num)
        for sub_model in self.model:
            if sub_model['symbol']=='<':
                predict += sub_model['am']*np.where(self.data<sub_model['cut_off'],1,-1)
            elif sub_model['symbol']=='>':
                predict += sub_model['am']*np.where(self.data>sub_model['cut_off'],1,-1)
        predict = np.sign(predict)
        accuracy = np.sum(predict == self.ground_truth)
        return accuracy/self.data_num    

