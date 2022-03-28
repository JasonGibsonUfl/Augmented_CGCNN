import pickle as pkl
import os
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data import CIFData,collate_pool,DataLoader

import torch
from torch.autograd import Variable
import torch.nn as nn

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import utils

from torch.optim.lr_scheduler import MultiStepLR

from tqdm import tqdm

class Test:
    def __init__(self,size=50,root_dir = './mldir/'):
        self.size = size
        self.model = None
        self.root_dir = root_dir

    def get_loaders(self,hl=False):
        if hl:
            #dataset_relaxed = CIFData(f'{self.root_dir}/relaxed_scaled')
            #dataset_unrelaxed = CIFData(f'{self.root_dir}/unrelaxed_scaled')
            dataset_relaxed = CIFData(f'{self.root_dir}/relaxed')
            dataset_unrelaxed = CIFData(f'{self.root_dir}/unrelaxed')
            dataset_unrelaxed = CIFData(f'{self.root_dir}/preston')

            self.relaxed_loader = DataLoader(dataset_relaxed, batch_size=len(dataset_relaxed),collate_fn=collate_pool,num_workers=0,shuffle=True,pin_memory=True)
            self.unrelaxed_loader = DataLoader(dataset_unrelaxed, batch_size=len(dataset_unrelaxed),collate_fn=collate_pool,num_workers=0,shuffle=True,pin_memory=True)
        else:
            dataset_train = CIFData(f'{self.root_dir}tune_{self.size}')
            dataset_val = CIFData(f'{self.root_dir}val_{self.size}')
            dataset_test = CIFData(f'{self.root_dir}test_{self.size}')

            self.train_loader = DataLoader(dataset_train, batch_size=len(dataset_train),collate_fn=collate_pool,num_workers=0,shuffle=True,pin_memory=True)
            self.val_loader = DataLoader(dataset_val, batch_size=len(dataset_val),collate_fn=collate_pool,num_workers=0,shuffle=False,pin_memory=True)
            self.test_loader = DataLoader(dataset_test, batch_size=len(dataset_test),collate_fn=collate_pool,num_workers=0,shuffle=False,pin_memory=True)

    def load_model(self,model_path,n_h = 8, h_fea_len = 64, n_convs = 3,dropout=False,org=True):
        self.model_path = model_path
        normalizer,model = utils.get_model(self.model_path,n_h = n_h, h_fea_len = h_fea_len, n_convs = n_convs,DO=dropout,org=org)
        self.normalizer = normalizer
        model.eval()
        self.model = model

    def run(self,name,n_h = 8, h_fea_len = 64, n_convs = 3,load=True,dropout=False):
        #name = f'Final_nh_{n_h}_h_fea_l_{h_fea_len}_convs_{n_convs}_size_med_dataset_2p'
        path = f'/blue/hennig/jasongibson/graphNN/cgcnn/runs/{name}model_best.pth.tar'
        if load:
            self.load_model(path,n_h,h_fea_len,n_convs,dropout=dropout)

        fig, ([ax1, ax2]) = plt.subplots(1, 2, figsize=(16,5),sharey = 'row')#,sharex='col')#, sharey='row')
        fig.subplots_adjust(hspace=.085, wspace = 0.085)

        x_relaxed,y_relaxed = utils.predict(self.model, self.normalizer,self.relaxed_loader)
        x_unrelaxed,y_unrelaxed = utils.predict(self.model, self.normalizer,self.unrelaxed_loader)
        
        return x_relaxed, y_relaxed, x_unrelaxed,y_unrelaxed

    def model_predict(self, h_fea_lens=[64], n_hs=[8], convs=[3],ds_name='2p',p_size='med',dropout=False):
        for hfl in h_fea_lens:
            for n_h in n_hs:
                for conv in convs:
                    name = f'Final_nh_{n_h}_h_fea_l_{hfl}_convs_{conv}_size_{p_size}_dataset_{ds_name}_tanh'
                    x_r,y_r,x_un,y_un = self.run(name,n_h = n_h, h_fea_len = hfl, n_convs = conv,dropout=dropout)
                    #print(self.model)
                    mae_r, rmse_r, r2_r = utils.eval_predictions(x_r, y_r)
                    mae, rmse, r2 = utils.eval_predictions(x_un, y_un)
                    std = np.std(x_un)
                    score = 1-rmse/std
                    print(f'{name}: RMSE[unrelaxed/relaxed] = {1000*rmse:.2f}/{1000*rmse_r:.2f},\t MAE = {1000*mae:.2f}/{1000*mae_r:.2f},\t std = {1000*std:.2f},\t score = {score:.2f},\t r2 = {r2:.4f}')

if __name__ == '__main__':
    tester = Test(root_dir='./mldir3/',size=25)
    #n = '/blue/hennig/jasongibson/graphNN/CGCNN-HD/Mg-Mn-O_database/org_cifs'
    tester.get_loaders(hl=True)
    #dataset_relaxed = CIFData(n)
    #tester.relaxed_loader = DataLoader(dataset_relaxed, batch_size=len(dataset_relaxed),collate_fn=collate_pool,num_workers=0,shuffle=True,pin_memory=True)
    #nr = '/blue/hennig/jasongibson/graphNN/CGCNN-HD/Mg-Mn-O_database/lattice_scaled'
    #dataset_unrelaxed = CIFData(nr)
    #tester.unrelaxed_loader = DataLoader(dataset_unrelaxed, batch_size=len(dataset_unrelaxed),collate_fn=collate_pool,num_workers=0,shuffle=True,pin_memory=True)
    #model_path = 'bests/CGCNN_4.pth.tar'
    model_path = 'bests/CGCNN.pth.tar'

    tester.normalizer,tester.model = utils.get_model(model_path,n_h = 6, h_fea_len = 64, n_convs = 3,DO=True)
    xr,yr,xur,yur = tester.run('CGCNN',6,64,3,False)
    mae_r, rmse_r, r2_r = utils.eval_predictions(xr, yr)
    mae, rmse, r2 = utils.eval_predictions(xur, yur)
    std = np.std(xur)
    score = 1-rmse/std
    print(xur)
    print(yur)
    print(f'{model_path}: RMSE[unrelaxed/relaxed] = {1000*rmse:.2f}/{1000*rmse_r:.2f},\t MAE = {1000*mae:.2f}/{1000*mae_r:.2f},\t std = {1000*std:.2f},\t score = {score:.2f},\t r2 = {r2:.4f}')
    #pkl.dump(xr,open('CGCNN_xr_7.pkl','wb'))
    #pkl.dump(yr,open('CGCNN_yr_7.pkl','wb'))
    #pkl.dump(xur,open('CGCNN_xur_7.pkl','wb'))
    #pkl.dump(yur,open('CGCNN_yur_7.pkl','wb'))


    model_path = 'bests/PCGCNN_4.pth.tar'
    model_path = 'bests/PCGCNN.pth.tar'

    tester.normalizer,tester.model = utils.get_model(model_path,n_h = 6, h_fea_len = 64, n_convs = 3,DO=True)
    xr,yr,xur,yur = tester.run('PCGCNN',6,64,3,False)
    mae_r, rmse_r, r2_r = utils.eval_predictions(xr, yr)
    mae, rmse, r2 = utils.eval_predictions(xur, yur)
    std = np.std(xur)
    score = 1-rmse/std
    print(xur)
    print(yur)
    print(f'{model_path}: RMSE[unrelaxed/relaxed] = {1000*rmse:.2f}/{1000*rmse_r:.2f},\t MAE = {1000*mae:.2f}/{1000*mae_r:.2f},\t std = {1000*std:.2f},\t score = {score:.2f},\t r2 = {r2:.4f}')
    #pkl.dump(xr,open('PCGCNN_xr_7.pkl','wb'))
    #pkl.dump(yr,open('PCGCNN_yr_7.pkl','wb'))
    #pkl.dump(xur,open('PCGCNN_xur_7.pkl','wb'))
    #pkl.dump(yur,open('PCGCNN_yur_7.pkl','wb'))
    '''
    model_path = 'bests/CGCNN_HD.pth.tar'
    tester.normalizer,tester.model = utils.get_model(model_path,n_h = 6, h_fea_len = 64, n_convs = 3,DO=True,org=False)
    xr,yr,xur,yur = tester.run('CGCNN_HD',6,64,3,False)
    mae_r, rmse_r, r2_r = utils.eval_predictions(xr, yr)
    mae, rmse, r2 = utils.eval_predictions(xur, yur)
    std = np.std(xur)
    score = 1-rmse/std
    print(f'{model_path}: RMSE[unrelaxed/relaxed] = {1000*rmse:.2f}/{1000*rmse_r:.2f},\t MAE = {1000*mae:.2f}/{1000*mae_r:.2f},\t std = {1000*std:.2f},\t score = {score:.2f},\t r2 = {r2:.4f}')
    pkl.dump(xr,open('CGCNN_HD_xr_6.pkl','wb'))
    pkl.dump(yr,open('CGCNN_HD_yr_6.pkl','wb'))
    pkl.dump(xur,open('CGCNN_HD_xur_6.pkl','wb'))
    pkl.dump(yur,open('CGCNN_HD_yur_6.pkl','wb'))

    model_path = 'bests/PCGCNN_HD.pth.tar'
    tester.normalizer,tester.model = utils.get_model(model_path,n_h = 6, h_fea_len = 64, n_convs = 3,DO=True,org=False)
    xr,yr,xur,yur = tester.run('PCGCNN_HD_2',6,64,3,False)
    mae_r, rmse_r, r2_r = utils.eval_predictions(xr, yr)
    mae, rmse, r2 = utils.eval_predictions(xur, yur)
    std = np.std(xur)
    score = 1-rmse/std
    print(f'{model_path}: RMSE[unrelaxed/relaxed] = {1000*rmse:.2f}/{1000*rmse_r:.2f},\t MAE = {1000*mae:.2f}/{1000*mae_r:.2f},\t std = {1000*std:.2f},\t score = {score:.2f},\t r2 = {r2:.4f}')
    pkl.dump(xr,open('PCGCNN_HD_xr_6.pkl','wb'))
    pkl.dump(yr,open('PCGCNN_HD_yr_6.pkl','wb'))
    pkl.dump(xur,open('PCGCNN_HD_xur_6.pkl','wb'))
    pkl.dump(yur,open('PCGCNN_HD_yur_6.pkl','wb'))
    '''
