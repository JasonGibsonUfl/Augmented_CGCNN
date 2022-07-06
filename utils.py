import os
import multiprocessing

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from model import CrystalGraphConvNet, ConvLayer,CrystalGraphConvNetDO
from data import *

import torch
from torch.autograd import Variable
import torch.nn as nn
from scipy.interpolate import interpn

from matplotlib.colors import Normalize

from random import sample

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from tqdm import tqdm
def zero_list(amount):
    return [[] for _ in range(amount)]


class mySequential(torch.nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


class MPCrystalGraphConvNet(CrystalGraphConvNet):
    """
    Same as CrystalGraphConvNet but instead of 1 gpu, each 
    convolutional layer gets a gpu. Create a crystal graph 
    convolutional neural network for predicting total
    material properties.
    """
    def __init__(
        self,
        orig_atom_fea_len,
        nbr_fea_len,
        atom_fea_len=64,
        n_conv=3,
        h_fea_len=128,
        n_h=2,
        classification=False,
    ):
        super(MPCrystalGraphConvNet, self).__init__(
            orig_atom_fea_len,
            nbr_fea_len,
            atom_fea_len,
            n_conv,
            h_fea_len,
            n_h,
            classification=False,
        )
        self.embedding
        self.seq2 = mySequential(
            self.conv_to_fc_softplus,
            self.conv_to_fc,
            self.dropout,
            self.conv_to_fc_softplus,
            *self.fcs,
        )
        self.fc_out

    def forward(self, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx):
        atom_fea = self.embedding(atom_fea)
        for i, conv_func in enumerate(self.convs):
            atom_fea = conv_func(atom_fea, nbr_fea, nbr_fea_idx)
        pool = self.pooling(atom_fea, crystal_atom_idx)
        seq2 = self.seq2(pool)
        return self.fc_out(seq2)


def get_model(model_path ,n_h = 6, h_fea_len = 64, n_convs = 3, org=True):
    """
    Initializes normalizer and model from a pretrained model.
    
    Parameters
    ----------
    model_path: str
        Path to saved model with .pth.tar exstension
    n_h: int
        Number of hidden layers 
    h_fea_len: int
        Width of each layer
    n_convs: int
        Number of convolutional layers
    org: Bool
      if true the CGCNN model is loaded if False the CGCNN-HD model is loaded    
    Returns
    -------
    normalizer: 
        Normalizer used to normalizer target property
    model: 
        pre-trained model
    """
    model = CrystalGraphConvNetDO(92, 41, n_conv=n_convs, h_fea_len=h_fea_len, n_h=n_h,org=org)
    model.load_state_dict(torch.load(model_path, map_location="cpu")["state_dict"], strict=False)

    state_dict = torch.load(model_path, map_location="cpu")
    normalizer = Normalizer(torch.tensor(0.25))
    m = state_dict["normalizer"]["mean"]
    std = state_dict["normalizer"]["std"]
    normalizer.mean = m
    normalizer.std = std
    return normalizer, model


def eval_predictions(x, y):
    """
    Computes metrics to evaluate model predictions.
    
    Parameters
    ----------
    x: list
        list of target/prediction property
    y: list
        list of target/prediction property        
    
    Returns
    -------
    mae: float
        Mean absolute error of prediction
    rmse: float
        Root mean squared error of predictions
    r2: float
        r2 value of prediction/target
    """
    mae = mean_absolute_error(y, x)*1000
    rmse = np.sqrt(mean_squared_error(y, x))*1000
    r2 = r2_score(y, x)
    return mae, rmse, r2



def density_scatter(
    x,
    y,
    ax=None,
    fig=None,
    xl=False,
    sort=False,
    bins=20,
    lab="",
    title="",
    **kwargs,
):
    """
    Produces nice plot of target vs prediction
    """
    def get_density_scatter_params(x,y):
        """
        Used to position text on plots
        """
        abs_min = min([min(x),min(y)])
        abs_max = max([max(x),max(y)])
        return abs_min, abs_max
    mae, rmse, r2 = eval_predictions(x, y)
    abs_min, abs_max = get_density_scatter_params(x,y)
    if ax is None:
        print('NONE')
        f, ax = plt.subplots()
    data, x_e, y_e = np.histogram2d(x, y, bins=bins, density=True)
    z = interpn(
        (0.5 * (x_e[1:] + x_e[:-1]), 0.5 * (y_e[1:] + y_e[:-1])),
        data,
        np.vstack([x, y]).T,
        method="splinef2d",
        bounds_error=False,
    )

    z[np.where(np.isnan(z))] = 0.0

    if sort:
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

    ax.scatter(x, y, c=z, **kwargs)
    ax.set_title(title, y=1.0, pad=-14, fontsize=14)
    ax.text(abs_min, abs_max, f"{lab}", fontsize=14, ha="left")

    ax.text(abs_max, abs_min+abs(abs_min-abs_max)*.14, f"R\N{SUPERSCRIPT TWO}: {r2:.4f}", fontsize=14, ha="right")
    ax.text(abs_max, abs_min+abs(abs_min-abs_max)*.07, f"MAE: {1000*mae:.2f} meV/atom", fontsize=14, ha="right")
    ax.text(abs_max, abs_min+abs(abs_min-abs_max)*.0, f"RMSE: {1000*rmse:.2f} meV/atom", fontsize=14, ha="right")
    
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.plot([abs_min, abs_max], [abs_min, abs_max], "--k")
    if xl:
        ax.set_xlabel(r"$E^{DFT}_f$ $\it(ev/atom)$", fontsize=15)
    return ax


def get_input(input, target):
    """
    return input for model
    
    Parameters
    ----------
    
    Returns
    -------
    """
    if torch.cuda.is_available():
        input_var = (
            Variable(input[0].cuda(non_blocking=True)),
            Variable(input[1].cuda(non_blocking=True)),
            input[2].cuda(non_blocking=True),
            [crys_idx.cuda(non_blocking=True) for crys_idx in input[3]])
        target_var = Variable(target.cuda(non_blocking =True))
    else:
        input_var = (Variable(input[0]), Variable(input[1]), input[2], input[3])
        target_var = Variable(target)
    return input_var, target_var

def predict(model, normalizer, loader):
    """
    Makes predictions on dataset
    
    Parameters
    ----------
    model: model.CrysyalGraphConvNet
        model used to make predictions
    normalizer: data.Normalizer
        normalizer used to train model
    loader: data.CIFData
        CIFData loader of prediction structures
        
    Returns
    -------
    tar: list
        list of DFT computed properties
    pred: list
        list of predicted properties
    """
    #model.cuda()
    model.eval()
    [pred, tar, test_ids] = zero_list(3)
    for i, (input, target, batch_cif_ids) in enumerate(loader):
        target_normed = normalizer.norm(target)
        input_var,_ = get_input(input,target_normed)
        pred += normalizer.denorm(model(*input_var)).view(-1).tolist()
        tar += target.view(-1).tolist()
        test_ids += batch_cif_ids
        #print(i,flush=True)
    return tar, pred


def run(per_model_path, unrelaxed=True, model=None, path="./cgcnn/data/LiGe"):
    normalizer, model = get_model(per_model_path, unrelaxed, model, path)
    tar, pred = predict(model, normalizer, loader)
    mae, rmse, r2 = eval_predictions(tar, pred)
    return tar, pred, mae, rmse, r2, normalizer, model
