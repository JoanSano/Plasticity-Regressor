import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns

def euclidean_distance(pred, real):
    """ Computes the Euclidean Distance between a predicted and real graph. 
    Input:
        pred: batch of predicted graphs -- flattened
        real: batch of real targeted graphs -- flattened
    Output:
        mse: array of size (batch_size) with the average MSE for each graph 
        edge_mse: array of size (batch_size, features) with the MSE for each graph and edge
    """
    mse = nn.L1Loss()
    edge_se = nn.L1Loss(reduction='none')
    N, features = pred.shape
    ED, edge_ED = torch.zeros(N), torch.zeros((N, features))
    for i in range(N):
        ED[i] = mse(pred[i], real[i])
        edge_ED[i] = edge_se(pred[i], real[i])
    return ED, edge_ED

def plot_distances(metrics, path, subjects, show=False, fig_size=(10,6)):
    """ Plots metrics regarding prediction errors """   
    graph_errors = metrics[0].numpy()
    edge_errors = metrics[1]
    edge_avg = torch.mean(edge_errors, dim=0).numpy()
    edge_std = torch.std(edge_errors, dim=0).numpy()

    # Global prediction error distribution
    plt.figure(figsize=fig_size)
    sns.histplot(data=graph_errors, binwidth=.04)
    plt.xlabel("Graph Absolut Error")
    plt.savefig(path+'Graph_AbsDistance.png')
    if show:
        plt.show()
    plt.close()

    # Edge Error distribution
    plt.figure(figsize=fig_size)
    plt.subplot(1,2,1)
    sns.histplot(data=edge_avg)
    plt.xlabel("Edge Absolut Error")
    plt.subplot(1,2,2)
    sns.histplot(data=edge_avg, binwidth=.04)
    plt.xlim([0, 1])
    plt.xlabel("Edge Absolut Error")
    plt.savefig(path+'Edge_AbsDistance.png')
    if show:
        plt.show()
    plt.close()

    return torch.tensor(edge_avg), torch.tensor(edge_std)

if __name__ == '__main__':
    pass
