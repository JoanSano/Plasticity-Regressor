import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import matplotlib.pylab as plt
import seaborn as sns
from scipy.spatial.distance import jensenshannon

from utils.data import unflatten_data

class BayesianWeightedLoss(nn.Module):
    def __init__(self, anat_prior, type='mse'):
        super().__init__()
        self.prior = anat_prior
        if type == 'mse':
            self.loss = nn.MSELoss()
        elif type == 'huber':
            self.loss = nn.HuberLoss()
        else:
            raise ValueError('Loss function not implemented')

    def forward(self, output, target):
        # This loss function can be tweeked to include topological features, cosine similarity
        #   or even maximizin the KL divergence with respect to the control group...?
        posterior = output * 0
        for t in range(output.shape[0]):
            posterior[t] = torch.mul(output[t], self.prior)
        return self.loss(posterior, target) 

class PCC(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, output, target):
        """
        NOT SURE THIS METRIC MAKES SENSE IN THIS CASE!!!
        Inputs:
            output: network output tensor of size (N, Features) N>1! 
            target: tensor of size (N, Features)
        Outputs:
            cc: correlation coefficient of each feature - tensor of size (Features,)
            mean_cc: mean correlation coefficient - scalar 
        """

        vx = output - torch.mean(output, dim=self.dim)
        vy = target - torch.mean(target, dim=self.dim)
        cc = torch.sum(vx * vy, dim=self.dim) / (torch.sqrt(torch.sum(vx ** 2, dim=self.dim)) * torch.sqrt(torch.sum(vy ** 2, dim=self.dim)))
        mean_cc = torch.mean(cc)
        std_cc = torch.std(cc)
        return cc, mean_cc, std_cc

class CosineSimilarity(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, output, target):
        """
        Inputs:
            output: network output tensor of size (N, Features)
            target: tensor of size (N, Features)
        Outputs:
            cs: cosine similarity of each feature vector - tensor of size (N,)
            mean_cs: mean cosine similarity - scalar 
        """

        cos = nn.CosineSimilarity(dim=self.dim)
        cs = cos(output, target)
        mean_cs = torch.mean(cs)
        std_cs = torch.std(cs)
        return cs, mean_cs, std_cs

def degree_distribution(flattened, rois, maximum_degree=200, d_dg=1.):
    """
    Returns the probability distribution and the degrees in the graph. 
    Inputs:
        flattened: flattened graph
        rois: number of nodes
        maximum_degree: (int) maximum degree to which spans the probability
        d_dg: degree interval upon which the probability refers to (float)
    Outputs:
        prob: probability distribution of each degree in the network (numpy array)
        dgs: degrees present in the network until maximum_degree
    """
    degree_prob = np.zeros((int(maximum_degree//d_dg),))
    dgs = np.arange(0, maximum_degree+1)
    adj = np.array(unflatten_data(flattened, rois=rois, norm=False)[0], dtype=np.float64)
    D_G = [jj for _,jj in nx.from_numpy_array(adj).degree(weight='weight')]
    #probs = (np.bincount([jj for _,jj in D_G])/rois)
    #dgs = np.unique([jj for _,jj in D_G])
    for d in range(maximum_degree):
        d_inf, d_sup = dgs[d], dgs[d+1]
        degree_prob[d] = np.sum((D_G>d_inf)*(D_G<d_sup))
    return degree_prob, dgs

def KL_JS_divergences(input, target, rois, eps=1e-8):
    """ Computes the KL and JS Divergences between two degree distributions.
    Input:
        input: degree distribution of the input graph
        target: degree distribution of the target graph
        rois: number of nodes in the graph (to be used in the degree computation)
        eps: float to avoid log(0)
    Output:
        KL: divergence (torch scalar) 
        JS: divergence (torch scalar)
    """

    input_degree, _ = degree_distribution(input, rois)
    target_degree, _ = degree_distribution(target, rois)
    kl = np.sum(target_degree*np.log(target_degree+eps) - target_degree*np.log(input_degree+eps))
    js = jensenshannon(input_degree, target_degree)
    return kl, js 

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
