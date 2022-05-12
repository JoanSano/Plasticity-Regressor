import numpy as np
import torch
import torch.nn as nn
import matplotlib.pylab as plt
import seaborn as sns

class BayesianWeightedLoss(nn.Module):
    def __init__(self, anat_prior):
        super().__init__()
        self.prior = anat_prior
        self.mse = nn.MSELoss()

    def forward(self, output, target):
        # This loss function can be tweeked to include topological features, cosine similarity
        #   or even maximizin the KL divergence with respect to the control group...?
        posterior = output * 0
        for t in range(output.shape[0]):
            posterior[t] = torch.mul(output[t], self.prior)
        return self.mse(posterior, target) 

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
