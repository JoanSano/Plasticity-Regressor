import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import wandb
import warnings

from models.networks import LinearRegres, NonLinearRegres
from models.metrics import BayesianWeightedLoss

class Model(nn.Module):
    """ Generic object with methods common to different networks. 
    Shoud be usable for other frameworks. 
    """
    def __init__(self, network, optimizer, criterion, data, args):
        super().__init__()
        self.network = network.to(args.device)
        self.optimizer= optimizer
        self.criterion = criterion
        self.data = data 
        self.args = args

        # Data split between train and validation
        if args.validation:
            try:
                self.train_data, self.val_data = self.__split()
                self.val_step = True
            except:
                self.train_data = data
                self.val_step = False
                warnings.warn('The split percentatge does not allow for validations steps')
        else:
            self.train_data = data
            self.val_step = False

    def __split(self):
        """ Prepare data to feed the network.
        Inputs:
            None. The data is a list containing (input, target) with each 'domain' being a tensor
            of size (N, Features).
        Output:
            train_data: tuple with train timepoints
            val_data: tuple with validation timepoints 
        """
        N = self.data[0].shape[0]
        tr_N = (100-self.args.split)*0.01
        tr_indices, val_indices = train_test_split(range(N),train_size=tr_N)
        train_data, val_data = list(), list()
        for p in range(len(self.data)):
            train_data.append(torch.index_select(self.data[p], 0, torch.tensor(tr_indices)))
            val_data.append(torch.index_select(self.data[p], 0, torch.tensor(val_indices)))
        return tuple(train_data), tuple(val_data) 
        
    def __epoch(self, loader, backprop):
        epoch_loss, num_batches = [0, 0]
        for input_batch, target_batch in zip(loader[0], loader[1]):
            input_batch = input_batch.double().to(self.args.device)
            target_batch = target_batch.double().to(self.args.device)
            
            prediction = self.network(input_batch)
            loss = self.criterion(prediction, target_batch)

            if backprop: # If loss backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        return epoch_loss/num_batches

    def train(self):
        """ Trains the model. If specified and if possible also doing validation."""
        if self.args.wandb: 
            wandb.init(project="Plasticity-Regressor", entity="joansano")
            wandb.config.update(self.args)
        tr_loader, val_loader = list(), list()
        for ep in range(1, self.args.epochs+1):
            # Loading training and validation data
            for domain in range(len(self.train_data)):
                tr_loader.append(DataLoader(self.train_data[domain], batch_size=self.args.batch, shuffle=True))
                if self.val_step:
                    val_loader.append(DataLoader(self.val_data[domain], batch_size=self.args.batch, shuffle=True))
            # Training   
            with torch.enable_grad():
                loss_tr = self.__epoch(tr_loader, backprop=True)
                if self.args.wandb: 
                    wandb.log({"Batch Training Loss": loss_tr}, step=ep)
                else:
                    print("Epoch {}/{}: Training loss: {}".format(ep, self.args.epochs, loss_tr))
            # Validation
            if self.val_step and (ep%self.args.val_freq==0):
                with torch.no_grad():
                    loss_val = self.__epoch(val_loader, backprop=False)
                    if self.args.wandb: 
                        wandb.log({"Batch validation Loss": loss_val}, step=ep)
                    else:
                        print("Validation loss: {}".format(loss_val))
            # Live updating
            if self.args.wandb: 
                wandb.watch(self.network)

    def test(self, x):
        """ Generates a prediction of a given batch """
        with torch.no_grad():
            return self.network(x.double().to(self.args.device))

def return_specs(args, prior=None):
    """
    Returns the object necessary to build the model
    Inputs:
        args: argparser containing the input arguments
        prior: (optional) Necessary to build the bayesian weighted loss function
    Returns:
        regres: torch network to train (python object)
        loss: torch loss function used to train the network (python object)
        sgd: torch object used to train train the network (python object)
    """
    if args.regressor == 'linear': 
        regres = LinearRegres(args.rois)
    elif args.regressor == 'nonlinear':
        regres = NonLinearRegres(args.rois)
    else:
        raise ValueError("Regressor not implemented")

    if args.loss == 'bayes_mse':
        loss = BayesianWeightedLoss(prior)
    elif args.loss == 'huber':
        loss = nn.HuberLoss()
    else:
        raise ValueError("Loss function not implemented")

    if args.optimizer == 'sgd': 
        sgd = optim.SGD(regres.parameters(), lr=args.learning_rate)
    else:
        raise ValueError("Optimizer not implemented")

    return regres, loss, sgd

if __name__ == '__main__':
    pass