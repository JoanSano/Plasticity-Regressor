import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import wandb
import warnings

class Network(nn.Module):
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
            input_batch = input_batch.to(self.args.device)
            target_batch = target_batch.to(self.args.device)
            
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
        wandb.init(project="Plasticity-Regressor", entity="joansano")
        wandb.config.update(self.args)
        tr_loader, val_loader = list(), list()
        for ep in tqdm(range(1, self.args.epochs+1)):
            # Loading training and validation data
            for domain in range(len(self.train_data)):
                tr_loader.append(DataLoader(self.train_data[domain], batch_size=self.args.batch, shuffle=True))
                if self.val_step:
                    val_loader.append(DataLoader(self.val_data[domain], batch_size=self.args.batch, shuffle=True))
            # Training   
            with torch.enable_grad():
                loss_tr = self.__epoch(tr_loader, backprop=True)
                wandb.log({"Batch Training Loss": loss_tr}, step=ep)
            # Validation
            if self.val_step:
                with torch.no_grad():
                    loss_val = self.__epoch(val_loader, backprop=False)
                    wandb.log({"Batch validation Loss": loss_val}, step=ep)
            # Live updating
            wandb.watch(self.network)

    def test(self, x):
        """ Generates a prediction of a given batch """
        with torch.no_grad():
            return self.network(x.to(self.args.device))

if __name__ == '__main__':
    pass