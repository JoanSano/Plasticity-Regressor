import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

from models.methods import Network
from models.networks import LinearRegres, NonLinearRegres
from utils.data import check_path, graph_dumper, two_session_graph_loader
from utils.graphs import GraphFromCSV, create_anat_prior, load_anat_prior
from utils.paths import get_subjects, get_info

parser = argparse.ArgumentParser()
# General settings
parser.add_argument('-D', '--device', type=str, default='cuda', help="Device in which to run the code")
parser.add_argument('-F', '--folder', type=str, default='results', help="Results directory")
parser.add_argument('-M', '--model', type=str, default='model', help="Trained model name")

# Data specs
parser.add_argument('-S', '--split', type=int, default=20, help="Training and testing splitting")
parser.add_argument('-R', '--rois', type=int, default=170, help="Number of ROIs to use")
parser.add_argument('-A', '--augment', type=int, default=1, help="Data augmentation factor")
parser.add_argument('-V', '--validation', type=bool, default=False, help="Add validation step")
parser.add_argument('-P', '--prior', type=bool, default=False, help="Load available prior")

# Machine-learning specs
parser.add_argument('-E', '--epochs', type=int, default=100, help="Number of epochs to train")
parser.add_argument('-B', '--batch', type=int, default=4, help="Batch size")
parser.add_argument('-RE', '--regressor', type=str, default='linear', choices=['linear','nonlinear'], help="Type of regression")
args = parser.parse_args()

if __name__ == '__main__':
    if args.device == 'cuda':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else: 
        args.device = 'cpu'
    print("Code running on", args.device)

    # Relevant paths
    folder = args.folder+'_'+args.model+'/'
    check_path(folder)
    graph_path = check_path(folder+'csv/')
    png_path = check_path(folder+'png/')
    flat_path = check_path(folder+'flat/')

    # Loading training data
    data, subjects = two_session_graph_loader('data/', rois=args.rois, augmentation=args.augment)
    # Creating/loading priors
    if args.prior:
        prior, mean_connections = load_anat_prior('data/')
    else:
        prior, mean_connections = create_anat_prior('data/', save=True)

    # Defining the objects
    if args.regressor.lower() == 'linear':
        regres = LinearRegres(args.rois)
    else:
        regres = NonLinearRegres(args.rois)
    mse = nn.MSELoss()
    sgd = optim.SGD(regres.parameters(), lr=0.01)
    model = Network(regres, sgd, mse, data, args)

    # Training the model and saving
    model.train()
    torch.save(regres, folder+args.model+'.ckpt')

    # Test with the same data and save the results
    prediction = model.test(data[0])
    graph_dumper(graph_path, prediction.cpu(), subjects, suffix='predicted')
    graph_files = get_subjects(folder, session='csv')
    for f in graph_files:
        _, _, _, name = get_info(f)
        sg = GraphFromCSV(f, name, png_path)
        sg.unflatten_graph(to_default=True, save_flat=True)
        sg.process_graph(log=False, reshuffle=True)
        Path(png_path+name+'_flatCM.csv').rename(flat_path+name+'_flatCM.csv')