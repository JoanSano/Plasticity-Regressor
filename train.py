import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

from models.methods import Network
from models.networks import LinearRegres, NonLinearRegres
from models.metrics import euclidean_distance, plot_distances
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
    metrics_csv_path = check_path(folder+'metrics/numerical/')
    metrics_fig_path = check_path(folder+'metrics/figures/')

    # Training data
    data, subjects = two_session_graph_loader('data/', rois=args.rois, augmentation=args.augment)

    # Creating or loading priors
    if args.prior:
        prior, mean_connections = load_anat_prior('data/')
    else:
        prior, mean_connections = create_anat_prior('data/', save=True)
        sg = GraphFromCSV('data/prior.csv', 'prior', 'data/')
        sg.unflatten_graph(to_default=True, save_flat=True)
        sg.process_graph(log=False, reshuffle=True, bar_label='Probability of Connection')

    # Defining the objects and training the model
    if args.regressor.lower() == 'linear':
        regres = LinearRegres(args.rois)
    else:
        regres = NonLinearRegres(args.rois)
    mse = nn.MSELoss()
    sgd = optim.SGD(regres.parameters(), lr=0.01)
    model = Network(regres, sgd, mse, data, args)
    train_predictions, val_predictions = model.train()
    preds = torch.cat((train_predictions, val_predictions), dim=0).cpu()
    torch.save(regres, folder+args.model+'.ckpt')

    # Weighting predictions with anatomical priors
    for t in range(preds.shape[0]):
        preds[t] = torch.mul(preds[t], prior)

    # Saving Predicted Outputs
    print("================")
    print("Saving Outputs ...")
    graph_dumper(graph_path, preds, subjects, suffix='predictions')
    graph_files = get_subjects(folder, session='csv')
    for f in graph_files:
        _, _, _, name = get_info(f)
        sg = GraphFromCSV(f, name, png_path)
        sg.unflatten_graph(to_default=True, save_flat=True)
        sg.process_graph(log=False, reshuffle=True)
        Path(png_path+name+'_flatCM.csv').rename(flat_path+name+'_flatCM.csv')

    print("================")
    print("Computing Error Metrics ...")
    # Compute metrics to asses the reliability of the generated graphs
    mse, edge_mse = euclidean_distance(preds, data[1])
    graph_dumper(metrics_csv_path, mse, ["Graphs_Error"], suffix='Mean')
    graph_dumper(metrics_csv_path, edge_mse, subjects, suffix='Edge_Distance')
    edge_avg, edge_std = plot_distances((mse, edge_mse), metrics_fig_path, subjects)
    graph_dumper(metrics_csv_path, torch.stack((edge_avg, edge_std), dim=0), ["Edge_Error", "Edge_Std"], suffix='avg')
    sg = GraphFromCSV(metrics_csv_path+"Edge_Error_avg.csv", "Edge_Error_avg", metrics_fig_path)
    sg.unflatten_graph(to_default=True, save_flat=False)
    sg.process_graph(log=False, reshuffle=True, bar_label='Distance')
    sg = GraphFromCSV(metrics_csv_path+"Edge_Std_avg.csv", "Edge_Std_avg", metrics_fig_path)
    sg.unflatten_graph(to_default=True, save_flat=False)
    sg.process_graph(log=False, reshuffle=True, bar_label='Standard Deviation')
    error_files = get_subjects(folder+'metrics/', session='numerical', subject_ID="sub")
    for f in error_files:
        _, _, _, name = get_info(f)
        sg = GraphFromCSV(f, name, metrics_fig_path)
        sg.unflatten_graph(to_default=True, save_flat=False)
        sg.process_graph(log=False, reshuffle=True, bar_label='Distance')

    print("=====Logs=======")
    print("Mean Absolut Error of the Regression:", torch.mean(mse))
    print("Mean STD of the Absolut Error:", torch.std(mse))
    
