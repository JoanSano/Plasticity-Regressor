import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import LeaveOneOut
from pathlib import Path
import pandas as pd

from models.methods import Network
from models.networks import LinearRegres, NonLinearRegres
from models.metrics import euclidean_distance, plot_distances, PCC, BayesianWeightedLoss, CosineSimilarity
from utils.data import check_path, graph_dumper, two_session_graph_loader, prepare_data
from utils.graphs import GraphFromCSV, create_anat_prior, load_anat_prior
from utils.paths import get_subjects, get_info

parser = argparse.ArgumentParser()
# General settings
parser.add_argument('-D', '--device', type=str, default='cuda', help="Device in which to run the code")
parser.add_argument('-F', '--folder', type=str, default='results', help="Results directory")
parser.add_argument('-M', '--model', type=str, default='model', help="Trained model name")
parser.add_argument('-W', '--wandb', type=bool, default=False, help="Whether to use wandb")

# Data specs
parser.add_argument('-S', '--split', type=int, default=20, help="Training and testing splitting")
parser.add_argument('-R', '--rois', type=int, default=170, help="Number of ROIs to use")
parser.add_argument('-A', '--augment', type=int, default=1, help="Data augmentation factor")
parser.add_argument('-V', '--validation', type=bool, default=False, help="Add validation step")
parser.add_argument('-P', '--prior', type=bool, default=False, help="Load available prior")

# Machine-learning specs
parser.add_argument('-E', '--epochs', type=int, default=2, help="Number of epochs to train")
parser.add_argument('--val_freq', type=int, default=5, help="Number of epochs between validation steps")
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

    # Preparing data
    (CONTROL, CON_subjects), (data, PAT_subjects), (PAT_1session, PAT_1session_subjects) = prepare_data(
    'data/', dtype=torch.float64, rois=170, norm=False, flatten=True, del_rois=None #[35,36,81,82]
    )

    # Creating or loading priors
    if args.prior:
        prior, mean_connections = load_anat_prior('data/')
    else:
        prior, mean_connections = create_anat_prior(CONTROL, 'data/', save=True, rois=170)
        sg = GraphFromCSV('data/prior.csv', 'prior', 'data/')
        sg.unflatten_graph(to_default=True, save_flat=True)
        sg.process_graph(log=False, reshuffle=True, bar_label='Probability of Connection')

    # Cross-Validation
    CV = LeaveOneOut()
    N_folds = CV.get_n_splits(data[0])
    CV_summary = pd.DataFrame(columns=['Subject', 'BayesMSE', 'MAE', 'PCC', 'CosineSimilarity'])
    if args.regressor.lower() == 'linear': 
        final_regres = LinearRegres(args.rois)
    else:
        final_regres = NonLinearRegres(args.rois)
    final_model = final_regres.state_dict()

    for fold, (train_index, test_index) in enumerate(CV.split(data[0])):
        print("=========================================")
        print("Fold number {} out of {} \n".format(fold+1, N_folds))

        input_train, input_test = data[0][train_index], data[0][test_index]
        target_train, target_test = data[1][train_index], data[1][test_index]
        data_fold = (input_train, target_train)
        subject = PAT_subjects[test_index[0]]

        # Defining the objects
        if args.regressor.lower() == 'linear': 
            regres = LinearRegres(args.rois)
        else:
            regres = NonLinearRegres(args.rois)
        mse = BayesianWeightedLoss(prior) #nn.MSELoss()
        sgd = optim.SGD(regres.parameters(), lr=0.01)
        model = Network(regres, sgd, mse, data_fold, args)

        # Training and testing
        model.train()
        pred_LOO = model.test(input_test).cpu()

        # Metrics
        test_mse = F.mse_loss(pred_LOO, target_test)
        test_mae = F.l1_loss(pred_LOO, target_test)
        _, pcc, _ = PCC().forward(pred_LOO, target_test)
        _, cs, _ = CosineSimilarity().forward(pred_LOO, target_test)
        CV_summary.loc[len(CV_summary.index)] = [subject, test_mse.item(), test_mae.item(), pcc.item(), cs.item()]

        # Creating the final model
        for key in final_model.keys():
            if fold==0:
                final_model[key] = regres.state_dict()[key]
            else:
                final_model[key] += regres.state_dict()[key]/N_folds
    
    # Saving the mean model
    final_regres.load_state_dict(final_model)
    torch.save(final_regres, folder+args.model+'.ckpt') 
    # TOMORROW: 
    # # 1 - check that the mean is correctly done
    # # 2 - prepare Hippo 
    # # 3 - train
    # # 4 - graphical results and figures and tables
    # # 5 - continue writing 

    # Saving performance evaluation
    CV_summary.to_csv(folder+args.model+'_performance.tsv', sep='\t', index=False)

    # To average the models #
    # https://discuss.pytorch.org/t/average-each-weight-of-two-models/77008

    # Saving Predicted Outputs
    """ 
    print("================")
    print("Saving Outputs ...")
    graph_dumper(graph_path, preds, PAT_subjects, suffix='preds')
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
    graph_dumper(metrics_csv_path, edge_mse, PAT_subjects, suffix='Edge_Distance')
    edge_avg, edge_std = plot_distances((mse, edge_mse), metrics_fig_path, PAT_subjects)
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
    """
    
