import numpy as np
import torch
from .paths import *
from subprocess import Popen, STDOUT, PIPE
import pandas as pd

def random_graph_gen(size=50, sample_size=1, states=[-1, 1], to_torch=False):
    """ Generates sample_size number of random graphs.
    Inputs:
        size: Number of ROIs
        sample_size: Total number of graphs
        states: States from which to choose the values of each node/edge
        to torch: Return the graphs as a tensor
    Outputs:
        graphs: The generated graphs
    """
    graphs = np.random.choice(states, size=(sample_size,size,size))
    if to_torch:
        graphs = torch.tensor(graphs)
    return graphs    

def graph_dumper(data_path, graphs, subject_list, suffix='evolved'):
    """ Dumps a tensor with a given number of graphs to a list of csv files.
    Inputs:
        data_path: path to which the csv files will be dumped
        graphs: tensor/array of graphs (Be careful with the dimensions of this object)
        subject_list: List of names that map to the graphs
        suffix: string to add before the extension
    Output: 
        None
    """
    for i, sub in enumerate(subject_list):
        name = sub + '_'+suffix+'.csv'
        gr = graphs[i].numpy().reshape((1,graphs.shape[-1]))
        dataframe = pd.DataFrame(data=gr.astype(float))
        dataframe.to_csv(data_path+name, sep=',', header=False, float_format='%.6f', index=False)

def common_subjects(data_path, sessions='*', subject_ID='*'):
    """ Finds the common subjects between two sessions """
    # Loading files
    files = list()
    for ses in sessions:
        files.extend(get_subjects(data_path, ses, subject_ID))
    # Creating lists
    subjects_pre = set()
    subjects_post = list()
    for f in files:
        _, session, patient, _ = get_info(f)
        if 'preop' in session:
            subjects_pre.add(patient)
        else:
            subjects_post.append(patient)
    # Finding intersection
    return subjects_pre.intersection(subjects_post)

def two_session_graph_loader(data_path, rois=170, augmentation=3, mu=0, sigma=1, sessions=['preop', 'postop'], subject_ID='*', norm=False):
    """
    Loads data that is available in two sessions. Training data is augmented as many times as 'augmentation' and lognormal noise is added.
    """
    if augmentation == 1:
        noise = False
    else:
        noise = True

    features = int(rois*(rois-1)/2)
    subjects = common_subjects(data_path, sessions=sessions, subject_ID=subject_ID)
    tr_samples = len(subjects)
    pre_cond, post_cond = np.zeros((tr_samples*augmentation, features)), np.zeros((tr_samples*augmentation, features))

    noise_gen = np.random.default_rng()
    for i, pat in enumerate(subjects):
        output = Popen(f"find {data_path if len(data_path)>0 else '.'} -wholename *{pat}*.csv", shell=True, stdout=PIPE)
        files = str(output.stdout.read()).strip("b'").split('\\n')[:-1]
        if 'preop' in files[0]:
            preop_graph = pd.read_csv(files[0], delimiter=',', header=None).values[0, :features]
            postop_graph = pd.read_csv(files[1], delimiter=',', header=None).values[0, :features]
        else:
            preop_graph = pd.read_csv(files[1], delimiter=',', header=None).values[0, :features]
            postop_graph = pd.read_csv(files[0], delimiter=',', header=None).values[0, :features]
       
        for aug in range(augmentation):
            pre_cond[i+aug*tr_samples,:] = np.log1p(preop_graph) + noise*noise_gen.lognormal(mu, sigma, size=preop_graph.shape)
            post_cond[i+aug*tr_samples,:] = np.log1p(postop_graph) + noise*noise_gen.lognormal(mu, sigma, size=postop_graph.shape)

            if norm:
                pre_cond[i+aug*tr_samples,:] = pre_cond[i+aug*tr_samples,:]/np.max(pre_cond[i+aug*tr_samples,:])
                post_cond[i+aug*tr_samples,:] = post_cond[i+aug*tr_samples,:]/np.max(post_cond[i+aug*tr_samples,:])
    
    return (torch.tensor(pre_cond, dtype=torch.float64), torch.tensor(post_cond, dtype=torch.float64)), subjects

def single_session_graph_loader(data_path, session, rois=170, augmentation=3, mu=0, sigma=1, subject_ID='*', norm=False):
    pass

def subset(data, cut=1):
    """ Returns a subset of the data """
    return data[:1]

if __name__ == '__main__':
    pass