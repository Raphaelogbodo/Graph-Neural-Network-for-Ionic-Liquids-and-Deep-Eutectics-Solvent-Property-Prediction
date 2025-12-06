import requests
import os, sys, re
from pathlib import Path
import pandas
import pandas as pd
import numpy as np
from scipy import stats
import pubchempy as pcp
from mordred import Calculator, descriptors
import statistics
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, PowerTransformer, QuantileTransformer
import matplotlib.pyplot as plt
import torch
import torch_geometric
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d, Dropout, Sequential, ReLU, ModuleList
from torch_geometric.data import Data, DataLoader, InMemoryDataset, Dataset, Batch
from torch_geometric.nn import NNConv, GCNConv, MFConv, GATConv, CGConv, GraphConv, GINConv
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool, GlobalAttention, TopKPooling, global_mean_pool
from torch_geometric.nn.aggr import AttentionalAggregation
from torch_geometric.nn import GraphNorm, LayerNorm
from torchmetrics import R2Score, MeanAbsoluteError, MeanAbsolutePercentageError
from permetrics.regression import RegressionMetric
import plotly.graph_objects as go
import yaml
from tqdm import tqdm
import random
import optuna
import ilthermopy as ilt
import py3Dmol
from openbabel import openbabel
from openbabel import pybel
from rdkit import Chem, rdBase
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem
from io import StringIO
from contextlib import redirect_stderr
import logging
pd.plotting.register_matplotlib_converters()
pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns',None)
pd.set_option('display.max_colwidth', None)
from utils import *
from architecture import *


def prepare_system():
    device = get_device_and_torch_info()
    torch_geometric.seed_everything(42)
    return device

def prepare_pandas_datasets(params, transformer=None, cond_transformer=None):
    
    if params['WHICH_LIQUID'] == 'il':
        dataset = prepare_datasets(params)
    elif params['WHICH_LIQUID'] == 'des':
        dataset = prepare_datasets_des(params)
    if params['CLEAN']:
        cond_names = renaming_and_cleaning(params, dataset)
    if params['WHICH_LIQUID'] == 'il':
        if logging:
            logging.info(f'Non-duplicated ILs: {len(list(set(dataset.smile.values)))}')
        else:
            print(f'Non-duplicated ILs: {len(list(set(dataset.smile.values)))}')
    elif params['WHICH_LIQUID'] == 'des':
        if logging:
            logging.info(f'Non-duplicated DESs: {len(list(set(dataset.smile.values)))}')
        else:
            print(f'Non-duplicated DESs: {len(list(set(dataset.smile.values)))}')
    logging.info('Cleaning dataset')
    remove_unwanted_liquids(dataset, False)
    if params['VERBOSE']:
        dataset['y'].plot.hist(label='y original')
        plt.legend(loc='best')
    original_dataset_len = len(dataset)
    if logging:
        logging.info(f"{dataset['y'].skew() = }")
        logging.info(f"{original_dataset_len = }")
        logging.info('Removing outliers from dataset')
    else:
        print(f"{dataset['y'].skew() = }")
        print(f"{original_dataset_len = }")
        print('Removing outliers from dataset')
    remove_outliers(dataset)
    if params['VERBOSE']:
        if logging:
            logging.info(f"{dataset['y'].skew() = }")
            logging.info(f"{len(dataset) / original_dataset_len * 100 = }")
        else:
            print(f"{dataset['y'].skew() = }")
            print(f"{len(dataset) / original_dataset_len * 100 = }")
        dataset['y'].plot.hist(label='y removed outliers')
        plt.legend(loc='best')
    if params['FEATURE_TRANSFORM'] and params['FEATURE_TRANSFORM_METHOD'] is not None:
        if transformer:
            if params['WHICH_LIQUID'] == 'il':
                transformer.fit(dataset[['y']])
                dataset['y'] = transformer.transform(dataset[['y']])
                transformer.save()
            elif params['WHICH_LIQUID'] == 'des':
                transformer.fit(dataset[['y']])
                #transformer.load()
                dataset['y'] = transformer.transform(dataset[['y']])
                transformer.save()
    if params['FEATURE_TRANSFORM'] and params['VERBOSE']:
        dataset['y'].plot.hist(label='y transformed')
        plt.legend(loc='best')
    if params['WHICH_LIQUID'] == 'il':
        plt.savefig(os.path.join(Figures, f'{params["TARGET_FEATURE_NAME"]}_distributions.png'), dpi=600)
    elif params['WHICH_LIQUID'] == 'des':
        plt.savefig(os.path.join(Figures, f'des_{params["TARGET_FEATURE_NAME"]}_distributions.png'), dpi=600)
    y_dataset_min = dataset['y'].min()
    y_dataset_max = dataset['y'].max()
    dataset.reset_index(inplace = True, drop = True)
    if params['CREATE_RDKITMOL_COL']:
        dataset['mol'] = dataset['smile'].apply(lambda x: Chem.MolFromSmiles(x) if x else np.nan)
        dataset = dataset.dropna().reset_index(drop=True)
    numbered = [i for i in range(1, len(dataset)+1)]
    dataset['numbered'] = np.array(numbered)
    if params['TESTSET_FROM_DATASET'] == True:
        dataset_train, dataset_test = stratified_split(params, dataset)
        if params['SCALE_CONDITIONS']:
            #dataset = scale_transform_condition(dataset, dataset_test, cond_names)
            if params['WHICH_LIQUID'] == 'il':
                dataset = cond_transformer.fit_transform(dataset, cond_names, df_test=dataset_test)
            elif params['WHICH_LIQUID'] == 'des':
                dataset = cond_transformer.fit_transform(dataset, cond_names, df_test=dataset_test)
                #dataset = cond_transformer.transform_new(dataset, cond_names)
        return dataset, dataset_test, cond_names
    else:
        if params['SCALE_CONDITIONS']:
            #dataset = scale_transform_condition(dataset, cond=cond_names)
            if params['WHICH_LIQUID'] == 'il':
                dataset = cond_transformer.fit_transform(dataset, cond_names, df_test=dataset_test)
            elif params['WHICH_LIQUID'] == 'des':
                #dataset = cond_transformer.transform_new(dataset, cond_names)
                dataset = cond_transformer.fit_transform(dataset, cond_names, df_test=dataset_test)
        return dataset, cond_names
    
def set_training(params=None, device=None,scaler_paths=None):
    global target_transformer, cond, feature_transformer, dataset, dataset_test, cond_names, data_list, df_gfeats, loader_des, val_loader_des, test_loader_des, train_list_des, valid_list_des, test_list_des
    if params['FEATURE_TRANSFORM']:
        target_transformer = TargetTransformer(method=params['FEATURE_TRANSFORM_METHOD'], filename=os.path.join(scaler_paths, f"{params['TARGET_FEATURE_NAME']}_{params['FEATURE_TRANSFORM_METHOD']}.pkl"))
    else:
        target_transformer = None
    cond_names = params['CONDITION_NAMES'] 
    cond = '_'.join(cond_names) if len(cond_names) > 1 else cond_names[0]
    if params['SCALE_CONDITIONS']:
        feature_transformer = ConditionalScaler(scaler_type=params['SCALE_CONDITIONS_TYPE'], save_path=os.path.join(scaler_paths, f"{cond}_{params['TARGET_FEATURE_NAME']}-{params['SCALE_CONDITIONS_TYPE']}.pkl"))
    else:
        feature_transformer = None 
    if params['TESTSET_FROM_DATASET'] == True:
        dataset, dataset_test, cond_names = prepare_pandas_datasets(params,transformer=target_transformer, cond_transformer=feature_transformer)
    else:
       dataset, cond_names = prepare_pandas_datasets(params,transformer) 
    if params['GLOBAL_FEATURES']:
        glob_transformer = GlobTransformer(method=params['SCALE_CONDITIONS_TYPE'], filename=os.path.join(scaler_paths, f"{params['TARGET_FEATURE_NAME']}_global_features_transformer.pkl"))
        data_list, df_gfeats = prepare_graph_data_from_mol(params, dataset, glob_transformer=glob_transformer, rbf_centers=None, rbf_gamma=10.0)
    else:
        glob_transformer = None
        data_list = prepare_graph_data_from_mol(params, dataset,glob_transformer=glob_transformer, rbf_centers=None, rbf_gamma=10.0)

    loader, val_loader, test_loader, train_list, valid_list, test_list = prepare_data_loaders_numbered(params, dataset, dataset_test, data_list)

    return loader, val_loader, test_loader, data_list, target_transformer, feature_transformer, glob_transformer

def set_training_des(params=None, device=None,scaler_paths=None):
    global target_transformer, cond, feature_transformer, dataset, dataset_test, cond_names, data_list, df_gfeats, loader, val_loader, test_loader, train_list, valid_list, test_list
    params['WHICH_LIQUID'] = 'des'

    if params['FEATURE_TRANSFORM']:
        target_transformer = TargetTransformer(method=params['FEATURE_TRANSFORM_METHOD'], filename=os.path.join(scaler_paths, f"des_{params['TARGET_FEATURE_NAME']}_{params['FEATURE_TRANSFORM_METHOD']}.pkl"))
    else:
        target_transformer = None
    
    cond_names = params['CONDITION_NAMES'] 
    cond = '_'.join(cond_names) if len(cond_names) > 1 else cond_names[0]
    if params['SCALE_CONDITIONS']:
        feature_transformer = ConditionalScaler(scaler_type=params['SCALE_CONDITIONS_TYPE'], save_path=os.path.join(scaler_paths, f"des_{cond}_{params['TARGET_FEATURE_NAME']}-{params['SCALE_CONDITIONS_TYPE']}.pkl"))
    else:
        feature_transformer = None 
    
    if params['TESTSET_FROM_DATASET'] == True:
        dataset, dataset_test, cond_names = prepare_pandas_datasets(params,transformer=target_transformer, cond_transformer=feature_transformer)
    else:
       dataset, cond_names = prepare_pandas_datasets(params,transformer) 
    
    if params['GLOBAL_FEATURES']:
        glob_transformer = GlobTransformer(method=params['SCALE_CONDITIONS_TYPE'], filename=os.path.join(scaler_paths, f"des_{params['TARGET_FEATURE_NAME']}_global_features_transformer.pkl"))
        data_list, df_gfeats = prepare_graph_data_from_mol(params, dataset, glob_transformer=glob_transformer, rbf_centers=None, rbf_gamma=10.0)
    else:
        glob_transformer = None
        data_list = prepare_graph_data_from_mol(params, dataset, glob_transformer=glob_transformer, rbf_centers=None, rbf_gamma=10.0)
    
    loader_des, val_loader_des, test_loader_des, train_list_des, valid_list_des, test_list_des = prepare_data_loaders_numbered(params, dataset, dataset_test, data_list)

    return loader_des, val_loader_des, test_loader_des, data_list, target_transformer, feature_transformer, glob_transformer

def run_model(params=None, data_list=None,loader=None, val_loader=None, test_loader=None, target_transformer=None):
    batch = data_list[0]
    input_dim = batch.x.shape[1]
    edge_dim = batch.edge_attr.shape[1]
    cond_dim = len(params['CONDITION_NAMES'])
    params['INPUT_DIM'] = int(input_dim)
    params['EDGE_DIM'] = int(edge_dim)
    if params['GLOBAL_FEATURES']:
        global_feats_dim = int(batch.global_feats.shape[1])
        params['COND_DIM'] = cond_dim + global_feats_dim
    else:
        global_feats_dim = 0 
        params['COND_DIM'] = cond_dim 
    
    model = run_experiment_with_search(ARCHS, 
                           params, 
                           loader, 
                           val_loader, 
                           test_loader, 
                           embedding_size=params['EMBEDDING_SIZE'],
                           linear_size=params['LINEAR_SIZE'],
                           input_dim=params['INPUT_DIM'], 
                           edge_dim=params['EDGE_DIM'], 
                           cond_dim=params['COND_DIM'],
                           EPOCHS=params['EPOCH'], 
                           device=params['DEVICE'],
                           transformer=target_transformer)
    logging.info(model)
    logging.info('\n')
    return model 

def run_experiment_with_search(ARCHS, params, loader, val_loader, test_loader,
                               embedding_size, linear_size, input_dim, edge_dim,
                               cond_dim,EPOCHS=None, device="cuda", transformer=None):
    EPOCHS=params['EPOCH']

    model = ARCHS[params['ARCH']](
        input_channels=input_dim,
        edge_channels=edge_dim,
        embedding_size=embedding_size,
        linear_size=linear_size,
        add_params_num=cond_dim,
        pooling=params['POOLING_METHOD']
    ).to(device)

    # Loss, optimizer, scheduler
    loss_fn = nn.MSELoss()
    #optimizer = optim.Adam(model.parameters(), lr=params.get("LR", 1e-3))  # Plain Adam optimizer
    #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    optimizer = optim.AdamW(model.parameters(), lr=params.get("LR", 1e-3), weight_decay=1e-4)  # Adm optimizer with decouple regularization
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-5)   # Helps in escaping local minima and can improve generalization
    loss_r2 = R2Score().to(device)
    loss_mae = MeanAbsoluteError().to(device)
    loss_mare = MeanAbsolutePercentageError().to(device)

    # Training
    if params['SEARCH_MODE'] or params['TRAIN_MODE']:
        train, evaluate = prepare_training_baseline(params, device, optimizer, model, loss_fn, loss_r2, loss_mae)
        losses, val_losses, coeffs, val_coeffs, maes, val_maes = perform_training(
            params, train, evaluate, loader, val_loader, scheduler, EPOCHS=EPOCHS, log_every=params['LOG_EVERY'], loss_threshold=params['LOSS_THRESHOLD']
        )

    if params['SEARCH_MODE']:
        # Return validation R2 (or MAE) for hyperparameter optimization
        return np.mean(val_maes[-10:])

    if params['TRAIN_MODE']:
        test_losses, test_coeffs, test_maes = perform_validation(
            params, evaluate, loader, val_loader, test_loader, repeats=10
        )
        plot_metrics(
            params, losses, val_losses, test_losses, coeffs, val_coeffs, test_coeffs,
            tag=f"{params['ARCH']}-{params['POOLING_METHOD']}-{params['TARGET_FEATURE_NAME']}"
        )
        print_results_of_final_eval_baseline(
            params, model, device, loader, val_loader, test_loader,
            transformer=transformer, repeats=10
        )

    return model


def run_transfer(model, params, loader, val_loader, test_loader,EPOCHS=None, device="cuda", transformer=None, lr_head=1e-3, lr_encoder=1e-5, finetune=False):
    model = model.to(device)

    # Loss, optimizer, scheduler
    loss_fn = nn.MSELoss()
    loss_r2 = R2Score().to(device)
    loss_mae = MeanAbsoluteError().to(device)
    loss_mare = MeanAbsolutePercentageError().to(device)

    # Separate params: head vs encoder (if finetune)
    if finetune:
        encoder_params = [p for n,p in model.encoder.named_parameters() if p.requires_grad]
        head_params = [p for n,p in model.head.named_parameters() if p.requires_grad]
        optimizer = optim.AdamW([
            {'params': head_params, 'lr': lr_head},
            {'params': encoder_params, 'lr': lr_encoder}
        ], weight_decay=1e-4)
    else:
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr_head, weight_decay=1e-4)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    # Training
    if params['TRAIN_MODE']:
        train, evaluate = prepare_training_baseline(params, device, optimizer, model, loss_fn, loss_r2, loss_mae)
        losses, val_losses, coeffs, val_coeffs, maes, val_maes = perform_training(
            params, train, evaluate, loader, val_loader, scheduler, EPOCHS=EPOCHS, log_every=params['LOG_EVERY'], loss_threshold=params['LOSS_THRESHOLD']
        )

    if params['TRAIN_MODE']:
        test_losses, test_coeffs, test_maes = perform_validation(
            params, evaluate, loader, val_loader, test_loader, repeats=10
        )
        plot_metrics(
            params, losses, val_losses, test_losses, coeffs, val_coeffs, test_coeffs,
            tag=f"{params['ARCH']}-{params['POOLING_METHOD']}-{params['TARGET_FEATURE_NAME']}"
        )
        print_results_of_final_eval_baseline(
            params, model, device, loader, val_loader, test_loader,
            transformer=transformer, repeats=10
        )

    return model

def instantiate_model(params, data_list):
    batch = data_list[0]
    input_dim = batch.x.shape[1]
    edge_dim = batch.edge_attr.shape[1]
    cond_dim = len(params['CONDITION_NAMES'])
    params['INPUT_DIM'] = int(input_dim)
    params['EDGE_DIM'] = int(edge_dim)
    if params['GLOBAL_FEATURES']:
        global_feats_dim = int(batch.global_feats.shape[1])
        params['COND_DIM'] = cond_dim + global_feats_dim
    else:
        global_feats_dim = 0 
        params['COND_DIM'] = cond_dim 
    
    model = ARCHS[params['ARCH']](
        input_channels=params['INPUT_DIM'],
        edge_channels=params['EDGE_DIM'],
        embedding_size=params['EMBEDDING_SIZE'],
        linear_size=params['LINEAR_SIZE'],
        add_params_num=params['COND_DIM'],
        pooling=params['POOLING_METHOD']
    ).to(params['DEVICE'])

    return model 

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 

def objective(trial):
    set_seed(params['SEED'])
    # Update params dynamically
    params["ARCH"] = trial.suggest_categorical("ARCH", ["NNConvModel", "GCNConvModel"])
    params["num_emb_layers"] = trial.suggest_int("num_emb_layers", 3,4 )
    params["num_lin_layers"] = trial.suggest_int("num_lin_layers", 2,2)
    params['POOLING_METHOD'] = trial.suggest_categorical("POOLING_METHOD", ['add', 'mean', 'mean_max', 'attention'])
    
    # Embedding sizes
    embedding_size = []
    for i in range(params["num_emb_layers"]):
        embedding_size.append(trial.suggest_categorical(f"embedding_size_{i}", [16, 32]))
    
    # Linear sizes
    linear_size = []
    for i in range(params["num_lin_layers"]):
        linear_size.append(trial.suggest_categorical(f"linear_size_{i}", [16, 8]))
    
    # Learning rate
    params["LR"] = trial.suggest_float("LR", 1e-3, 1e-2, log=True)

    embedding_size = sorted(embedding_size, reverse=False)
    linear_size = sorted(linear_size, reverse=True)
    # Check extreme sizes early (optional)
    if max(embedding_size) > 32 or max(linear_size) > 16:
        return -1.0
    
    try:
        # Run your experiment with search mode
        val_r2 = run_experiment_with_search(
            ARCHS,
            params,
            loader, val_loader, test_loader,
            embedding_size, linear_size,
            input_dim=params['INPUT_DIM'], 
            edge_dim=params['EDGE_DIM'], 
            cond_dim=params['COND_DIM'],
            EPOCHS=10, 
            device=device
        )
        
        # If NaN occurs, fail gracefully
        if np.isnan(val_r2) or np.isinf(val_r2):
            return -1.0
        
        return val_r2
    
    except Exception as e:
        # Print for debugging but fail trial gracefully
        logging.info(f"Trial failed due to exception: {e}")
        return -1.0


# Run Optuna study 
def run_search(objective, direction="miniimize", n_trials=30, show_progress_bar=True):
    pruner = optuna.pruners.MedianPruner(
    n_startup_trials=5,   # wait at least 5 trials before pruning
    n_warmup_steps=1,     # minimum number of epochs before pruning
    interval_steps=1      # check pruning every step
    )

    study = optuna.create_study(direction=direction, pruner=pruner)  # maximize R²
    study.optimize(objective, n_trials=n_trials, show_progress_bar=show_progress_bar)
    
    # Print best results
    if study.best_trial.value is not None:
        logging.info("\nBest Trial:")
        print(f"MAE: {study.best_trial.value:.4f}")
        logging.info("Best Hyperparameters:")
        for k, v in study.best_trial.params.items():
            logging.info(f"  {k}: {v}")
    else:
        logging.info("No successful trials yet.")
    return study 

'''Visualize hyperparameters search optimization history'''
def show_optimization(study):
    from optuna.visualization import (
        plot_optimization_history,
        plot_param_importances,
        plot_parallel_coordinate
    )
    
    # 1. How the objective improved over trials
    fig1 = plot_optimization_history(study)
    fig1.write_image(os.path.join(Figures,'objective_trial_improvement.png'), scale=3)
    #fig1.show()
    
    # 2. Which hyperparameters are most important
    fig2 = plot_param_importances(study)
    fig2.write_image(os.path.join(Figures,'hyperparameter_importance.png'), scale=3)
    #fig2.show()
    
    # 3. Visualize interactions between hyperparameters
    fig3 = plot_parallel_coordinate(study)
    fig3.write_image(os.path.join(Figures,'interaction_between_hyperparameters.png'), scale=3)
    #fig3.show()