import requests
import os, sys, re, glob
from pathlib import Path
import pandas
import pandas as pd
import numpy as np
from scipy import stats
import pubchempy as pcp
import pickle
import gzip
from mordred import Calculator, descriptors
import statistics
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, PowerTransformer, QuantileTransformer
import joblib
import matplotlib.pyplot as plt
import torch
import torch_geometric
import torch_scatter
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torchview import draw_graph
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
import warnings

warnings.filterwarnings(
    "ignore",
    message="The behavior of DataFrame concatenation with empty or all-NA entries is deprecated",
    category=FutureWarning
)

pd.plotting.register_matplotlib_converters()
pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns',None)
pd.set_option('display.max_colwidth', None)

from architecture import *

ARCHS = {'NNConv':NNConvModel,
         'GCNConv':GCNConvModel,
        }
ARCHS_FLEXIBLE = {'NNConv':NNConvModel,
                 'GCNConv':GCNConvModel,
                 }

def make_dir(base, folder):
    path = os.path.join(base, folder)
    os.makedirs(path, exist_ok=True)
    return path

#### Neccessary Paths Definitions 
home_dir = '/Users/rnogbodo/ML/raph_GNN' # replace with your own main folder path
DATA = make_dir(home_dir,'data')
Figures = make_dir(home_dir,'Figures')
TEXTFILES = make_dir(home_dir,'textfiles')
MODELS = make_dir(home_dir,'models')
scaler_paths = make_dir(home_dir,'scalers')
params_path = os.path.join(home_dir, 'code', 'params.yaml')
logfiles = make_dir(home_dir, 'logfiles')

def get_device_and_torch_info():
    """
    Checks the validity of the Torch library by performing several operations.

    This function checks the availability of a CUDA device and sets the device to be
    used by Torch accordingly. It also prints the device being used by Torch, the
    version of Torch, and the version of CUDA used for Torch compilation. Additionally,
    it attempts to perform a CUDA operation using `torch.zeros(1).cuda()` and catches
    any potential runtime errors.

    Returns:
        device (torch.device): The device used by Torch, either "cuda" or "cpu".

    """
    try:
        gpu_memory = [torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i)
                      for i in range(torch.cuda.device_count())]
        best_gpu = gpu_memory.index(max(gpu_memory))
        device = torch.device(f"cuda:{best_gpu}")
        logging.info(f"Using GPU {best_gpu}")
    except:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f'Device used by torch: {device}')
    logging.info(f'Version of torch: {torch.__version__}')
    logging.info(f'CUDA used for torch compilation: {torch.version.cuda}')
    try:
        logging.info(f'{torch.zeros(1).cuda()}')
    except RuntimeError as inst:
        logging.info(f'Runtime Error {inst}')
    return device

def get_params_settings(yaml_params_path='./params.yaml'):
    """
    Set the parameters for the experiment.
    Args:
        yaml_params_path (str, optional): The path to the YAML file containing the parameters. Defaults to './params.yaml'.
    Returns:
        dict: A dictionary containing the parameters.
    """
    params = yaml.load(open(yaml_params_path), Loader=yaml.loader.SafeLoader)
    return params


def rotate_log_files_recency(params, log_file):
    logs = []
    log_dir = os.path.dirname(log_file)
    if params['TRANSFER']:
        lgs = glob.glob(os.path.join(log_dir, f'transfer_{params["TARGET_FEATURE_NAME"]}*.log'))
    else:
        lgs = glob.glob(os.path.join(log_dir, f'{params["TARGET_FEATURE_NAME"]}*.log'))
    for lg in lgs:
        logs.append(os.path.basename(lg))
    import re 
    vals = []
    for fl in logs: 
        if re.search('(\d+)', fl):
            vals.append(int(re.search('(\d+)', fl).group()))

    if vals:
        num = max(vals) + 1
        if params['TRANSFER']:
            log_file = os.path.join(log_dir, f'transfer_{params["TARGET_FEATURE_NAME"]}-{num}.log')
        else:
            log_file = os.path.join(log_dir, f'{params["TARGET_FEATURE_NAME"]}-{num}.log')
    else:
        if len(logs) == 1:
            if params['TRANSFER']:
                log_file = os.path.join(log_dir, f'transfer_{params["TARGET_FEATURE_NAME"]}-1.log')
            else:
                log_file = os.path.join(log_dir, f'{params["TARGET_FEATURE_NAME"]}-1.log')
        else:
            if params['TRANSFER']:
                log_file = os.path.join(log_dir, f'transfer_{params["TARGET_FEATURE_NAME"]}.log')
            else:
                log_file = os.path.join(log_dir, f'{params["TARGET_FEATURE_NAME"]}.log')
    return log_file

def reset_logging(log_file=None):
    # Close existing loggers
    for handler in logging.root.handlers[:]:
        handler.close()
        logging.root.removeHandler(handler)
    
    # Remove old log file safely
    if os.path.exists(log_file):
        os.remove(log_file)
    
    # Reinitialize logging
    logging.basicConfig(filename=log_file, level=logging.INFO)
    logging.info("Logging restarted successfully.")
    

def start_logging(params, log_folder=None):
    if params['TRANSFER']:
        log_file = os.path.join(home_dir, 'logfiles', f'transfer_{params["TARGET_FEATURE_NAME"]}.log')
    else:
        log_file = os.path.join(home_dir, 'logfiles', f'{params["TARGET_FEATURE_NAME"]}.log')
    if log_folder is None:
        log_folder = os.getcwd()
    os.makedirs(log_folder, exist_ok=True)
    log_path = os.path.join(log_folder, log_file)
    log_path = rotate_log_files_recency(params, log_path)
    
    reset_logging(log_path)

    # Create a custom logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Global log level

    # Remove any existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # File handler
    file_handler = logging.FileHandler(log_path, mode='a')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(levelname)s | %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    logging.info(f"Logging started ({os.path.basename(log_file)} + console)")


def prepare_datasets(params):
    """
    Prepare the datasets for training and testing.

    Parameters:
        params (dict): A dictionary containing the parameters for dataset preparation.

    Returns:
        dataset (pd.DataFrame): The dataset prepared for training or testing.
        clean_dataset (pd.DataFrame, optional): The cleaned dataset for testing, if applicable.
    """
    if params['CLEAN'] == True:
        dataset = pd.read_excel(os.path.join(DATA, f'{params["TARGET_FEATURE_NAME"]}_ambient.xlsx'))
    else:
        dataset = pd.read_csv(os.path.join(DATA, f'{params["TARGET_FEATURE_NAME"]}_all_condition_range.xlsx'))
    return dataset

def prepare_datasets_des(params):
    """
    Prepare the datasets for training and testing.

    Parameters:
        params (dict): A dictionary containing the parameters for dataset preparation.

    Returns:
        dataset (pd.DataFrame): The dataset prepared for training or testing.
        clean_dataset (pd.DataFrame, optional): The cleaned dataset for testing, if applicable.
    """
    if params['CLEAN'] == True:
        dataset = pd.read_excel(os.path.join(DATA, f'des-{params["TARGET_FEATURE_NAME"]}_ambient.xlsx'))
    else:
        dataset = pd.read_csv(os.path.join(DATA, f'des-{params["TARGET_FEATURE_NAME"]}_all_condition_range.xlsx'))
    return dataset

def renaming_and_cleaning(params, dataset):
    """
    Renames target features and cleans the dataset based on the given parameters.

    Args:
        params (dict): A dictionary containing the parameters for renaming and cleaning.
        dataset (pandas.DataFrame): The dataset to be renamed and cleaned.

    Returns:
        list: A list of condition names based on the target feature name.

    Raises:
        None
    """
    if params["TARGET_FEATURE_NAME"] == 'density':
        dataset.rename(columns={"density_Kg*m-3": "y"}, inplace=True)
        cond_names = params['CONDITION_NAMES'] #['temperature_K', 'pressure_MPa']
    elif params["TARGET_FEATURE_NAME"] == 'viscosity':
        dataset.rename(columns={"viscosity_cP": "y"}, inplace=True)
        if params["TRANSFER"] == False:
            cond_names = params['CONDITION_NAMES']
        else:
            #dataset['pressure_MPa'] = 0.1
            cond_names = params['CONDITION_NAMES'] # ['temperature_K', 'pressure_MPa']
    elif params["TARGET_FEATURE_NAME"] == 'conductivity':
        dataset.rename(columns={"conductivity_mS-m": "y"}, inplace=True)
        if params["TRANSFER"] == False:
            cond_names = params['CONDITION_NAMES']
        else:
            #dataset['P_MPa'] = 0.1
            cond_names = params['CONDITION_NAMES'] #['temperature_K', 'pressure_MPa']
    elif params["TARGET_FEATURE_NAME"] == 'refractive_index':
        dataset.rename(columns={"refractive_index": "y"}, inplace=True)
        if params["TRANSFER"] == False:
            cond_names = params['CONDITION_NAMES']
        else:
            #dataset['P_MPa'] = 0.1
            cond_names = params['CONDITION_NAMES'] #['temperature_K', 'pressure_MPa']

    # droping columns with no value other than accountancy
    dataset['smile'] = dataset.get('il_smile', dataset.get('des_smile'))
    if 'il_name' in dataset.columns:
        dataset['name'] = dataset['il_name']
    if 'HBA' in dataset.columns and 'HBD' in dataset.columns:
        dataset['name'] = [f"{a} {b}" for a,b in zip(dataset['HBA'].values.tolist(),dataset['HBD'].values.tolist())]
    dataset.drop_duplicates(inplace=True)
    print(dataset.columns.values.tolist())
    return cond_names

def remove_outliers(dataset, outlier_method='MAD'):
    """
    Remove outliers from the dataset using the specified outlier detection method.

    Parameters:
        dataset (pandas.DataFrame): The dataset containing the outliers.
        outlier_method (str): The method to be used for outlier detection. 
            Available options: 'Z_score', 'IQR', 'log_IQR', 'MAD', '' (default).

    Returns:
        None

    Raises:
        Exception: If an unavailable option is specified for the outlier_method.

    Notes:
        - This function modifies the dataset in-place by dropping the outlier rows.
        - The outlier detection methods are as follows:
            - Z_score: Uses the Z-score method to detect outliers.
            - IQR: Uses the Interquartile Range (IQR) method to detect outliers.
            - log_IQR: Uses the log-transformed version of the IQR method to detect outliers.
            - MAD: Uses the Median Absolute Deviation (MAD) method to detect outliers.
            - '': Does not perform any outlier detection.

    """
    if outlier_method == 'Z_score':
        # outlier detection using Z-score
        threshold = 3
        for _ in range(1):
            to_drop = dataset[(np.abs(stats.zscore(dataset['y'])) > threshold)]
            dataset.drop(to_drop.index, axis = 0, inplace = True)
        logging.info(f'Dropped {to_drop.shape[0]} outliers')

    elif outlier_method == 'IQR':
        # outlier detection using IQR
        Q1, Q3 = np.percentile(dataset['y'], [25,75])
        ul = Q3 + 1.5 * (Q3 - Q1)
        ll = Q1 - 1.5 * (Q3 - Q1)
        to_drop = dataset[(dataset['y'] < ll) | (dataset['y'] > ul)]
        dataset.drop(to_drop.index, axis = 0, inplace = True)
        logging.info(f'Dropped {to_drop.shape[0]} outliers')

    elif outlier_method == 'log_IQR':
        # outlier detection using log version of IQR
        Q1,Q3 = np.percentile(np.log(dataset['y']), [25,75])
        ul = Q3 + 1.5 * (Q3 - Q1)
        ll = Q1 - 1.5 * (Q3 - Q1)
        to_drop = dataset[(np.log(dataset['y']) < ll) | (np.log(dataset['y']) > ul)]
        dataset.drop(to_drop.index, axis = 0, inplace = True)
        logging.info(f'Dropped {to_drop.shape[0]} outliers')

    elif outlier_method == 'MAD':
        # outlier detection using median absolute deviation method (MAD)
        threshold = 3
        med = np.median(dataset['y'])
        mad = np.abs(stats.median_abs_deviation(dataset['y']))
        to_drop = dataset[((dataset['y'] - med) / mad) > threshold]
        dataset.drop(to_drop.index, axis = 0, inplace = True)
        logging.info(f'Dropped {to_drop.shape[0]} outliers')
        
    elif outlier_method == '':
        logging.info('No outlier detection method.')
    else:
        raise Exception('Sorry! Unavailable option')

def remove_unwanted_liquids(dataset, verbose=True):
    """
    Removes problematic and undefined liquids from the dataset.
    
    Parameters:
        dataset (DataFrame): The dataset containing the liquids.
        verbose (bool): If True, print the deleted smiles.
    
    Returns:
        None
    """
    dataset['formal_charge'] = dataset['smile'].apply(lambda x: Chem.GetFormalCharge(Chem.MolFromSmiles(x)) if x else np.nan)
    dataset = dataset.dropna()
    dataset = dataset[dataset['formal_charge'] == 0].reset_index(drop=True)

    for smile in dataset.smile.values:
        temp = smile.split('.')
        # check for multicationic liquids and remove them
        if 'HBA' not in list(dataset.columns):  # This is for ILs only and not Deep Eutectics
            if len(temp) > 2:
                dataset.drop(dataset[dataset['smile'] == smile].index, inplace = True)
                if verbose:
                    if logging:
                        logging.info(f'deleting {smile}')
        # check for possible nan in smiles
        if 'nan' in temp:
            dataset.drop(dataset[dataset['smile'] == smile].index, inplace = True)
            if verbose:
                if logging:
                    logging.info(f'deleting {smile}')

def create_rdkit_mols(dataset):
    dataset['mol'] = dataset['smile'].apply(lambda x: Chem.MolFromSmiles(x) if x else None)


class TargetTransformer:   # transforms target variable to be predicted
    def __init__(self, method='none',filename="target_transformer.pkl"):
        """
        method: combinations like:
            'none', 'log', 'log1p',
            'minmax', 'standard', 'robust', 'yeo-johnson', 'quantile',
            or compound forms like 'log+minmax', 'log1p+standard', etc.
        """
        self.method = method.lower()
        self.transforms = self.method.split('+')
        self.filename = filename
        self.scaler = None

    def fit(self, y):
        # Handle log transforms before fitting scalers
        if 'log' in self.transforms:
            if np.any(y <= 0):
                raise ValueError("Log transform requires positive target values.")
            y = np.log(y)
        elif 'log1p' in self.transforms:
            y = np.log1p(y)

        # Fit the second stage scaler if present
        if 'minmax' in self.transforms:
            self.scaler = MinMaxScaler().fit(y)
        elif 'standard' in self.transforms:
            self.scaler = StandardScaler().fit(y)
        elif 'robust' in self.transforms:
            self.scaler = RobustScaler().fit(y)
        elif 'yeo-johnson' in self.transforms:
            self.scaler = PowerTransformer(method='yeo-johnson').fit(y)
        elif 'quantile' in self.transforms:
            self.scaler = QuantileTransformer(output_distribution='normal').fit(y)

    def transform(self, y):

        if 'log' in self.transforms:
            if np.any(y <= 0):
                raise ValueError("Log transform requires positive target values.")
            y = np.log(y)
        elif 'log1p' in self.transforms:
            y = np.log1p(y)

        if self.scaler is not None:
            y = self.scaler.transform(y)

        return y

    def inverse_transform(self, y):

        if self.scaler is not None:
            y = self.scaler.inverse_transform(y)

        # Apply inverse of log transforms last (reverse order)
        if 'log' in self.transforms:
            y = np.exp(y)
        elif 'log1p' in self.transforms:
            y = np.expm1(y)

        return y

    def save(self):
        joblib.dump({'method': self.method, 'scaler': self.scaler}, self.filename)

    def load(self):
        obj = joblib.load(self.filename)
        self.method = obj['method']
        self.transforms = self.method.split('+')
        self.scaler = obj['scaler']


class ConditionalScaler:  # Transforms external conditions
    """
    Scale selected dataframe columns conditionally using a specified scaler type
    with optional log/log1p transformations. Automatically saves and loads the fitted
    scaler for reuse.

    Parameters
    ----------
    scaler_type : str
        Type of scaler to use: 'minmax', 'standard', or 'robust'.
        Can combine with log transforms: e.g., 'log+minmax', 'log1p+robust'.
    save_path : str, optional
        Path to save the trained scaler (default: '{scaler_type}_scaler.pkl')
    """

    def __init__(self, scaler_type="minmax", save_path=None):
        self.method = scaler_type.lower()
        self.transforms = self.method.split('+')
        self.save_path = save_path or f"{self.method}_scaler.pkl"
        self.scaler = None

    def _initialize_scaler(self):
        """Initialize the chosen scaler type (excluding log transforms)."""
        if 'minmax' in self.transforms:
            return MinMaxScaler()
        elif 'standard' in self.transforms:
            return StandardScaler()
        elif 'robust' in self.transforms:
            return RobustScaler()
        else:
            return None  # No scaler if only log/log1p is used

    def fit_transform(self, df, cond, df_test=None, id_col='numbered'):
        """
        Fit the scaler on training data and transform both train/test sets.

        Parameters
        ----------
        df : pd.DataFrame
            Main dataframe (training + test combined).
        df_test : pd.DataFrame, optional
            Separate dataframe for test molecules. Used to fit only on training molecules.
        cond : list
            List of columns to scale.
        id_col : str
            Column name that uniquely identifies molecules (used for mask).
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame.")
        if cond is None or len(cond) == 0:
            raise ValueError("cond must be a non-empty list of columns to scale.")

        # Apply log/log1p transforms first
        for col in cond:
            if 'log' in self.transforms:
                if (df[col] <= 0).any():
                    raise ValueError("Log transform requires positive values.")
                df[col] = np.log(df[col])
            elif 'log1p' in self.transforms:
                df[col] = np.log1p(df[col])

        self.scaler = self._initialize_scaler()

        if df_test is not None:
            train_mask = ~df[id_col].isin(df_test[id_col])
            if self.scaler is not None:
                self.scaler.fit(df.loc[train_mask, cond])
                df.loc[train_mask, cond] = self.scaler.transform(df.loc[train_mask, cond])
                df.loc[~train_mask, cond] = self.scaler.transform(df.loc[~train_mask, cond])
        else:
            if self.scaler is not None:
                self.scaler.fit(df[cond])
                df[cond] = self.scaler.transform(df[cond])

        # Save the scaler for reuse
        joblib.dump({'method': self.method, 'scaler': self.scaler}, self.save_path)
        print(f"Scaler saved to: {self.save_path}")

        return df

    def transform_new(self, new_data, cond):
        """
        Apply the previously saved scaler on new unseen data.

        Parameters
        ----------
        new_data : pd.DataFrame
            New test dataframe to scale.
        cond : list
            Columns to scale (same as during training).
        """
        obj = joblib.load(self.save_path)
        self.method = obj['method']
        self.transforms = self.method.split('+')
        self.scaler = obj['scaler']

        # Apply log/log1p transforms first
        for col in cond:
            if 'log' in self.transforms:
                if (new_data[col] <= 0).any():
                    raise ValueError("Log transform requires positive values.")
                new_data[col] = np.log(new_data[col])
            elif 'log1p' in self.transforms:
                new_data[col] = np.log1p(new_data[col])

        # Apply scaler if exists
        if self.scaler is not None:
            new_data[cond] = self.scaler.transform(new_data[cond])

        return new_data

    def inverse_transform(self, scaled_data, cond):
        """
        Revert scaled columns back to original scale.
        """
        obj = joblib.load(self.save_path)
        self.method = obj['method']
        self.transforms = self.method.split('+')
        self.scaler = obj['scaler']

        # Inverse scale first
        if self.scaler is not None:
            scaled_data[cond] = self.scaler.inverse_transform(scaled_data[cond])

        # Apply inverse of log/log1p
        for col in cond:
            if 'log' in self.transforms:
                scaled_data[col] = np.exp(scaled_data[col])
            elif 'log1p' in self.transforms:
                scaled_data[col] = np.expm1(scaled_data[col])

        return scaled_data


class GlobTransformer:  # Transforms the global features
    def __init__(self, method='none', filename="global_features_transformer.pkl"):
        """
        method: combinations like:
            'none', 'log', 'log1p',
            'minmax', 'standard', 'robust', 'yeo-johnson', 'quantile',
            or compound forms like 'log+minmax', 'log1p+standard', etc.
        """
        self.method = method.lower()
        self.transforms = self.method.split('+')
        self.filename = filename
        self.scaler = None
        self.offset_ = None  # for handling negative values in log1p
        self.columns_ = None  # store DataFrame column names

    def fit(self, y: pd.DataFrame):
        if not isinstance(y, pd.DataFrame):
            raise TypeError("Expected a pandas DataFrame as input.")
        self.columns_ = y.columns

        y = y.copy()

        # Handle log transforms before fitting scalers
        if 'log' in self.transforms:
            if (y <= 0).any().any():
                raise ValueError("Log transform requires strictly positive target values.")
            y = np.log(y)

        elif 'log1p' in self.transforms:
            # Compute offset to shift negative data before log1p
            min_vals = y.min()
            self.offset_ = np.abs(min_vals) + 1e-9
            y = np.log1p(y + self.offset_)

        # Fit the second-stage scaler if present
        if 'minmax' in self.transforms:
            self.scaler = MinMaxScaler().fit(y)
        elif 'standard' in self.transforms:
            self.scaler = StandardScaler().fit(y)
        elif 'robust' in self.transforms:
            self.scaler = RobustScaler().fit(y)
        elif 'yeo-johnson' in self.transforms:
            self.scaler = PowerTransformer(method='yeo-johnson').fit(y)
        elif 'quantile' in self.transforms:
            self.scaler = QuantileTransformer(output_distribution='normal').fit(y)

    def transform(self, y: pd.DataFrame):
        if not isinstance(y, pd.DataFrame):
            raise TypeError("Expected a pandas DataFrame as input.")
        y = y.copy()

        if 'log' in self.transforms:
            if (y <= 0).any().any():
                raise ValueError("Log transform requires strictly positive target values.")
            y = np.log(y)

        elif 'log1p' in self.transforms:
            if self.offset_ is None:
                min_vals = y.min()
                self.offset_ = np.abs(min_vals) + 1e-9
            y = np.log1p(y + self.offset_)

        if self.scaler is not None:
            y[:] = self.scaler.transform(y)

        return y

    def inverse_transform(self, y: pd.DataFrame):
        if not isinstance(y, pd.DataFrame):
            raise TypeError("Expected a pandas DataFrame as input.")
        y = y.copy()

        if self.scaler is not None:
            y[:] = self.scaler.inverse_transform(y)

        # Apply inverse of log transforms last (reverse order)
        if 'log' in self.transforms:
            y = np.exp(y)
        elif 'log1p' in self.transforms:
            if self.offset_ is None:
                raise ValueError("Offset missing — cannot invert log1p transform.")
            y = np.expm1(y) - self.offset_

        return y

    def save(self):
        joblib.dump({
            'method': self.method,
            'scaler': self.scaler,
            'offset_': self.offset_,
            'columns_': self.columns_
        }, self.filename)

    def load(self):
        obj = joblib.load(self.filename)
        self.method = obj['method']
        self.transforms = self.method.split('+')
        self.scaler = obj['scaler']
        self.offset_ = obj.get('offset_', None)
        self.columns_ = obj.get('columns_', None)

def separate_cat_from_an_similes(smile):
    def get_charges(smile):
        mol = Chem.MolFromSmiles(smile)
        mol_mol = Chem.MolToMolBlock(mol)
        bel_mol = pybel.readstring('mol', mol_mol)
        ob_charge_model = openbabel.OBChargeModel.FindType(params['OB'])
        ob_charge_model.ComputeCharges(bel_mol.OBMol)
        charges = [bel_mol.OBMol.GetAtom(i+1).GetFormalCharge() for i, atom in enumerate(mol.GetAtoms())]
        return charges

    smiles = smile.split('.')
    cat_smiles = []  ; an_smiles = []
    if len(smiles) >= 2:
        for smile in smiles:
            charges = get_charges(smile)
            if sum(charges) > 0:
                cat_smiles.append(smile)
            elif sum(charges) < 0:
                an_smiles.append(smile)
        cat_smile = '.'.join(cat_smiles)
        an_smile = '.'.join(an_smiles)
        return (cat_smile, an_smile)
    elif len(smiles) < 2:
        return smile

def RDOptimize(params, smi, track_problems=False):
    if track_problems: potentially_problematic_one_smi = False
    mol = Chem.MolFromSmiles(smi)
    with StringIO() as buf:
        with redirect_stderr(buf):
            mol_h = Chem.AddHs(mol)
            res_molh_error = buf.getvalue()
            if res_molh_error != '':
                logging.info(res_molh_error, smi, 'issue during adding Hs', sep=' ')
                if track_problems: potentially_problematic_one_smi = True
    try:
        ps = AllChem.ETKDGv3()
        ps.randomSeed = params['SEED']
        ps.useRandomCoords = True
        ps.maxAttempts = 50
        with StringIO() as buf:
            with redirect_stderr(buf):
                cids = AllChem.EmbedMultipleConfs(mol_h, numConfs = 10, params = ps)
                res_cids_error = buf.getvalue()
                if res_cids_error != '':
                    logging.info(res_cids_error, smi, 'issue during multiple embedding', sep=' ')
                    if track_problems: potentially_problematic_one_smi = True
            results = AllChem.MMFFOptimizeMoleculeConfs(mol_h, maxIters = 500)
        if len(results) == 0:
            with StringIO() as buf:
                with redirect_stderr(buf):
                    res_embedding = AllChem.EmbedMolecule(mol_h, useRandomCoords=True, randomSeed=params['SEED'])
                    res_embedding_error = buf.getvalue()
                    logging.info(res_embedding_error, smi, 'issue after optim', sep=' ')
                    if track_problems: potentially_problematic_one_smi = True
                    if res_embedding_error != '':
                        logging.info(res_embedding_error, smi, 'issue with single embedding - try to compute 2D', sep=' ')
                        AllChem.Compute2DCoords(mol_h) # to provide at least 2D coords - OB performs better anyway
            final_molecule = mol_h
        else:
            min_energy, min_energy_index = 10000, 0
            for index, result in enumerate(results):
                if(min_energy>result[1]):
                    min_energy = result[1]
                    min_energy_index = index
            final_molecule = Chem.Mol(mol_h, False, min_energy_index)
    except ValueError as veinst:
        with StringIO() as buf:
            with redirect_stderr(buf):
                res_embedding = AllChem.EmbedMolecule(mol_h, useRandomCoords=True, randomSeed=params['SEED'])
                res_embedding_error = buf.getvalue()
                if res_embedding_error != '':
                    logging.info(veinst, res_embedding_error, smi, 'issue with ValueError - not very known reason', sep=' ')
                    if track_problems: potentially_problematic_one_smi = True
        final_molecule = mol_h
    if track_problems:
        return final_molecule, potentially_problematic_one_smi
    else:
        return final_molecule

def safe_embed_mol(mol, track_problems=False):
    smi = Chem.MolToSmiles(mol)  # regenerate SMILES in case you passed Mol directly
    mol_opt = RDOptimize(params, smi, track_problems=track_problems)
    if track_problems:
        mol_opt, flag = mol_opt
        return mol_opt, flag
    return mol_opt

# Atom charges
def get_atom_charges(params, smiles, return_type="numpy", allow_metal=True):
    """
    Compute per-atom Gasteiger charges with NaN-safe handling and fallbacks.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        if logging:
            logging.warning(f"Invalid SMILES skipped: {smiles}")
        else:
            print(f"Invalid SMILES skipped: {smiles}")
        return None, []

    if params.get('ADD_HS', False):
        mol = Chem.AddHs(mol)

    charges = []

    if params['CHARGE_ENGINE'] == 'rdkit':
        try:
            AllChem.ComputeGasteigerCharges(mol)
            for i, atom in enumerate(mol.GetAtoms()):
                try:
                    c = float(atom.GetProp(params['CHARGE_MODEL']))
                except Exception:
                    c = np.nan  # Mark missing charge
                charges.append(c)
        except Exception:
            charges = [np.nan] * mol.GetNumAtoms()
            if logging:
                logging.debug(f"Failed to compute charges with RDKit for {smiles}")
            else:
                print(f"Failed to compute charges with RDKit for {smiles}")

    elif params['CHARGE_ENGINE'] == 'ob':
        try:
            mol_block = Chem.MolToMolBlock(mol)
            bel_mol = pybel.readstring('mol', mol_block)
            ob_charge_model = openbabel.OBChargeModel.FindType(params['CHARGE_MODEL'])
            ob_charge_model.ComputeCharges(bel_mol.OBMol)
            charges = [bel_mol.OBMol.GetAtom(i+1).GetFormalCharge() for i in range(mol.GetNumAtoms())]
        except Exception:
            charges = [np.nan] * mol.GetNumAtoms()
            if logging:
                logging.debug(f"Failed to compute charges with OpenBabel for {smiles}")
            else:
                print(f"Failed to compute charges with OpenBabel for {smiles}")

    #  Replace NaN or infinite charges safely
    charges = [0.0 if (not np.isfinite(c)) else c for c in charges]

    # Optional normalization
    if params.get('NORMALIZE', False):
        try:
            formal_charge = Chem.GetFormalCharge(mol)
            total = sum(charges)
            correction = (formal_charge - total) / len(charges)
            charges = [c + correction for c in charges]
        except Exception:
            pass

    # Attach charges to atom properties
    for atom, c in zip(mol.GetAtoms(), charges):
        atom.SetDoubleProp("charge", float(c))

    return mol, np.array(charges) if return_type == "numpy" else charges

def get_atom_features(params, mol, return_type="numpy", allow_metal=True):
    from rdkit.Chem import Descriptors
    from mendeleev import element
    pt = Chem.GetPeriodicTable()

    # Handle hydrogens consistently
    if params['REMOVE_HS']:
        mol = Chem.RemoveHs(mol)
        num_Hs = [0] * mol.GetNumAtoms()  # pad with zeros for shape consistency
    else:
        mol = Chem.AddHs(mol)
        num_Hs = [atom.GetTotalNumHs(includeNeighbors=True) for atom in mol.GetAtoms()]

    # Charges
    _, charges = get_atom_charges(params, Chem.MolToSmiles(mol), allow_metal=allow_metal)
    if len(charges) != mol.GetNumAtoms() and allow_metal:
        charges = [0.0] * mol.GetNumAtoms()
        if logging:
            logging.info(f"Set default charges for unknown atoms in molecule {Chem.MolToSmiles(mol)}")
        else:
            print(f"Set default charges for unknown atoms in molecule {Chem.MolToSmiles(mol)}")

    # Hybridization (encode safely)
    hybr_list = []
    for i, atom in enumerate(mol.GetAtoms()):
        try:
            hybr_list.append(str(atom.GetHybridization()))
        except Exception:
            hybr_list.append("UNK")
            if allow_metal:
                if logging:
                    logging.info(f"Set unknown hybridization for atom {i} in {Chem.MolToSmiles(mol)}")
                else:
                    print(f"Set unknown hybridization for atom {i} in {Chem.MolToSmiles(mol)}")

    hybr_enc = LabelEncoder()
    hybr = hybr_enc.fit_transform(hybr_list)

    # Other atom descriptors
    atomic_number = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    degree = [atom.GetDegree() for atom in mol.GetAtoms()]
    aromaticity = [atom.GetIsAromatic() for atom in mol.GetAtoms()]
    formal_charge = [atom.GetFormalCharge() for atom in mol.GetAtoms()]
    in_ring = [atom.IsInRing() for atom in mol.GetAtoms()]
    rvdw = [pt.GetRvdw(atom.GetAtomicNum()) for atom in mol.GetAtoms()]
    total_valence = [atom.GetTotalValence() for atom in mol.GetAtoms()]

    # Electronegativity lookup (Pauling Scale)
    pauling_en = {
        1: 2.20,  3: 0.98,  4: 1.57,  5: 2.04,  6: 2.55,  7: 3.04,  8: 3.44,  9: 3.98,
        11: 0.93, 12: 1.31, 13: 1.61, 14: 1.90, 15: 2.19, 16: 2.58, 17: 3.16,
        19: 0.82, 20: 1.00, 21: 1.36, 22: 1.54, 23: 1.63, 24: 1.66, 25: 1.55, 26: 1.83,
        27: 1.88, 28: 1.91, 29: 1.90, 30: 1.65, 31: 1.81, 32: 2.01, 33: 2.18, 34: 2.55,
        35: 2.96, 36: 3.00, 37: 0.82, 38: 0.95, 39: 1.22, 40: 1.33, 41: 1.60, 42: 2.16,
        43: 1.90, 44: 2.20, 45: 2.28, 46: 2.20, 47: 1.93, 48: 1.69, 49: 1.78, 50: 1.96,
        51: 2.05, 52: 2.10, 53: 2.66, 54: 2.60, 55: 0.79, 56: 0.89, 57: 1.10, 58: 1.12,
        59: 1.13, 60: 1.14, 62: 1.17, 64: 1.20, 66: 1.22, 67: 1.23, 68: 1.24, 69: 1.25,
        71: 1.27, 72: 1.30, 73: 1.50, 74: 2.36, 75: 1.90, 76: 2.20, 77: 2.20, 78: 2.28,
        79: 2.54, 80: 2.00, 81: 1.62, 82: 2.33, 83: 2.02, 84: 2.00, 85: 2.20,
        87: 0.70, 88: 0.90, 89: 1.10, 90: 1.30, 91: 1.50, 92: 1.38, 93: 1.36, 94: 1.28,
        95: 1.30, 96: 1.30, 97: 1.30, 98: 1.30, 99: 1.30, 100: 1.30, 101: 1.30,
        102: 1.30, 103: 1.30
    }

    try:
        en_values = [pauling_en.get(atom.GetAtomicNum(), 0.0) for atom in mol.GetAtoms()]
    except:
        en_values = [element(atom.GetAtomicNum()).en_pauling for atom in mol.GetAtoms()]

    # Build feature matrix
    feature_matrix = np.array([
        atomic_number,
        charges,
        hybr,
        aromaticity,
        num_Hs,
        in_ring,
        en_values,
        formal_charge,
        degree,
        rvdw,
        total_valence
    ]).T

    if return_type == 'numpy':
        return feature_matrix.astype(np.float32)
    elif return_type == 'torch':
        return torch.tensor(feature_matrix, dtype=torch.float32)


def get_edges_info(params, mol, pos=None, rbf_centers=None, rbf_gamma=10.0, return_type="torch", allow_metal=True):
    from rdkit.Chem import rdMolTransforms

    # Electronegativity lookup (Pauling Scale)
    pauling_en = {
        1: 2.20,  3: 0.98,  4: 1.57,  5: 2.04,  6: 2.55,  7: 3.04,  8: 3.44,  9: 3.98,
        11: 0.93, 12: 1.31, 13: 1.61, 14: 1.90, 15: 2.19, 16: 2.58, 17: 3.16,
        19: 0.82, 20: 1.00, 21: 1.36, 22: 1.54, 23: 1.63, 24: 1.66, 25: 1.55, 26: 1.83,
        27: 1.88, 28: 1.91, 29: 1.90, 30: 1.65, 31: 1.81, 32: 2.01, 33: 2.18, 34: 2.55,
        35: 2.96, 36: 3.00, 37: 0.82, 38: 0.95, 39: 1.22, 40: 1.33, 41: 1.60, 42: 2.16,
        43: 1.90, 44: 2.20, 45: 2.28, 46: 2.20, 47: 1.93, 48: 1.69, 49: 1.78, 50: 1.96,
        51: 2.05, 52: 2.10, 53: 2.66, 54: 2.60, 55: 0.79, 56: 0.89, 57: 1.10, 58: 1.12,
        59: 1.13, 60: 1.14, 62: 1.17, 64: 1.20, 66: 1.22, 67: 1.23, 68: 1.24, 69: 1.25,
        71: 1.27, 72: 1.30, 73: 1.50, 74: 2.36, 75: 1.90, 76: 2.20, 77: 2.20, 78: 2.28,
        79: 2.54, 80: 2.00, 81: 1.62, 82: 2.33, 83: 2.02, 84: 2.00, 85: 2.20,
        87: 0.70, 88: 0.90, 89: 1.10, 90: 1.30, 91: 1.50, 92: 1.38, 93: 1.36, 94: 1.28,
        95: 1.30, 96: 1.30, 97: 1.30, 98: 1.30, 99: 1.30, 100: 1.30, 101: 1.30,
        102: 1.30, 103: 1.30
    }

    # Safe embedding
    if params['USE_RBF'] == True or params['USE_ABF'] == True: 
        mol, flagged = safe_embed_mol(mol, track_problems=True)
    mol = Chem.AddHs(mol)

    row, col, edge_types = [], [], []
    bond_types_list = []

    # Bond type mapping
    bond_map = {
        Chem.rdchem.BondType.SINGLE: 0,
        Chem.rdchem.BondType.DOUBLE: 1,
        Chem.rdchem.BondType.TRIPLE: 2,
        Chem.rdchem.BondType.AROMATIC: 3
    }

    # Collect bonds
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [i, j]
        col += [j, i]
        bt = bond_map.get(bond.GetBondType(), 0)
        edge_types += [bt, bt]
        bond_types_list += [bt, bt]

    # Ionic connections
    if params['ADD_IONIC']:
        mol_block = Chem.MolToMolBlock(mol)
        bel_mol = pybel.readstring("mol", mol_block)
        pos_atoms = [i for i in range(mol.GetNumAtoms()) if bel_mol.OBMol.GetAtom(i+1).GetFormalCharge() > 0]
        neg_atoms = [i for i in range(mol.GetNumAtoms()) if bel_mol.OBMol.GetAtom(i+1).GetFormalCharge() < 0]
        if params['CONNECT_ALL']:
            pairs = [(p,n) for p in pos_atoms for n in neg_atoms]
        else:
            pairs = [(pos_atoms[0], neg_atoms[0])] if pos_atoms and neg_atoms else []
        for i,j in pairs:
            row += [i,j]
            col += [j,i]
            edge_types += [4,4]
            bond_types_list += [4,4]

    # One-hot encoding of bond types
    num_types = 5
    edge_features = np.eye(num_types)[bond_types_list]

    # Extra bond features (coordinate-free)
    bonds = mol.GetBonds()
    bond_is_conj = [int(bond.GetIsConjugated()) for bond in bonds for _ in (0,1)]
    bond_in_ring = [int(bond.IsInRing()) for bond in bonds for _ in (0,1)]
    bond_en_diff = []
    for bond in bonds:
        a1, a2 = bond.GetBeginAtom(), bond.GetEndAtom()
        en1 = pauling_en.get(a1.GetAtomicNum(), 0.0)
        en2 = pauling_en.get(a2.GetAtomicNum(), 0.0)
        diff = abs(en1 - en2)
        bond_en_diff += [diff, diff]

    extra_bond_feats = np.vstack([bond_is_conj, bond_in_ring, bond_en_diff]).T

    # Pad extra bond features to match edge_features length (for ionic edges)
    if len(extra_bond_feats) < len(edge_features):
        pad_len = len(edge_features) - len(extra_bond_feats)
        extra_bond_feats = np.vstack([extra_bond_feats, np.zeros((pad_len, extra_bond_feats.shape[1]), dtype=np.float32)])

    edge_features = np.hstack([edge_features, extra_bond_feats])

    # RBF features
    if params['USE_RBF'] and pos is not None:
        if rbf_centers is None:
            rbf_centers = np.linspace(0, 10, 20)
        try:
            conf = mol.GetConformer()
            pos_np = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
        except:
            pos_np = np.zeros((mol.GetNumAtoms(), 3))
        distances = np.linalg.norm(pos_np[row] - pos_np[col], axis=1)
        rbf_feats = np.exp(-rbf_gamma * (distances[:, None] - rbf_centers[None, :])**2)
        edge_features = np.hstack([edge_features, rbf_feats])

    # Angle-based features
    if params['USE_ABF'] and pos is not None:
        try:
            conf = mol.GetConformer()
            angles = []
            for i,j in zip(row,col):
                nbrs = [nbr.GetIdx() for nbr in mol.GetAtomWithIdx(j).GetNeighbors() if nbr.GetIdx()!=i]
                if nbrs:
                    cos_vals = [np.cos(rdMolTransforms.GetAngleRad(conf, i,j,k)) for k in nbrs]
                    angles.append(np.mean(cos_vals))
                else:
                    angles.append(0.0)
            edge_features = np.hstack([edge_features, np.array(angles)[:,None]])
        except:
            pass  # fallback

    edge_index = np.array([row, col], dtype=np.int64)
    edge_features = np.array(edge_features, dtype=np.float32)
    edge_types = np.array(edge_types, dtype=np.int64)

    if return_type=="torch":
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        edge_features = torch.tensor(edge_features, dtype=torch.float)
        edge_types = torch.tensor(edge_types, dtype=torch.long)

    return edge_index, edge_features, edge_types


def compute_quality_descriptors(mol):
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem
    """
    Compute high-quality, chemically meaningful descriptors for a molecule (IL ion).
    No 3D embedding is required.
    """
    descs = {
        # Basic physicochemical
        'MolWt': Descriptors.MolWt(mol),                       # Mass-related
        'MolLogP': Descriptors.MolLogP(mol),                   # Hydrophobicity
        'TPSA': Descriptors.TPSA(mol),                         # Polar surface area
        #'HeavyAtomCount': Descriptors.HeavyAtomCount(mol),     # Size indicator
        #'NumValenceElectrons': Descriptors.NumValenceElectrons(mol),

        # Structural / topological 
        'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),  # Flexibility
        'FractionCSP3': rdMolDescriptors.CalcFractionCSP3(mol),   # Hybridization ratio
        'NumAliphaticRings': rdMolDescriptors.CalcNumAliphaticRings(mol),
        'NumAromaticRings': rdMolDescriptors.CalcNumAromaticRings(mol),
        #'LabuteASA': rdMolDescriptors.CalcLabuteASA(mol),         # Surface area proxy

        # Charge and polarity
        'NumHAcceptors': rdMolDescriptors.CalcNumHBA(mol),
        'NumHDonors': rdMolDescriptors.CalcNumHBD(mol),

        # Complexity
        'BertzCT': Descriptors.BertzCT(mol)  # Graph-theoretic molecular complexity
    }
    
    return torch.tensor(list(descs.values())) , list(descs.keys())

def prepare_graph_data_from_mol(params, dataset, glob_transformer=None, cond_names=None, input_type='numpy', rbf_centers=None, rbf_gamma=10.0,allow_metal=True):
    """
    Prepare list of graph data from a dataset for GNN.

    Parameters:
        dataset (pd.DataFrame): Dataset containing molecules and conditions.
        cond_names (list): List of condition column names (optional).
        input_type (str): 'numpy' or 'torch'.
        use_rbf (bool): Add radial basis function features.
        use_abf (bool): Add angular basis function features.
        rbf_centers (array-like): Centers for RBF.
        rbf_gamma (float): Gamma for RBF.
        add_ionic (bool): Connect all positive to all negative atoms.
        allow_metal (bool): If True, keep molecules with metals (use dummy positions if embedding fails).

    Returns:
        data_list (list of torch_geometric.data.Data)
    """
    dataset = dataset.reset_index(drop=True)
    data_list = []

    if params['GLOBAL_FEATURES']:
        mol_list = dataset['mol'].tolist()
        global_feats_columns = compute_quality_descriptors(mol_list[0])[1]
        # Compute all global features
        all_feats = torch.stack([compute_quality_descriptors(m)[0] for m in mol_list])  # shape (n_mols, n_feats)

        df_gfeats = pd.DataFrame(all_feats, columns=global_feats_columns)

        if params['NORM_GLOBAL_FEATURE'] and glob_transformer:
            if params['PREDICT']:
                df_gfeats[global_feats_columns] = glob_transformer.transform(df_gfeats[global_feats_columns])
            else:
                glob_transformer.fit(df_gfeats[global_feats_columns])
                df_gfeats[global_feats_columns] = glob_transformer.transform(df_gfeats[global_feats_columns])
                glob_transformer.save()


    for idx in tqdm(dataset.index):  # for idx, row in tqdm(dataset.iterrows())
        row = dataset.loc[idx]
        row_gf = df_gfeats.loc[idx]
        mol = Chem.MolFromSmiles(row['smile'])
        if mol is None:
            continue
        mol = Chem.AddHs(mol)
        
        # Embed molecule
        if params['USE_RBF'] or params['USE_ABF']: 
            try:           
                embed_status = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
                if embed_status < 0:
                    raise ValueError("Embedding failed")
                conf = mol.GetConformer()
                pos = torch.tensor([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())],
                                   dtype=torch.float)
            except Exception as e:
                if allow_metal:
                    pos = torch.zeros((mol.GetNumAtoms(), 3), dtype=torch.float)
                    if logging:
                        logging.info(f"Embedding failed or exception occurred, using zero positions for {row['smile']}: {e}")
                    else:
                        print(f"Embedding failed or exception occurred, using zero positions for {row['smile']}: {e}")
                else:
                    if logging:
                        logging.warning(f"Embedding failed, skipping {row['smile']}: {e}")
                    else:
                        print(f"Embedding failed, skipping {row['smile']}: {e}")
                    continue
        else:
            pos = None

        # Atom features
        try:
            x = torch.tensor(get_atom_features(params, mol, allow_metal=allow_metal), dtype=torch.float)
        except Exception as e:
            if logging:
                logging.warning(f"Failed to compute atom features for {row['smile']}: {e}")
            else:
                print(f"Failed to compute atom features for {row['smile']}: {e}")
            continue

        # Edge information
        try:
            edge_index, edge_attr, edge_types = get_edges_info(
                params, mol, pos=pos, rbf_centers=rbf_centers, rbf_gamma=rbf_gamma,
                return_type='torch'
            )
        except Exception as e:
            if logging:
                logging.warning(f"Failed to compute edges for {row['smile']}: {e}")
            else:
                print(f"Failed to compute edges for {row['smile']}: {e}")
            continue

        # Target
        y = torch.tensor(row['y'], dtype=torch.float).view(1, 1)
        

        # Build graph
        data = torch_geometric.data.Data(x=x,
                                         edge_index=edge_index,
                                         edge_attr=edge_attr,
                                         y=y)

        # Optional condition features
        if params['CONDITION_NAMES'] is not None:
            cond_values = row[params['CONDITION_NAMES']].to_numpy(dtype=float).reshape(1, -1)
            data.cond = torch.tensor(cond_values, dtype=torch.float32)

        # SMILES string for reference
        data.smi = row['smile']
        if 'numbered' in dataset.columns:
            data.numbered = row['numbered']

        if 'id' in dataset.columns: # checks column
            data.id = row['id']
        if 'references' in dataset.columns:
            data.ref = row['references']
        if 'name' in dataset.columns:
            data.name = row['name']

        # Edge type tensor
        data.edge_type = edge_types
        if params['GLOBAL_FEATURES']:
            #global_feats = compute_quality_descriptors(mol)[0].reshape(1, -1)
            #if params['NORM_GLOBAL_FEATURE'] and glob_transformer:
            #    #global_feats = (global_feats - min_feats) / (max_feats - min_feats)
            #    global_feats = glob_transformer.transform(global_feats)
                
            data.global_feats = torch.tensor(row_gf[global_feats_columns].values, dtype=torch.float32).unsqueeze(0)

        data_list.append(data) 
    if params['GLOBAL_FEATURES']:
        return data_list, df_gfeats
    else:
        return data_list  

def stratified_split(params, df):
    """
    Perform a stratified train-test split on both continuous and categorical variables.
    Falls back to random split if any stratification group has <2 samples.
    """
    from sklearn.model_selection import train_test_split
    import pandas as pd
    import numpy as np

    df_strat = df.copy()

    # Bin continuous columns
    if params.get('CONTINUOUS_COLUMNS'):
        for col in params['CONTINUOUS_COLUMNS']:
            try:
                df_strat[f"{col}_bin"] = pd.qcut(
                    df_strat[col],
                    q=params.get('NUM_BINS', 4),
                    duplicates="drop"
                )
            except ValueError:
                # fallback if too few unique values
                df_strat[f"{col}_bin"] = pd.cut(
                    df_strat[col],
                    bins=params.get('NUM_BINS', 4),
                    duplicates="drop"
                )

    # Build stratify columns
    strat_cols = []
    if params.get('STRATIFY_CONT_ONLY', False):
        if params.get('CONTINUOUS_COLUMNS'):
            strat_cols = [f"{col}_bin" for col in params['CONTINUOUS_COLUMNS']]
    else:
        if params.get('CONTINUOUS_COLUMNS'):
            strat_cols.extend([f"{col}_bin" for col in params['CONTINUOUS_COLUMNS']])
        if params.get('CATEGORICAL_COLUMNS'):
            strat_cols.extend(params['CATEGORICAL_COLUMNS'])

    # Create stratify label
    if strat_cols:
        df_strat["stratify_label"] = df_strat[strat_cols].astype(str).agg("_".join, axis=1)
        label_counts = df_strat["stratify_label"].value_counts()

        # Check for groups with < 2 samples
        too_small = (label_counts < 2).any()

        if too_small:
            print("Warning: some stratification bins too small — falling back to random split.")
            stratify_labels = None
        else:
            stratify_labels = df_strat["stratify_label"]
    else:
        stratify_labels = None

    # Final split (safe fallback)
    train_df, test_df = train_test_split(
        df,
        test_size=params.get('TEST_FRACTION', 0.2),
        random_state=params.get('SEED', 42),
        stratify=stratify_labels
    )

    return train_df, test_df

def prepare_data_loaders(params, dataset, smi_for_test, data):
    import random
    if params['SPLITTER'] == 'scaffold':
        per_train = params['TRAIN_FRACTION']
        smi_list = []
        for smile in dataset.smile.values:
            if (smile not in smi_list) and (smile not in smi_for_test.smile.values.tolist()):
                smi_list.append(smile)
        random.Random(params['SEED']).shuffle(smi_list)
        train_smi = smi_list[:int(len(smi_list) * per_train)]
        valid_smi = smi_list[int(len(smi_list) * per_train):]

        train_list, valid_list, test_list = [], [], []
        for item in tqdm(data):
            if item.smi in train_smi:
                train_list.append(item)
            elif item.smi in valid_smi:
                valid_list.append(item)
            else:
                if item.smi in smi_for_test.smile.values.tolist():
                    for index, row in smi_for_test[smi_for_test['smile'] == item.smi].iterrows():
                        cond_match = (row[params['CONDITION_NAMES']].values.astype(np.float32) == item.cond.view(-1).numpy()).all()
                        y_match = row['y'] == item.y
                        numbered = row['numbered'] = item.numbered
                        if y_match and numbered:
                            test_list.append(item)
        logging.info(f'{len(test_list) = }, {len(smi_for_test) = }')
    elif params['SPLITTER'] == 'random':
        per_train = params['TRAIN_FRACTION']
        train_val_list, test_list = [], []
        for item in tqdm(data):
            if item.smi in smi_for_test.smile.values:
                for index, row in smi_for_test[smi_for_test['smile'] == item.smi].iterrows():
                    cond_match = (row[params['CONDITION_NAMES']].values.astype(np.float32) == item.cond.view(-1).numpy()).all()
                    y_match = row['y'] == item.y
                    numbered = row['numbered'] = item.numbered
                    if y_match and numbered:
                        test_list.append(item)
            else:
                train_val_list.append(item)
        random.Random(params['SEED']).shuffle(train_val_list)
        train_list = train_val_list[:int(len(train_val_list) * per_train)]
        valid_list = train_val_list[int(len(train_val_list) * per_train):]
    else:
        raise Exception('Sorry! Unavailable option')
    loader = DataLoader(train_list,
                        batch_size=params['NUM_GRAPHS_PER_BATCH'], shuffle=True)
    val_loader = DataLoader(valid_list,
                        batch_size=params['NUM_GRAPHS_PER_BATCH'], shuffle=True)
    test_loader = DataLoader(test_list,
                        batch_size=params['NUM_GRAPHS_PER_BATCH'], shuffle=True)
    
    if params['SAVE_GRAPH_LIST'] and params['WHICH_LIQUID'] == 'il':
        GraphDataIO.save(train_list, os.path.join(DATA, f"{params['TARGET_FEATURE_NAME']}_train_graphs.pkl.gz"), compress=params['COMPRESS'])
        GraphDataIO.save(valid_list, os.path.join(DATA,f"{params['TARGET_FEATURE_NAME']}_valid_graphs.pkl.gz"), compress=params['COMPRESS'])
        GraphDataIO.save(test_list, os.path.join(DATA,f"{params['TARGET_FEATURE_NAME']}_test_graphs.pkl.gz"), compress=params['COMPRESS'])
        GraphDataIO.save(data, os.path.join(DATA,f"{params['TARGET_FEATURE_NAME']}_all_data_graphs.pkl.gz"), compress=params['COMPRESS'])
    if params['SAVE_GRAPH_LIST'] and params['WHICH_LIQUID'] == 'des':
        GraphDataIO.save(train_list, os.path.join(DATA, f"des_{params['TARGET_FEATURE_NAME']}_train_graphs.pkl.gz"), compress=params['COMPRESS'])
        GraphDataIO.save(valid_list, os.path.join(DATA,f"des_{params['TARGET_FEATURE_NAME']}_valid_graphs.pkl.gz"), compress=params['COMPRESS'])
        GraphDataIO.save(test_list, os.path.join(DATA,f"des_{params['TARGET_FEATURE_NAME']}_test_graphs.pkl.gz"), compress=params['COMPRESS'])
        GraphDataIO.save(data, os.path.join(DATA,f"des_{params['TARGET_FEATURE_NAME']}_all_data_graphs.pkl.gz"), compress=params['COMPRESS'])

    return loader, val_loader, test_loader, train_list, valid_list, test_list

def prepare_data_loaders_numbered(params, dataset, smi_for_test, data):
    import random
    if params['SPLITTER'] == 'scaffold':
        per_train = params['TRAIN_FRACTION']
        num_list = []
        for num in dataset.numbered.values:
            if (num not in num_list) and (num not in smi_for_test.numbered.values.tolist()):
                num_list.append(num)
        random.Random(params['SEED']).shuffle(num_list)
        train_num = num_list[:int(len(num_list) * per_train)]
        valid_num = num_list[int(len(num_list) * per_train):]

        train_list, valid_list, test_list = [], [], []
        for item in tqdm(data):
            if item.numbered in train_num:
                train_list.append(item)
            elif item.numbered in valid_num:
                valid_list.append(item)
            else:
                if item.numbered in smi_for_test.numbered.values.tolist():
                    for index, row in smi_for_test[smi_for_test['numbered'] == item.numbered].iterrows():
                        cond_match = (row[params['CONDITION_NAMES']].values.astype(np.float32) == item.cond.view(-1).numpy()).all()
                        y_match = row['y'] == item.y
                        numbered = row['numbered'] = item.numbered
                        if y_match and numbered:
                            test_list.append(item)
        if logging:
            logging.info(f'{len(test_list) = }, {len(smi_for_test) = }')
        else:
            print(f'{len(test_list) = }, {len(smi_for_test) = }')
    elif params['SPLITTER'] == 'random':
        per_train = params['TRAIN_FRACTION']
        train_val_list, test_list = [], []
        for item in tqdm(data):
            if item.numbered in smi_for_test.numbered.values:
                for index, row in smi_for_test[smi_for_test['numbered'] == item.numbered].iterrows():
                    cond_match = (row[params['CONDITION_NAMES']].values.astype(np.float32) == item.cond.view(-1).numpy()).all()
                    y_match = row['y'] == item.y
                    numbered = row['numbered'] = item.numbered
                    if y_match and numbered:
                        test_list.append(item)
            else:
                train_val_list.append(item)
        random.Random(params['SEED']).shuffle(train_val_list)
        train_list = train_val_list[:int(len(train_val_list) * per_train)]
        valid_list = train_val_list[int(len(train_val_list) * per_train):]
    else:
        raise Exception('Sorry! Unavailable option')
    loader = DataLoader(train_list,
                        batch_size=params['NUM_GRAPHS_PER_BATCH'], shuffle=True)
    val_loader = DataLoader(valid_list,
                        batch_size=params['NUM_GRAPHS_PER_BATCH'], shuffle=True)
    test_loader = DataLoader(test_list,
                        batch_size=params['NUM_GRAPHS_PER_BATCH'], shuffle=True)
    
    if params['SAVE_GRAPH_LIST'] and params['WHICH_LIQUID'] == 'il':
        GraphDataIO.save(train_list, os.path.join(DATA, f"{params['TARGET_FEATURE_NAME']}_train_graphs.pkl.gz"), compress=params['COMPRESS'])
        GraphDataIO.save(valid_list, os.path.join(DATA,f"{params['TARGET_FEATURE_NAME']}_valid_graphs.pkl.gz"), compress=params['COMPRESS'])
        GraphDataIO.save(test_list, os.path.join(DATA,f"{params['TARGET_FEATURE_NAME']}_test_graphs.pkl.gz"), compress=params['COMPRESS'])
        GraphDataIO.save(data, os.path.join(DATA,f"{params['TARGET_FEATURE_NAME']}_all_data_graphs.pkl.gz"), compress=params['COMPRESS'])
    if params['SAVE_GRAPH_LIST'] and params['WHICH_LIQUID'] == 'des':
        GraphDataIO.save(train_list, os.path.join(DATA, f"des_{params['TARGET_FEATURE_NAME']}_train_graphs.pkl.gz"), compress=params['COMPRESS'])
        GraphDataIO.save(valid_list, os.path.join(DATA,f"des_{params['TARGET_FEATURE_NAME']}_valid_graphs.pkl.gz"), compress=params['COMPRESS'])
        GraphDataIO.save(test_list, os.path.join(DATA,f"des_{params['TARGET_FEATURE_NAME']}_test_graphs.pkl.gz"), compress=params['COMPRESS'])
        GraphDataIO.save(data, os.path.join(DATA,f"des_{params['TARGET_FEATURE_NAME']}_all_data_graphs.pkl.gz"), compress=params['COMPRESS'])

    return loader, val_loader, test_loader, train_list, valid_list, test_list

from torch.amp import autocast, GradScaler
def prepare_training_baseline(params, device, optimizer, model, loss_fn, loss_r2, loss_mae):
    """
    Returns train() and evaluate() functions tied to the provided model and optimizer.
    Works for MoleculeGNN baseline.
    """
    use_amp = device.type == 'cuda'
    dev_type = 'cuda' if use_amp else 'cpu'
    scaler = GradScaler(enabled=use_amp)  # just enable AMP, device_type removed

    def train(loader):
        model.train()
        losses = torch.tensor(0.0, device=device)
        r2s = torch.tensor(0.0, device=device)
        maes = torch.tensor(0.0, device=device)

        for i, batch in enumerate(loader):
            batch = batch.to(device)
            optimizer.zero_grad()

            with autocast(device_type=dev_type,enabled=use_amp):
                if params['GLOBAL_FEATURES']:
                    pred, embedding = model(batch.x.float(),
                                            batch.edge_index,
                                            batch.edge_attr,
                                            batch.batch,
                                            batch.cond,
                                            batch.global_feats)
                else:
                    pred, embedding = model(batch.x.float(),
                                            batch.edge_index,
                                            batch.edge_attr,
                                            batch.batch,
                                            batch.cond)
                loss = torch.sqrt(loss_fn(pred, batch.y))

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if params['TRANSFER']:
                torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_norm=1.0)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            # accumulate metrics
            losses = losses + loss.detach()
            maes = maes + loss_mae(pred, batch.y).detach()
            if batch.y.numel() >= 2:
                r2s = r2s + loss_r2(pred, batch.y).detach()

        n_batches = len(loader)
        losses_avg = (losses / n_batches).clone().detach()
        r2 = (r2s / n_batches).clone().detach()
        mae = (maes / n_batches).clone().detach()

        return loss, embedding, losses_avg, r2, mae

    def evaluate(val_loader):
        model.eval()
        losses = torch.tensor(0.0, device=device)
        r2s = torch.tensor(0.0, device=device)
        maes = torch.tensor(0.0, device=device)

        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                batch = batch.to(device)
                with autocast(device_type=dev_type,enabled=use_amp):
                    if params['GLOBAL_FEATURES']:
                        pred, _ = model(batch.x.float(),
                                        batch.edge_index,
                                        batch.edge_attr,
                                        batch.batch,
                                        batch.cond,
                                        batch.global_feats)
                    else:
                        pred, _ = model(batch.x.float(),
                                        batch.edge_index,
                                        batch.edge_attr,
                                        batch.batch,
                                        batch.cond)

                    loss = torch.sqrt(loss_fn(pred, batch.y))
                    mae = loss_mae(pred, batch.y)
                    r2 = loss_r2(pred, batch.y).detach() if batch.y.numel() >= 2 else torch.tensor(0.0, device=device)

                losses = losses + loss.detach()
                maes = maes + mae.detach()
                r2s = r2s + r2

        n_batches = len(val_loader)
        losses_avg = (losses / n_batches).clone().detach()
        r2 = (r2s / n_batches).clone().detach()
        mae = (maes / n_batches).clone().detach()

        return losses_avg, r2, mae

    return train, evaluate

def perform_training(params, train, evaluate, loader, val_loader, scheduler, EPOCHS=2, log_every=1, loss_threshold=0.001):
    print("Starting training...")
    losses, val_losses, coeffs, val_coeffs, maes, val_maes = [], [], [], [], [], []

    for epoch in range(1, EPOCHS + 1):
        loss, h, train_loss, train_r2, train_mae = train(loader)
        losses.append(train_loss.cpu().detach().numpy())
        coeffs.append(train_r2.cpu().detach().numpy())
        maes.append(train_mae.cpu().detach().numpy())

        val_loss, val_r2, val_mae = evaluate(val_loader)
        val_losses.append(val_loss.cpu().detach().numpy())
        val_coeffs.append(val_r2.cpu().detach().numpy())
        val_maes.append(val_mae.cpu().detach().numpy())

        if (epoch % log_every == 0) or (epoch == 1):
            logging.info(
                f"Epoch {epoch} | "
                f"Train Loss {train_loss:.3f} | Train MAE {train_mae:.3f} | Train R2 {train_r2:.2f} || "
                f"Val Loss {val_loss:.3f} | Val MAE {val_mae:.3f} | Val R2 {val_r2:.2f}"
            )

        if params['TRANSFER']:
            scheduler.step(val_loss)
        else:
            scheduler.step()

        # Early stopping based on loss threshold
        if loss_threshold is not None and train_loss <= loss_threshold:
            logging.info(f"Early stopping: training loss reached {train_loss:.4f} <= {loss_threshold}")
            break

    return losses, val_losses, coeffs, val_coeffs, maes, val_maes

def perform_validation(params, evaluate, loader, val_loader, test_loader, repeats=10):
    train_losses, train_r2s, train_maes = [], [], []
    val_losses, val_r2s, val_maes = [], [], []
    test_losses, test_r2s, test_maes = [], [], []

    for _ in range(repeats):
        tr_loss, tr_r2, tr_mae = evaluate(loader)
        va_loss, va_r2, va_mae = evaluate(val_loader)
        te_loss, te_r2, te_mae = evaluate(test_loader)

        train_losses.append(tr_loss.cpu().numpy().item())
        train_r2s.append(tr_r2.cpu().numpy().item())
        train_maes.append(tr_mae.cpu().numpy().item())

        val_losses.append(va_loss.cpu().numpy().item())
        val_r2s.append(va_r2.cpu().numpy().item())
        val_maes.append(va_mae.cpu().numpy().item())

        test_losses.append(te_loss.cpu().numpy().item())
        test_r2s.append(te_r2.cpu().numpy().item())
        test_maes.append(te_mae.cpu().numpy().item())

    logging.info(f"\nFinal evaluation for Model {params['ARCH']}-{params['POOLING_METHOD']}")
    logging.info(f"Train loss {np.mean(train_losses):.3f} ± {np.std(train_losses):.3f} | "
                 f"Train MAE {np.mean(train_maes):.3f} ± {np.std(train_maes):.3f} | "
                 f"Train R2 {np.mean(train_r2s):.2f} ± {np.std(train_r2s):.2f}")
    logging.info(f"Val   loss {np.mean(val_losses):.3f} ± {np.std(val_losses):.3f} | "
                 f"Val MAE {np.mean(val_maes):.3f} ± {np.std(val_maes):.3f} | "
                 f"Val R2 {np.mean(val_r2s):.2f} ± {np.std(val_r2s):.2f}")
    logging.info(f"Test  loss {np.mean(test_losses):.3f} ± {np.std(test_losses):.3f} | "
                 f"Test MAE {np.mean(test_maes):.3f} ± {np.std(test_maes):.3f} | "
                 f"Test R2 {np.mean(test_r2s):.2f} ± {np.std(test_r2s):.2f}")

    return test_losses, test_r2s, test_maes

def plot_metrics(params, losses, val_losses, test_losses, coeffs, val_coeffs,test_coeffs, tag="baseline"):
    losses = np.array(losses)
    val_losses = np.array(val_losses)
    test_losses = np.array(test_losses)
    coeffs = np.array(coeffs)
    val_coeffs = np.array(val_coeffs)
    test_coeffs = np.array(test_coeffs)

    # Save raw
    if params['TRANSFER']:
        np.savetxt(os.path.join(TEXTFILES, f'transfer_{tag}-train-loss.txt'), losses, fmt="%.4f")
        np.savetxt(os.path.join(TEXTFILES, f'transfer_{tag}-val-loss.txt'), val_losses, fmt="%.4f")
        np.savetxt(os.path.join(TEXTFILES, f'transfer_{tag}-test-loss.txt'), losses, fmt="%.4f")
    else:
        np.savetxt(os.path.join(TEXTFILES, f'{tag}-train-loss.txt'), losses, fmt="%.4f")
        np.savetxt(os.path.join(TEXTFILES, f'{tag}-val-loss.txt'), val_losses, fmt="%.4f")
        np.savetxt(os.path.join(TEXTFILES, f'{tag}-test-loss.txt'), losses, fmt="%.4f")

    # Loss plot
    fig_loss = go.Figure()
    fig_loss.add_scatter(y=losses, name="Train Loss")
    fig_loss.add_scatter(y=val_losses, name="Val Loss")
    fig_loss.add_scatter(y=test_losses, name="Test Loss")
    if params['TRANSFER']:
        fig_loss.write_image(os.path.join(Figures, f'transfer_{tag}-loss-plot.png'), scale=2)
    else:
        fig_loss.write_image(os.path.join(Figures, f'{tag}-loss-plot.png'), scale=2)

    # R² plot
    fig_r2 = go.Figure()
    fig_r2.add_scatter(y=coeffs, name="Train R²")
    fig_r2.add_scatter(y=val_coeffs, name="Val R²")
    fig_r2.add_scatter(y=test_coeffs, name="Test R²")
    if params['TRANSFER']:
        fig_r2.write_image(os.path.join(Figures, f'transfer_{tag}-r2-plot.png'), scale=2)
    else:
        fig_r2.write_image(os.path.join(Figures, f'{tag}-r2-plot.png'), scale=2)

    logging.info("Training complete")

def final_eval_baseline(params, model, device, test_loader, transformer=None):
    """
    Evaluate MoleculeGNN baseline on the test dataset and return original test and predicted values.
    transformer: fitted scaler/transformer used during training (optional)
    """
    original_y_test, original_y_pred = [], []
    smiles, refs, il_names, ids = [], [], [], []
    df_cond = pd.DataFrame(columns=params['CONDITION_NAMES'])
    ref = id = name = smi = True

    use_amp = device.type == 'cuda'
    dev_type = 'cuda' if use_amp else 'cpu'

    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            with autocast(device_type=dev_type,enabled=use_amp):
                if params['GLOBAL_FEATURES'] == True:
                    pred, _ = model(batch.x.float(),
                                    batch.edge_index,
                                    batch.edge_attr,
                                    batch.batch,
                                    batch.cond,
                                    batch.global_feats)
                else:
                    pred, _ = model(batch.x.float(),
                                    batch.edge_index,
                                    batch.edge_attr,
                                    batch.batch,
                                    batch.cond)

            ypred_for_eval = pred.cpu().numpy()
            ytrue_for_eval = batch.y.cpu().numpy()

            if params['FEATURE_TRANSFORM'] and params['FEATURE_TRANSFORM_METHOD'] != 'none':
                # Inverse-transform predictions and true labels
                ypred_for_eval = transformer.inverse_transform(ypred_for_eval).flatten()
                ytrue_for_eval = transformer.inverse_transform(ytrue_for_eval).flatten()
            else:
                ypred_for_eval = pred.flatten()
                ytrue_for_eval = batch.y.flatten()

            original_y_pred.extend(ypred_for_eval.tolist())
            original_y_test.extend(ytrue_for_eval.tolist())

            if hasattr(batch, 'smi') and batch.smi:
                smiles.extend(batch.smi)
                smi = True
            else:
                smi = False
            if hasattr(batch, 'name') and batch.name:
                il_names.extend(batch.name)
                name = True
            else:
                name = False
            if hasattr(batch, 'ref') and batch.ref:
                refs.extend(batch.ref)
                ref = True
            else:
                ref = False
            if hasattr(batch, 'id') and batch.ref:
                ids.extend(batch.id)
                id = True
            else:
                id = False
            if params['CONDITION_NAMES'] and hasattr(batch, 'cond'):
                df_cond = pd.concat([df_cond, pd.DataFrame(np.row_stack(batch.cond.detach().clone().cpu().numpy()), columns=params['CONDITION_NAMES'])], axis=0)
            
            # Optional memory cleanup for long evals
            torch.cuda.empty_cache()

    # Convert to NumPy arrays
    original_y_test_np = np.array(original_y_test)
    original_y_pred_np = np.array(original_y_pred)

    data = {'True_value':original_y_test_np, 'Predicted_value':original_y_pred_np}
    if params["FINAL_EVAL_DATAFRAME"]:
        if name: data['name'] = il_names
        if smi: data['smile'] = smiles
        if ref: data['references'] = refs
        if id: data['id'] = ids
        
        df = pd.DataFrame(data).reset_index(drop=True)

        if isinstance(df_cond, pd.DataFrame) and not df_cond.empty:
            df_cond = df_cond.reset_index(drop=True)
            df = pd.concat([df, df_cond], axis=1)
        return original_y_test_np, original_y_pred_np, df
    else:
        return original_y_test_np, original_y_pred_np


def print_results_of_final_eval_baseline(params, model, device, loader, val_loader, test_loader,
                                         transformer=None,repeats=10):
    """
    Evaluate MoleculeGNN baseline on original values (no normalization).
    Prints RMSE, MAE, R², MARE, A20 for train/val/test.
    """
    evaluator = RegressionMetric()

    train_losses, train_r2s, train_maes, train_mares, train_a20s = [], [], [], [], []
    val_losses, val_r2s, val_maes, val_mares, val_a20s = [], [], [], [], []
    test_losses, test_r2s, test_maes, test_mares, test_a20s = [], [], [], [], []

    for _ in range(repeats):
        # Train
        y_true_train, y_pred_train = final_eval_baseline(params, model, device, loader, transformer=transformer)
        train_losses.append(evaluator.root_mean_squared_error(y_true_train, y_pred_train))
        train_r2s.append(evaluator.coefficient_of_determination(y_true_train, y_pred_train))
        train_maes.append(evaluator.mean_absolute_error(y_true_train, y_pred_train))
        train_mares.append(evaluator.mean_absolute_percentage_error(y_true_train, y_pred_train))
        train_a20s.append(evaluator.a20_index(y_true_train, y_pred_train))

        # Val
        y_true_val, y_pred_val = final_eval_baseline(params, model, device, val_loader, transformer=transformer)
        val_losses.append(evaluator.root_mean_squared_error(y_true_val, y_pred_val))
        val_r2s.append(evaluator.coefficient_of_determination(y_true_val, y_pred_val))
        val_maes.append(evaluator.mean_absolute_error(y_true_val, y_pred_val))
        val_mares.append(evaluator.mean_absolute_percentage_error(y_true_val, y_pred_val))
        val_a20s.append(evaluator.a20_index(y_true_val, y_pred_val))

        # Test
        y_true_test, y_pred_test = final_eval_baseline(params, model, device, test_loader, transformer=transformer)
        test_losses.append(evaluator.root_mean_squared_error(y_true_test, y_pred_test))
        test_r2s.append(evaluator.coefficient_of_determination(y_true_test, y_pred_test))
        test_maes.append(evaluator.mean_absolute_error(y_true_test, y_pred_test))
        test_mares.append(evaluator.mean_absolute_percentage_error(y_true_test, y_pred_test))
        test_a20s.append(evaluator.a20_index(y_true_test, y_pred_test))

    logging.info(f"\nFinal evaluation for Model {params['ARCH']}-{params['POOLING_METHOD']} (test_on_original_data={params['TEST_ON_ORIGINAL_DATA']})")
    logging.info(f"Train: RMSE {np.mean(train_losses):.3f} ± {np.std(train_losses):.3f} | "
          f"MAE {np.mean(train_maes):.3f} ± {np.std(train_maes):.3f} | "
          f"R2 {np.mean(train_r2s):.2f} ± {np.std(train_r2s):.2f} | "
          f"MARE {np.mean(train_mares):.3f} ± {np.std(train_mares):.3f} | "
          f"A20 {np.mean(train_a20s):.2f} ± {np.std(train_a20s):.2f}")
    logging.info(f"Valid: RMSE {np.mean(val_losses):.3f} ± {np.std(val_losses):.3f} | "
          f"MAE {np.mean(val_maes):.3f} ± {np.std(val_maes):.3f} | "
          f"R2 {np.mean(val_r2s):.2f} ± {np.std(val_r2s):.2f} | "
          f"MARE {np.mean(val_mares):.3f} ± {np.std(val_mares):.3f} | "
          f"A20 {np.mean(val_a20s):.2f} ± {np.std(val_a20s):.2f}")
    logging.info(f"Test : RMSE {np.mean(test_losses):.3f} ± {np.std(test_losses):.3f} | "
          f"MAE {np.mean(test_maes):.3f} ± {np.std(test_maes):.3f} | "
          f"R2 {np.mean(test_r2s):.2f} ± {np.std(test_r2s):.2f} | "
          f"MARE {np.mean(test_mares):.3f} ± {np.std(test_mares):.3f} | "
          f"A20 {np.mean(test_a20s):.2f} ± {np.std(test_a20s):.2f}")

    # Save scatter plot
    if params['TRANSFER']:
        np.savetxt(os.path.join(TEXTFILES, f'transfer_{params["ARCH"]}-{params["POOLING_METHOD"]}-{params["TARGET_FEATURE_NAME"]}-label-pred.txt'), np.column_stack((y_true_test, y_pred_test)), fmt="%.4f")
    else:
        np.savetxt(os.path.join(TEXTFILES, f'{params["ARCH"]}-{params["POOLING_METHOD"]}-{params["TARGET_FEATURE_NAME"]}-label-pred.txt'), np.column_stack((y_true_test, y_pred_test)), fmt="%.4f")
    figt = go.Figure(data=go.Scatter(x=y_true_test, y=y_pred_test, mode='markers'))
    figt.update_layout(width=800, height=800,
                       xaxis_title=f'Original {params["TARGET_FEATURE_NAME"]}',
                       yaxis_title=f'Predicted {params["TARGET_FEATURE_NAME"]}')
    figt.add_shape(type="line", x0=min(y_true_test), y0=min(y_true_test),
                   x1=max(y_true_test), y1=max(y_true_test))
    if params['TRANSFER']:
        figt.write_image(os.path.join(Figures, f'transfer_{params["ARCH"]}-{params["POOLING_METHOD"]}-{params["TARGET_FEATURE_NAME"]}-pred-vs-label.png'), scale=2)
    else:
        figt.write_image(os.path.join(Figures, f'{params["ARCH"]}-{params["POOLING_METHOD"]}-{params["TARGET_FEATURE_NAME"]}-pred-vs-label.png'), scale=2)


def print_test_metrics(params, y_true_test, y_pred_test,repeats=5):
    """
    Evaluate MoleculeGNN baseline on original values (no normalization).
    Prints RMSE, MAE, R², MARE, A20 for train/val/test.
    """
    evaluator = RegressionMetric()
    test_losses, test_r2s, test_maes, test_mares, test_a20s = [], [], [], [], []

    for _ in range(repeats):
        test_losses.append(evaluator.root_mean_squared_error(y_true_test, y_pred_test))
        test_r2s.append(evaluator.coefficient_of_determination(y_true_test, y_pred_test))
        test_maes.append(evaluator.mean_absolute_error(y_true_test, y_pred_test))
        test_mares.append(evaluator.mean_absolute_percentage_error(y_true_test, y_pred_test))
        test_a20s.append(evaluator.a20_index(y_true_test, y_pred_test))

    print(f"\nFinal evaluation for Model {params['ARCH']}-{params['POOLING_METHOD']}")

    print(f"Test : RMSE {np.mean(test_losses):.3f} ± {np.std(test_losses):.3f} | "
          f"MAE {np.mean(test_maes):.3f} ± {np.std(test_maes):.3f} | "
          f"R2 {np.mean(test_r2s):.2f} ± {np.std(test_r2s):.2f} | "
          f"MARE {np.mean(test_mares):.3f} ± {np.std(test_mares):.3f} | "
          f"A20 {np.mean(test_a20s):.2f} ± {np.std(test_a20s):.2f}")


class GraphDataIO:
    """
    Utility class for saving and loading graph-related data (.pkl or .pkl.gz).
    Supports:
      - list of graphs
      - PyTorch Geometric Dataset
      - DataLoader
    Automatically moves GPU tensors to CPU before saving.
    """

    @staticmethod
    def _move_to_cpu(obj):
        """Recursively move tensors to CPU for safe pickling."""
        if isinstance(obj, Data):
            return obj.cpu()
        elif isinstance(obj, list):
            return [GraphDataIO._move_to_cpu(o) for o in obj]
        elif isinstance(obj, Dataset):
            return [GraphDataIO._move_to_cpu(o) for o in obj]
        elif isinstance(obj, DataLoader):
            return [GraphDataIO._move_to_cpu(data) for data in obj.dataset]
        return obj

    @staticmethod
    def save(obj, filename, compress=False):
        """
        Save a list, Dataset, or DataLoader to .pkl (or .pkl.gz if compress=True).
        Automatically moves all tensors to CPU.
        """
        # Normalize input to list of Data objects
        if isinstance(obj, DataLoader):
            data_to_save = list(obj.dataset)
        elif isinstance(obj, Dataset):
            data_to_save = list(obj)
        elif isinstance(obj, list):
            data_to_save = obj
        else:
            raise TypeError("Unsupported type. Expected list, Dataset, or DataLoader.")

        # Move all to CPU before pickling
        data_to_save = GraphDataIO._move_to_cpu(data_to_save)

        if compress or filename.endswith(".gz"):
            with gzip.open(filename if filename.endswith(".gz") else filename + ".gz", "wb") as f:
                pickle.dump(data_to_save, f)
        else:
            with open(filename, "wb") as f:
                pickle.dump(data_to_save, f)

        print(f"Saved {len(data_to_save)} graphs to '{filename}' (compressed={compress})")

    @staticmethod
    def load(filename):
        """Load a list of graphs (Data objects) from .pkl or .pkl.gz file."""
        if filename.endswith(".gz"):
            with gzip.open(filename, "rb") as f:
                data_list = pickle.load(f)
        else:
            with open(filename, "rb") as f:
                data_list = pickle.load(f)

        print(f"Loaded {len(data_list)} graphs from '{filename}'")
        return data_list

    @staticmethod
    def load_to_dataloader(filename, batch_size=32, shuffle=False):
        """Load and return as a PyTorch Geometric DataLoader."""
        data_list = GraphDataIO.load(filename)
        loader = DataLoader(data_list, batch_size=batch_size, shuffle=shuffle, num_workers=8, pin_memory=True, prefetch_factor=2)
        print(f"Created DataLoader with batch size {batch_size}")
        return loader


'''UTILITIES FOR TRANSFER MODELS'''

def load_checkpoint_into_encoder(encoder, ckpt_path, strict=False):
    sd = torch.load(ckpt_path, map_location='cpu')
    try:
        encoder.load_state_dict(sd, strict=strict)
    except Exception as e:
        # Try partial load (useful if checkpoint includes head)
        encoder.load_state_dict({k:v for k,v in sd.items() if k in encoder.state_dict()}, strict=False)

def freeze_encoder_layers(encoder):
    for p in encoder.parameters():
        p.requires_grad = False

def unfreeze_last_n_layers(encoder, n_layers):
    """
    Unfreeze the last n GNN conv layers. Works for both architectures.
    """
    # try module names
    if hasattr(encoder, 'conv_layers'):
        convs = encoder.conv_layers
    elif hasattr(encoder, 'convs'):
        convs = encoder.convs
    else:
        convs = []
    total = len(convs)
    start = max(0, total - n_layers)
    for i in range(start, total):
        for p in convs[i].parameters():
            p.requires_grad = True


# Embedding extraction
def extract_embeddings(encoder, loader, device='cpu', adapter=None):
    encoder.eval()
    embs = []
    ids = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            x = data.x
            if adapter is not None:
                x = adapter(x)
            _, emb = encoder(x, data.edge_index, getattr(data, 'edge_attr', None), data.batch, cond=getattr(data,'cond',None), global_feats=getattr(data,'global_feats',None))
            embs.append(emb.cpu())
            if hasattr(data, 'idx'):
                ids.extend(data.idx.cpu().tolist())
            else:
                ids.extend([None]*emb.size(0))
    return torch.cat(embs, dim=0).numpy(), ids


def train_transfer(model, train_loader, val_loader, device, epochs=50, lr_head=1e-3, lr_encoder=1e-5, finetune=False, grad_clip=1.0, save_path='transfer_best.pt'):
    device = torch.device(device)
    model.to(device)

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

    best_val = float('inf')
    for epoch in range(1, epochs+1):
        model.train()
        train_loss = 0.0
        n = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out, emb = model(data.x, data.edge_index, getattr(data, 'edge_attr', None), data.batch, cond=getattr(data,'cond',None), global_feats=getattr(data,'global_feats',None))
            target = data.y.view(-1).to(device)
            loss = F.mse_loss(out, target)
            loss.backward()
            # gradient clipping (clip only params that require grad)
            torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_norm=grad_clip)
            optimizer.step()

            train_loss += loss.item() * out.size(0)
            n += out.size(0)
        train_loss /= n if n>0 else 1.0

        # validation
        model.eval()
        val_loss = 0.0
        n = 0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                out, emb = model(data.x, data.edge_index, getattr(data, 'edge_attr', None), data.batch, cond=getattr(data,'cond',None), global_feats=getattr(data,'global_feats',None))
                target = data.y.view(-1).to(device)
                loss = F.mse_loss(out, target)
                val_loss += loss.item() * out.size(0)
                n += out.size(0)
        val_loss /= n if n>0 else 1.0
        scheduler.step(val_loss)

        logging.info(f"Epoch {epoch} | Train MSE: {train_loss:.4f} | Val MSE: {val_loss:.4f}")
        print(f"Epoch {epoch} | Train MSE: {train_loss:.4f} | Val MSE: {val_loss:.4f}")

        # save best
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), save_path)
            return model

    logging.info("Training finished. Best val MSE:", best_val)


class ModelSummaryVisuals:
    def __init__(self,model,batch):
        self.model = model
        self.batch = batch 

    def get_encoder(self,filepath=None):
        self.encoder = self.model.encoder
        self.model_graph = draw_graph(
            self.encoder,
            input_data=(self.batch.x, self.batch.edge_index, self.batch.edge_attr, self.batch.batch, self.batch.cond, self.batch.global_feats),
            depth=3,
            graph_name="NNConvModel",
            #expand_nested=True,    # optional: show inside modules
        )
        # Render to file (optional)
        self.model_graph.visual_graph.render(filepath)

    def get_transfer_model(self,filepath=None):
        self.model_graph = draw_graph(
            self.model,
            input_data=(self.batch.x, self.batch.edge_index, self.batch.edge_attr, self.batch.batch, self.batch.cond, self.batch.global_feats),
            depth=3,
            graph_name="NNConvModel",
            #expand_nested=True,    # optional: show inside modules
        )
        # Render to file (optional)
        self.model_graph.visual_graph.render(filepath)


import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
from torch.nn import functional as F

class ModelInsightVisualizer:
    """
    A unified class for visualizing model latent space and feature sensitivity.
    Works with PyTorch or PyTorch-Geometric models.
    """

    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.model.eval()

    @torch.no_grad()
    def extract_latent_embeddings(self, loader, layer_fn=None):
        """
        Extract latent representations from a model.
        layer_fn: function(model, batch) -> embedding
                  If None, use model.encoder output if available.
        """
        embeddings, labels = [], []
        for batch in loader:
            batch = batch.to(self.device)
            if layer_fn is not None:
                emb = layer_fn(self.model, batch)
            elif hasattr(self.model, 'encoder'):
                emb = self.model.encoder(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            else:
                raise ValueError("Provide a layer_fn or ensure model has an encoder.")
            embeddings.append(emb.detach().cpu())
            labels.append(batch.y.detach().cpu())
        return torch.cat(embeddings), torch.cat(labels)

    def visualize_latent_space(self, embeddings, labels=None, method='tsne', title=None):
        """
        Visualize latent embeddings using t-SNE, UMAP, or PCA.
        """
        if method == 'tsne':
            reducer = TSNE(n_components=2, perplexity=30, random_state=42)
        elif method == 'umap':
            reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
        elif method == 'pca':
            reducer = PCA(n_components=2)
        else:
            raise ValueError("method must be one of ['tsne', 'umap', 'pca']")

        reduced = reducer.fit_transform(embeddings.numpy())

        plt.figure(figsize=(7, 6))
        if labels is not None:
            plt.scatter(reduced[:, 0], reduced[:, 1], c=labels.numpy(), cmap='viridis', s=25)
            plt.colorbar(label='Property Value')
        else:
            plt.scatter(reduced[:, 0], reduced[:, 1], s=25)
        plt.title(title or f"Latent Space Visualization ({method.upper()})")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.show()

    def analyze_feature_sensitivity(self, loader, feature_names=None, num_batches=10):
        """
        Compute feature sensitivities (gradient-based importance).
        Returns mean absolute gradient per input feature.
        """
        grads = []
        for i, batch in enumerate(loader):
            if i >= num_batches:
                break
            batch = batch.to(self.device)
            batch.x.requires_grad_(True)
            out = self.model(batch)
            loss = out.mean()
            grads_batch = torch.autograd.grad(loss, batch.x, retain_graph=False)[0]
            grads.append(grads_batch.abs().mean(dim=0).detach().cpu().numpy())
        mean_grads = np.mean(grads, axis=0)

        plt.figure(figsize=(10, 4))
        x_labels = feature_names or [f"f{i}" for i in range(len(mean_grads))]
        plt.bar(x_labels, mean_grads)
        plt.xticks(rotation=45, ha='right')
        plt.title("Feature Sensitivity (Mean |∂y/∂x|)")
        plt.ylabel("Importance")
        plt.tight_layout()
        plt.show()

        return mean_grads

#   visualizer = ModelInsightVisualizer(model, device='cuda')
#   
#   # Extract latent embeddings for IL and DES datasets
#   il_embeddings, il_labels = visualizer.extract_latent_embeddings(il_loader)
#   des_embeddings, des_labels = visualizer.extract_latent_embeddings(des_loader)
#   
#   # Combine and visualize latent space overlap
#   combined_emb = torch.cat([il_embeddings, des_embeddings])
#   combined_labels = torch.cat([torch.zeros(len(il_labels)), torch.ones(len(des_labels))])
#   visualizer.visualize_latent_space(combined_emb, combined_labels, method='umap',
#                                     title="IL vs DES Latent Space (UMAP)")
#   
#   # Feature sensitivity analysis (chemical insight)
#   visualizer.analyze_feature_sensitivity(il_loader, feature_names=feature_names)
