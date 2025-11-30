# Graph Neural Network for Ionic Liquids and-Deep Eutectics Solvent Property Prediction - A Cheminformatics Study
A Deep Learning Framework for Predicting Physicochemical Properties of Ionic Liquids and Deep Eutectic Solvents using Message Passing Graph Neural Networks (NNConv)

## Overview

This project focuses on developing a **Graph Neural Network (GNN)** model based on the **Neural Network Convolution (NNConv)** architecture for the **prediction of key physicochemical properties of Ionic Liquids (ILs)**.  

It represents an intersection of **computational chemistry**, **cheminformatics**, and **deep learning**, leveraging molecular graph representations and message passing to capture intricate structural and electronic interactions within IL systems.

---

## Motivation

Ionic liquids (ILs) and Deep Eutectic Solvents (DESs) are versatile materials with broad applications in **green chemistry**, **energy storage**, **lubrication**, and **separation processes**. However, their vast chemical space makes experimental property measurement infeasible for all combinations.
This project aims to address that challenge by developing a **data-driven predictive framework** that learns directly from molecular structures to estimate physicochemical properties.

---

## Key Features

- **NNConv-based GNN Architecture:**  
  Implements a flexible **message passing neural network** that dynamically learns edge-conditioned transformations between atom pairs.

- **Transfer Learning Extension:**  
  The trained IL model was fine-tuned for training and evaluating the **Deep Eutectic Solvents (DESs)**.

- **Comprehensive Cheminformatics Integration:**  
  Utilizes tools such as **RDKit** and **OpenBabel** for:
  - Molecular graph construction  
  - Node features and global features generation  
  - SMILES parsing and molecular standardization  

- **Data Pipeline and Preprocessing:**  
  - Accesses and preprocesses IL property data directly from the **NIST ILThermo database**.  
  - Includes data cleaning, feature normalization, and dataset splitting utilities.  
  - Handles both **cation-anion pair representation** and global system descriptors.

- **End-to-End Reproducibility:**  
  Provides all scripts and utilities to:
  - Train the IL model
  - Transfer Learning of the IL model to DESs  
  - Evaluate model performance  
  - Visualize learning curves and prediction accuracy

- **Parameter Settings:**
  - All the settings used in this work for model architecture, data preprocessing, etc. are provided in the params.yaml file in code folder

---

## Model Architecture

The core architecture uses the **NNConv** layer from PyTorch Geometric to learn edge-conditioned message functions, enabling adaptive feature propagation across molecular graphs.

**Highlights:**
- Node embeddings represent atom-level features (atomic number, degree, valence, etc.)  
- Edge embeddings represent bond information (type, conjugation, ring status, etc.)  
- Global features incorporate molecular-level descriptors (e.g., polarity, molar mass, number of hydrogen donors and acceptors)  
- Final readout combines local and global embeddings through fully connected layers

---

## Technologies Used

| Category | Tools/Libraries | 
|-----------|-----------------|
| **Deep Learning Framework** | PyTorch, PyTorch Geometric |
| **Cheminformatics** | RDKit, OpenBabel |
| **Data Handling** | pandas, numpy, scikit-learn, Scipy |
| **Visualization** | matplotlib, seaborn | plotly
| **Database Access** | NIST ILThermo data integration |

---

## Datasets

- **Primary ILs data (training, validaton, evaluation) from:** [NIST ILThermo Database](https://ilthermo.boulder.nist.gov/)
- **DESs density data source for transfer learning (training, validation, evaluation) from:** https://www.sciencedirect.com/science/article/pii/S0378381222002916?via%3Dihub
- **Additional ILs density data for just testing on trained IL model from:** https://pubs.acs.org/doi/suppl/10.1021/acs.iecr.9b00130/suppl_file/ie9b00130_si_001.zip
- **Properties Modeled:** density, viscosity, conductivity, refractive index. Only density prediction was implemented for transfer learning for now.
- **Data Processing:** includes duplicate removal, removing outliers, removing undefined liquids, and molecular pairing, etc..

---

## Results and Insights

- The models demonstrate strong predictive accuracy (high R², low MARE) across multiple datasets.

- Transfer learning to Deep Eutectic Solvents shows promising adaptability demonstrating that the model trained on ILs preserved most structure–property information even when moved to a new material class (DESs).

| Datasets | R2 | RMSE | MAE | MARE | A20 |
|-----------|-----------------|-----------------|-----------------|-----------------|-----------------|
| **Primary ILs training data** | 0.99 | 13.16 | 8.02 | 0.007 | 1.00 |
| **Primary ILs validation data** | 0.99 | 13.81 | 8.10 | 0.007 | 1.00 |
| **Primary ILs Testing data** | 0.99 | 14.82 | 8.31 | 0.007 | 1.00 |
| **Additional ILs Testing data** | 0.96 | 31.31 | 18.9 | 0.015 | 1.00 |
| **DESs training data (transfer learning)** | 0.94 | 22.08 | 16.76 | 0.015 | 1.00 |
| **DESs validation data (transfer learning)** | 0.94 | 20.67 | 16.69 | 0.015 | 1.00 |
| **DESs Testing data (transfer learning)** | 0.93 | 23.51 | 17.17 | 0.015 | 1.00 |


---

## Project Impact

**This project has strengthened my expertise in:**

- Machine Learning for Molecular Systems

- Graph Representation Learning

- Cheminformatics and Data Engineering

- Database Integration and Automation

- Scientific Computing and Reproducible Research

**It also lays the groundwork for applying AI-driven design principles to novel ionic liquid discovery and optimization.**

## Folders and Description
``` 
code/
├── architecture.py
├── prepare_dataset.py
├── training.py
├── utils.py
├── draw_molecules.py
├── predict.ipynb
├── params.yaml
└── experiment.py
``` 
