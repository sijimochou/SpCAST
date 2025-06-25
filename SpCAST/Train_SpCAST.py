import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.sparse as sp
from scipy import stats
from sklearn.preprocessing import LabelEncoder
import os

import multiprocessing as mp

from .SpCAST import SpCAST, GeneExpressionDataset
from .utils import *

import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.autograd import Function
import torch.backends.cudnn as cudnn
cudnn.deterministic = True
cudnn.benchmark = True
import torch.nn.functional as F
import torch.nn as nn
import scanpy as sc
from itertools import cycle
import gc

# ------------------------- Data Preparation -------------------------
def prepare_data(scRNA_data, spRNA_data, batch_size=128, balance_classes=True):
    """
    Prepares train and test datasets and DataLoaders for scRNA and spatial RNA data with class balancing.

    Args:
        scRNA_data: Single-cell RNA-seq AnnData object.
        spRNA_data: Spatial RNA-seq AnnData object.
        batch_size: Batch size for DataLoader.
        balance_classes: Whether to balance classes in source data using weighted sampling.

    Returns:
        train_loader: DataLoader for training data.
        test_loader: DataLoader for testing data.
        input_dim: Input dimensionality (number of genes).
        num_classes: Number of unique cell types (classes).
        label_encoder: Fitted LabelEncoder for cell type labels.
    """
    if 'cell_type' not in scRNA_data.obs:
        raise ValueError("'cell_type' column is required in scRNA_data.obs.")
    
    label_encoder = LabelEncoder()

    # Prepare training data (scRNA-seq)
    if type(scRNA_data.X) != np.ndarray:
        X_train = scRNA_data.X.toarray()
    else:
        X_train = scRNA_data.X
        
    y_train = scRNA_data.obs['cell_type'].values  # Cell type labels
    y_train = label_encoder.fit_transform(y_train)  # Encode string labels to integer
    
    if type(spRNA_data.X) != np.ndarray:
        X_test = spRNA_data.X.toarray()
    else:
        X_test = spRNA_data.X
    
    source_dataset = GeneExpressionDataset(X_train, y_train)
    target_dataset = GeneExpressionDataset(X_test)

    if balance_classes:
        class_counts = np.bincount(y_train)
        print(f"Class distribution: {class_counts}")
        
        class_weights = 1. / np.log1p(class_counts)
        class_weights = class_weights / class_weights.sum()
        
        print(f"Class weights: {class_weights}")

        sample_weights = class_weights[y_train]
        
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        
        train_source_dataloader = DataLoader(
            source_dataset, 
            batch_size=batch_size, 
            sampler=sampler
        )
    else:
        train_source_dataloader = DataLoader(
            source_dataset, 
            batch_size=batch_size, 
            shuffle=True
        )

    train_target_dataloader = DataLoader(target_dataset, batch_size=batch_size, shuffle=True)
    
    test_source_dataloader = DataLoader(source_dataset, batch_size=batch_size, shuffle=False)
    test_target_dataloader = DataLoader(target_dataset, batch_size=batch_size, shuffle=False)

    input_dim = X_train.shape[1]  # Number of genes (features)
    num_classes = len(set(y_train))  # Number of unique cell types

    return train_source_dataloader, train_target_dataloader, test_source_dataloader, test_target_dataloader, input_dim, num_classes, label_encoder


# ------------------------- Model Training -------------------------
def gaussian_kernel(x, y, sigma=5.0):
    """
    Args:
        x: Tensor (n_samples, n_features)
        y: Tensor (m_samples, n_features)
        sigma: 
    Returns: (n_samples, m_samples)
    """
    x_size, y_size = x.size(0), y.size(0)
    dim = x.size(1)
    x = x.unsqueeze(1)  # (n, 1, d)
    y = y.unsqueeze(0)  # (1, m, d)
    tiled_x = x.expand(x_size, y_size, dim)
    tiled_y = y.expand(x_size, y_size, dim)
    squared_diff = torch.sum((tiled_x - tiled_y)**2, dim=2)
    return torch.exp(-squared_diff / (2 * sigma**2))


def mmd_loss(x, y, kernel='multi-scale', sigma_list=None):
    if sigma_list is None:
        sigma_list = [0.5, 1., 2., 5., 10.]
    
    xx = torch.zeros_like(gaussian_kernel(x, x, sigma=1.0))
    yy = torch.zeros_like(gaussian_kernel(y, y, sigma=1.0))
    xy = torch.zeros_like(gaussian_kernel(x, y, sigma=1.0))

    if kernel == 'multi-scale':
        for sigma in sigma_list:
            xx = xx + gaussian_kernel(x, x, sigma)
            yy = yy + gaussian_kernel(y, y, sigma)
            xy = xy + gaussian_kernel(x, y, sigma)
        
        scale_num = len(sigma_list) + 1 
        xx = xx / scale_num
        yy = yy / scale_num 
        xy = xy / scale_num

    mmd = (xx.mean() - 2 * xy.mean() + yy.mean())
    return torch.clamp(mmd, min=0)


def train_SpCAST(scRNA_data, spRNA_data, latent_dim1=30, latent_dim2=20, n_epochs=10, loss_step=1, 
                 batch_size=128, lr=0.001, gradient_clipping=5.0, weight_decay=0.0001, 
                 verbose=True, random_seed=2025, device=None, save_model=True, save_model_path='../model', save_model_name='model_params.pth'):
    """
    Train the SpCAST model with single-cell and spatial RNA-seq data.

    Args:
        scRNA_data: Single-cell RNA-seq AnnData object.
        spRNA_data: Spatial RNA-seq AnnData object.
        latent_dim1: Dimensionality of the latent space for reconstruction.
        latent_dim2: Dimensionality of the latent space for cell type class and domain class.
        n_epochs: Number of training epochs.
        loss_step: Interval for printing loss during training.
        batch_size: Batch size for training.
        lr: Learning rate for optimizer.
        gradient_clipping: Maximum value for gradient clipping.
        weight_decay: Weight decay for optimizer.
        verbose: If True, prints additional information.
        random_seed: Random seed for reproducibility.
        device: Device for computation (e.g., 'cuda' or 'cpu').

    Returns:
        model: Trained SpCAST model.
        test_loader: DataLoader for testing data.
    """
    
    # seed_everything()
    seed=random_seed
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    if verbose:
        print('Size of scRNA-seq data Input: ', scRNA_data.shape)
        print('Size of spRNA-seq data Input: ', spRNA_data.shape)
        
    # Prepare data and model
    train_source_dataloader, train_target_dataloader, test_source_dataloader, test_target_dataloader, input_dim, num_classes, label_encoder = prepare_data(scRNA_data, spRNA_data, batch_size=batch_size)

    model = SpCAST(input_dim=input_dim, latent_dim1=latent_dim1, latent_dim2=latent_dim2, num_classes=num_classes).to(device)
            
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay) 
    
    classification_criterion = nn.CrossEntropyLoss()
    recon_criterion = nn.MSELoss()
    
    print("Train the model..........")
    # Train the model
    model.train()
    for epoch in tqdm(range(n_epochs), desc="Training"):
        total_classification_loss = 0
        total_opt_loss = 0
        total_recon_loss = 0
        correct = 0
        total = 0
        for i, ((source_data, source_label), (target_data)) in enumerate(zip(cycle(train_source_dataloader), train_target_dataloader)):         
            source_data = source_data.to(device)   
            source_label = source_label.to(torch.long).to(device)
            target_data = target_data.to(device)

            mixed_data = torch.cat([source_data, target_data], dim=0).squeeze()          
                 
            # Forward Propagation                               
            combined, combined_class, reconstruct = model(mixed_data)
            source_combined, target_combined = torch.split(combined, [source_data.size(0), target_data.size(0)], dim=0)
            source_combined_class, target_combined_class = torch.split(combined_class, [source_data.size(0), target_data.size(0)], dim=0)
        
            # Compute Loss                               
            classification_loss = classification_criterion(source_combined_class, source_label)         
            opt_loss = mmd_loss(source_combined, target_combined)
            recon_loss = recon_criterion(mixed_data, reconstruct)

            total_loss = classification_loss + 0.03 * opt_loss + 0.01 * recon_loss

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            optimizer.step()

            total += source_label.size(0)
            correct += (source_combined_class.argmax(dim=1) == source_label).sum().item()

            total_classification_loss += classification_loss.item()
            total_opt_loss += opt_loss.item()       
            total_recon_loss += recon_loss.item()
            
        if verbose:
            if (epoch+1) % loss_step == 0:
                train_accuracy = correct / total
                print(f'Epoch:{epoch+1} MMD Loss:{total_opt_loss:.3f}  Class Loss:{total_classification_loss:.3f}  Recon loss: {total_recon_loss*100:.3f}  Train_Accuracy: {train_accuracy*100:.2f}% ')
            
    del train_source_dataloader 
    del train_target_dataloader
    gc.collect()
            
    print("Testing the model..........")
    # Test the model                                       
    model.eval()
    all_sp_preds = []
    all_sc_preds = []
    
    all_sp_obsm = []
    all_sc_obsm = []
    
    all_sp_reconstruct = []
    all_sc_reconstruct = []   
    
    prb_matrix = []
    with torch.no_grad():
        for data in tqdm(test_target_dataloader, desc="Testing spRNA data"):
            data = data.to(device)
            combined, combined_class, reconstruct = model(data)
            preds = combined_class.argmax(dim=1)
            all_sp_preds.extend(preds.cpu().numpy())
            all_sp_obsm.extend(combined.cpu().numpy())
            all_sp_reconstruct.extend(reconstruct.cpu().numpy())
            
            probabilities = torch.softmax(combined_class, dim=1)
            prb_matrix.append(probabilities.cpu().numpy())
            
        prb_matrix = np.vstack(prb_matrix)
          
        for data, _ in tqdm(test_source_dataloader, desc="Testing scRNA data"):
            data = data.to(device)
            combined, combined_class, reconstruct = model(data)
            preds = combined_class.argmax(dim=1)
            all_sc_preds.extend(preds.cpu().numpy())
            all_sc_obsm.extend(combined.cpu().numpy())
            all_sc_reconstruct.extend(reconstruct.cpu().numpy())
            
    # Add predictions to spatial RNA-seq data
    spRNA_data.obs['SpCAST_predicted'] = label_encoder.inverse_transform(all_sp_preds)
    scRNA_data.obs['SpCAST_predicted'] = label_encoder.inverse_transform(all_sc_preds)
    
    spRNA_data.obsm['SpCAST'] = np.array(all_sp_obsm)
    scRNA_data.obsm['SpCAST'] = np.array(all_sc_obsm)
    
    spRNA_data.layers['SpCAST_reconstruct'] = np.array(all_sp_reconstruct)
    scRNA_data.layers['SpCAST_reconstruct'] = np.array(all_sc_reconstruct)
    
    class_names = label_encoder.classes_  

    prob_df = pd.DataFrame(
        data=prb_matrix,
        columns=class_names  
    )
    prob_df['SpCAST_predicted'] = label_encoder.inverse_transform(all_sp_preds)
    
    if save_model:
        if not os.path.exists(save_model_path):  
            os.makedirs(save_model_path)
        save_name = os.path.join(save_model_path, save_model_name)
        label_mapping = {label: index for index, label in enumerate(label_encoder.classes_)}
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_params': {
                'input_dim': input_dim,
                'latent_dim1': latent_dim1,
                'latent_dim2': latent_dim2,
                'num_classes': num_classes
            },
            'label_mapping':label_mapping
        }, save_name)
    return prob_df
