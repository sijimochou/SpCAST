import os

import torch
import torch.nn as nn

import numpy as np
import shap
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm

from .SpCAST import SpCAST, GeneExpressionDataset
from .utils import *

import matplotlib.colors as mcolors
import scanpy as sc
import anndata as ad


class WrappedModel(nn.Module):
    def __init__(self, model):
        super(WrappedModel, self).__init__()
        self.model = model

    def forward(self, x):
        # Assuming the original model returns a tuple, extract the tensor you need
        combined, combined_class, reconstruct = self.model(x)
        probs = torch.softmax(combined_class, dim=1)
        return probs  # If it's already a tensor, return it as-is
    
    
def feature_importance_identification(
    scRNA_data,               # Single-cell RNA-seq data (e.g., AnnData object)
    spRNA_data,               # Spatial RNA-seq data (e.g., AnnData object)
    biomarkers_list=None,          # List of marker genes to be displayed on the heatmap
    batch_size=1,           # Batch size for SHAP computation (not used here but can be extended)
    device=None,              # Device for PyTorch operations (e.g., 'cuda' or 'cpu')
    save_model_path='../model', # Path to the directory where the model is saved
    save_model_name='model_params.pth' # Name of the saved model file
):
    """
    This function computes SHAP values for spatial RNA-seq data using a pre-trained model
    and visualizes the importance of genes for different cell types in a heatmap.

    Parameters:
    - scRNA_data: Single-cell RNA-seq data (used as background for SHAP).
    - spRNA_data: Spatial RNA-seq data (used for SHAP computation).
    - biomarkers_list: List of genes to display on the heatmap (marker genes).
    - batch_size: Batch size for SHAP computation (default: 1).
    - device: PyTorch device for computation (e.g., 'cuda' or 'cpu').
    - save_model_path: Path to the directory where the model is saved (default: '../model').
    - save_model_name: Name of the saved model file (default: 'model_params.pth').

    Returns:
    None. Displays a heatmap of SHAP values for the specified genes across cell types.
    """
    # Ensure the model save path exists
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)

    # Check if the saved model file exists
    save_name = os.path.join(save_model_path, save_model_name)
    if not os.path.isfile(save_name):
        raise FileNotFoundError(f"Model file not found: {save_name}")

    # Load the pre-trained model
    checkpoint = torch.load(save_name, map_location=device)
    model_params = checkpoint['model_params']  # Extract parameters
    label_mapping = checkpoint['label_mapping']
    y_encoder_label = [label_mapping[_] for _ in spRNA_data.obs['cell_type'].tolist()]
    
    ### only return combined_class
    model = SpCAST(**model_params).to(device)  # Initialize model with parameters
    model.load_state_dict(checkpoint['model_state_dict'])  # Load weights
    model = WrappedModel(model).to(device)
    
    model.eval()  # Set the model to evaluation mode

    # Prepare background data (single-cell RNA-seq) and test data (spatial RNA-seq)
    background = torch.from_numpy(scRNA_data.X[np.random.choice(scRNA_data.X.shape[0], 1000, replace=False)]
).float()[:1000].to(device)  # Use the first 1000 samples as background
    test_data = torch.from_numpy(spRNA_data.X).float().to(device)          # Spatial RNA-seq data for SHAP computation

    # Compute SHAP values using DeepExplainer
    print("Compute SHAP values using DeepExplainer ...")
    explainer = shap.DeepExplainer(model, background)
    
    shap_values_list = []  
    for i in tqdm(range(0, len(test_data), batch_size)):
        test_batch = test_data[i:i+batch_size]
        y_label = y_encoder_label[i]
        shap_values = explainer.shap_values(test_batch, check_additivity=False)
        shap_values_list.append(shap_values[y_label])
    
    if isinstance(shap_values_list[0], list):  
        shap_values_merged = [np.concatenate([batch[i] for batch in shap_values_list], axis=0)
                              for i in range(len(shap_values_list[0]))]
    else:  
        shap_values_merged = np.concatenate(shap_values_list, axis=0)

    print("Complete compute ...")
    
    # Check the shape of SHAP values
    num_samples, num_features = shap_values_merged.shape
    if num_features != len(spRNA_data.var_names):
        raise ValueError("Number of features in SHAP values does not match the number of genes in 'spRNA_data.var_names'")
    
    # get color mapping
    fig = sc.pl.embedding(spRNA_data, basis='spatial', color='cell_type', show=False, return_fig=True)
    cell_type_color_mapping = dict(zip(spRNA_data.obs.cell_type.cat.categories.tolist(), spRNA_data.uns['cell_type_colors']))
    plt.close(fig)
    
    # Create a DataFrame for SHAP values
    df_shap = pd.DataFrame(shap_values_merged, columns=spRNA_data.var_names.tolist())
    df_shap.index = spRNA_data.obs_names.tolist()
    df_shap['cell_type'] = spRNA_data.obs['cell_type']  # Add cell type information from spatial data

    # Sort SHAP values by cell type
    df_shap_sorted = df_shap.sort_values(by='cell_type')

    # Automatically select top 3 features per cell type if biomarkers_list is not provided
    if biomarkers_list is None:
        top_genes = []
        for cell_type in df_shap_sorted['cell_type'].unique():
            cell_type_shap = df_shap_sorted[df_shap_sorted['cell_type'] == cell_type].iloc[:, :-1]  # Exclude 'cell_type' column
            mean_shap_values = cell_type_shap.mean(axis=0)  # Compute mean SHAP values for each gene
            top_genes_for_cell_type = mean_shap_values.nlargest(3).index.tolist()  # Select top 3 genes
            top_genes.extend(top_genes_for_cell_type)  # Add top genes to the list
        biomarkers_list = list(set(top_genes))  # Remove duplicates and preserve unique genes
        print(f"Selected top genes for heatmap: {biomarkers_list}")

    # Plot the heatmap
    shap_plot = df_shap_sorted.iloc[:, :-1].T.loc[biomarkers_list, :]  # Transpose: rows are genes, columns are samples
    vmin, vmax = np.nanpercentile(shap_plot.values.flatten(), [1, 99])
    
#     plt.figure(figsize=(12, 8))
#     ax = sns.heatmap(
#             shap_plot,  
#             cmap="RdBu_r",                 # Red-blue colormap
#             vmin=min(vmin,-vmax),
#             vmax=max(-vmin,vmax),
#             xticklabels=False,             # Hide sample labels on the x-axis
#             yticklabels=biomarkers_list    # Use biomarkers_list as labels for the y-axis
#         )
#     col_colors = [mcolors.to_rgba(cell_type_color_mapping[_]) for _ in df_shap_sorted['cell_type'].astype(str).tolist()]
#     annotation_bar = np.array([col_colors])  
#     ax.imshow(annotation_bar, aspect="auto", extent=[0, shap_plot.shape[0], shap_plot.shape[1], shap_plot.shape[1] + 5]) 
    
#     plt.title("SHAP Values Heatmap (Grouped by Cell Types)")
#     plt.xlabel("Samples (Grouped by Cell Types)")
#     plt.ylabel("Genes")
#     plt.tight_layout()
#     plt.show()
    
    shap_values_df = pd.DataFrame(shap_values_merged, columns=spRNA_data.var_names.tolist())
    shap_values_df.index = spRNA_data.obs_names.tolist()
    
    temp = sc.AnnData(shap_values_df)
    temp.obs['cell_type'] = spRNA_data.obs['cell_type']
    # plt.figure(figsize=(12, 8))
    sc.pl.heatmap(
            temp,
            biomarkers_list,
            cmap="RdBu_r",                 # Red-blue colormap
            vmin=min(vmin,-vmax),
            vmax=max(-vmin,vmax),
            groupby='cell_type',
            var_group_positions=[],
            swap_axes=True,
            figsize=(10, 7),
        )
    del temp
    # col_colors = [mcolors.to_rgba(cell_type_color_mapping[_]) for _ in df_shap_sorted['cell_type'].astype(str).tolist()]
    # annotation_bar = np.array([col_colors])  
    # ax.imshow(annotation_bar, aspect="auto", extent=[0, shap_plot.shape[0], shap_plot.shape[1], shap_plot.shape[1] + 5]) 
    
    # plt.title("SHAP Values Heatmap (Grouped by Cell Types)")
    # plt.xlabel("Samples (Grouped by Cell Types)")
    # plt.ylabel("Genes")
    # plt.tight_layout()
    # plt.show()
    
    return shap_values_df
