import pandas as pd
import numpy as np
import scipy.sparse as sp
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torch
import scanpy as sc
from sklearn.neighbors import NearestNeighbors, kneighbors_graph, NearestNeighbors, KDTree, BallTree
from sklearn.metrics import pairwise_distances
from scipy.sparse import issparse, isspmatrix_csr, csr_matrix, spmatrix
import scipy
from tqdm import tqdm

# ------------------------- Preprocessing and Integration -------------------------
def preprocess_scRNA_data(adata, min_genes_in_cells=3, min_counts=300, min_genes=600, hvg_num=30000):
    """
    Preprocess an scRNA AnnData object (filtering, normalization, and selecting HVGs).

    Args:
        adata (AnnData): 
            AnnData object containing single-cell RNA-seq data to preprocess.
        min_genes_in_cells (int, optional): 
            Minimum number of cells that must express a gene for it to be kept. Genes expressed in fewer cells will be filtered out. 
            Default is 3.
        min_counts (int, optional): 
            Minimum number of total counts required for a cell to be kept. Cells with fewer counts will be filtered out. 
            Default is 300.
        min_genes (int, optional): 
            Minimum number of genes expressed in a cell for it to be kept. Cells expressing fewer genes will be filtered out. 
            Default is 600.
        hvg_num (int, optional): 
            Number of highly variable genes (HVGs) to select based on variability metrics. 
            Default is 30000.

    Returns:
        AnnData: 
            Preprocessed AnnData object after filtering, normalization, and HVG selection.
            
    Notes: 
        - This function filters low-quality cells and genes based on several thresholds.
        - It normalizes raw counts if the input matrix is integer-based, otherwise skips normalization.
        - Highly variable genes are selected using the Seurat V3 method.
    """
    print(f"Preprocessing scRNA-seq data ...")
    sc.pp.filter_genes(adata, min_cells=min_genes_in_cells)
    
    # Filter mitochondrial genes
    adata.var['mt_gene'] = adata.var_names.str.startswith(('MT-', 'mt-'))
    if adata.var['mt_gene'].any():
        adata = adata[:, ~adata.var.mt_gene].copy()
    
    # Filter cells based on minimum counts and minimum expressed genes
    before = adata.shape[0]
    sc.pp.filter_cells(adata, min_counts=min_counts)
    sc.pp.filter_cells(adata, min_genes=min_genes)
    
    print(f"scRNA Data before filtering: {before} cells  ||  scRNA Data after filtering: {adata.shape[0]} cells")

    # Normalize counts if data is raw (integer matrix)
    if is_integer_matrix(adata.X):
        print(f"Normalizing and log1p raw scRNA counts...")
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    else:
        print(f"scRNA Data already normalized. Skipping normalization.")
        
    # Select highly variable genes (HVGs)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=hvg_num, subset=True)
    print(f"sc data nonzero expression ratio: {np.count_nonzero(adata.X.toarray()) / adata.X.toarray().size * 100:.2f}%")
    print("-----------------------------------------------")
              
    return adata


def preprocess_spRNA_data(adata, min_genes_in_cells=3, hvg_num=30000, Enhance=True):
    """
    Preprocess an spRNA AnnData object (filtering, normalization, and selecting HVGs).
    Args:
        adata: AnnData object to preprocess.
        hvg_num: Number of highly variable genes to select on each data.
        min_genes_in_cells: Minimum number of cells expressing a gene for filtering.
        dataset_name: Name of the dataset (for logging).
    Returns:
        Preprocessed AnnData object.
    """
    print(f"Preprocessing spRNA-seq data ...")
    sc.pp.filter_genes(adata, min_cells=min_genes_in_cells)
    
    adata.var['mt_gene'] = adata.var_names.str.startswith(('MT-', 'mt-'))
    if adata.var['mt_gene'].any():
        adata = adata[:, ~adata.var.mt_gene].copy()
    
    if is_integer_matrix(adata.X):
        print(f"Normalizing and log1p raw spRNA counts...")
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    else:
        print(f"spRNA Data already normalized. Skipping normalization.")
        
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=hvg_num, subset=True)

    print("-----------------------------------------------")
    if Enhance:
        print(f"Enhance spRNA Data expression ...")

        spatial_k=30
        tree = KDTree(adata.obsm['spatial'], leaf_size=2)
        _, indices = tree.query(adata.obsm['spatial'], k=spatial_k + 1)
        indices = indices[:, 1:]
        spatial_weight = np.zeros((adata.obsm['spatial'].shape[0], adata.obsm['spatial'].shape[0]))
        for i in tqdm(range(indices.shape[0])):
            ind = indices[i]
            for j in ind:
                spatial_weight[i][j] = 1
        
        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=2000)
        data = adata[:, adata.var['highly_variable']].X.copy()
        n_components=30
        pca = PCA(n_components=n_components)
        if isinstance(data, np.ndarray):
            data_pca = pca.fit_transform(data)
        elif isinstance(data, csr_matrix):
            data = data.toarray()
            data_pca = pca.fit_transform(data)
        gene_correlation = 1 - pairwise_distances(data_pca, metric="correlation")

        gene_matrix = adata.X.copy().toarray()
        neighbour_k=4
        weights = spatial_weight * gene_correlation
        weights_list = []
        final_coordinates = []
        with tqdm(total=len(adata), desc="Enhance gene expression data using spatial coordinates") as pbar:
            for i in range(adata.shape[0]):
                current_spot = weights[i].argsort()[-neighbour_k:][:neighbour_k - 1]
                spot_weight = weights[i][current_spot]
                spot_matrix = gene_matrix[current_spot]
                if spot_weight.sum() > 0:
                    spot_weight_scaled = (spot_weight / spot_weight.sum())
                    weights_list.append(spot_weight_scaled)
                    spot_matrix_scaled = np.multiply(spot_weight_scaled.reshape(-1, 1), spot_matrix)
                    spot_matrix_final = np.sum(spot_matrix_scaled, axis=0)
                else:
                    spot_matrix_final = np.zeros(gene_matrix.shape[1])
                    weights_list.append(np.zeros(len(current_spot)))
                final_coordinates.append(spot_matrix_final)
                pbar.update(1)
                
        X_enhanced = gene_matrix + 0.2 * np.array(final_coordinates).astype(float)

        original_nonzero = np.count_nonzero(data) / data.size
        enhanced_nonzero = np.count_nonzero(X_enhanced) / X_enhanced.size

        print(f"sp data original nonzero expression ratio: {original_nonzero * 100:.2f}%    sp data enhanced nonzero expression ratio: {enhanced_nonzero * 100:.2f}%")
        adata.X = sp.csr_matrix(X_enhanced.astype(np.float32).copy())

        del data
        del data_pca
        del X_enhanced
    
    else:
        print(f"sp data nonzero expression ratio: {np.count_nonzero(adata.X.toarray()) / adata.X.toarray().size * 100:.2f}%")
    
    return adata

    
def concatenate_shared_genes(scRNA_data, spRNA_data, final_n_top_genes=2000, include_markers=True):
    """
    Concatenate scRNA and spatial RNA data based on shared genes.
    Args:
        scRNA_data: Single-cell RNA-seq data (AnnData object).
        spRNA_data: Spatial RNA-seq data (AnnData object).
        final_n_top_genes: final ntop HVG in concatenated AnnData.
        include_markers: also use scRNA data cell type marker genes for training.
    Returns:
        scRNA_data, spRNA_data, concatenated AnnData object.
    """
    
    shared_genes = list(set(scRNA_data.var_names) & set(spRNA_data.var_names))
    print(f"We have {len(shared_genes)} shared genes on scRNA and spatial RNA data")
    scRNA_data = scRNA_data[:, shared_genes]
    spRNA_data = spRNA_data[:, shared_genes]
    
    spatial = spRNA_data.obsm['spatial']
    sp_obs = spRNA_data.obs.copy()
    sc_obs = scRNA_data.obs.copy()
    
    spRNA_data.obs = spRNA_data.obs.drop(columns=spRNA_data.obs.columns)
    scRNA_data.obs = scRNA_data.obs[['cell_type']]
    
    adata_all = scRNA_data.concatenate(spRNA_data, index_unique=None)
    
    sc.pp.highly_variable_genes(adata_all, batch_key="batch", flavor="seurat_v3", n_top_genes=final_n_top_genes)
    
    if include_markers:
        print(f"Computing scRNA data marker genes ...")
        sc.tl.rank_genes_groups(scRNA_data, groupby='cell_type', method='wilcoxon')
        markers_df = pd.DataFrame(scRNA_data.uns["rank_genes_groups"]["names"]).iloc[0:30, :]
        markers = list(np.unique(markers_df.melt().value.values))
        markers = list(set(adata_all.var.loc[adata_all.var['highly_variable']==1].index)|set(markers)) # highly variable genes + cell type marker genes
    else:
        markers = adata_all.var.loc[adata_all.var['highly_variable']==1, ].index.tolist()
    print(f"We ultimately use {len(markers)} genes for training ...")
    
    adata_all = adata_all[:, markers].copy()
    
    if sp.issparse(adata_all.X):
        adata_all.X = adata_all.X.toarray()
        
    scRNA_data = adata_all[adata_all.obs['batch']=='0', ].copy()
    spRNA_data = adata_all[adata_all.obs['batch']=='1', ].copy()
    
    spRNA_data.obsm['spatial'] = spatial
    spRNA_data.obs = sp_obs.copy()
    
    scRNA_data.obs = sc_obs.copy()
    
    del adata_all
    
    return scRNA_data, spRNA_data

def is_integer_matrix(matrix):
    """
    Check if a matrix contains integer values (e.g., raw counts).
    Args:
        matrix: Matrix (dense or sparse).
    Returns:
        bool: True if the matrix contains integer values, False otherwise.
    """
    sample = matrix[:1000, :1000].todense() if sp.issparse(matrix) else matrix[:1000, :1000]
    return np.all(sample.astype(int) == sample)


# ------------------------- Final Integration Functions -------------------------
def preprocess_datasets(scRNA_data, spRNA_data, hvg_num=30000, min_counts=300, min_genes=600, final_n_top_genes=2000, min_genes_in_cells=3, datatype=None, include_markers=True):
    """
    Preprocess single-cell and spatial RNA-seq data and apply dimensionality reduction.

    Args:
        scRNA_data (AnnData): 
            Single-cell RNA-seq data in the form of an AnnData object to preprocess.
        spRNA_data (AnnData): 
            Spatial RNA-seq data in the form of an AnnData object to preprocess.
        hvg_num (int, optional): 
            Number of highly variable genes (HVGs) to select for each dataset during preprocessing. 
            Default is 30000.
        min_counts (int, optional): 
            Minimum number of total counts required for a cell to be kept in the dataset. 
            Default is 300.
        min_genes (int, optional): 
            Minimum number of genes expressed in a cell for it to be kept in the dataset. 
            Default is 600.
        final_n_top_genes (int, optional): 
            Number of highly variable genes (HVGs) to select on the final concatenated dataset (combined single-cell and spatial RNA-seq data). 
            Default is 2000.
        min_genes_in_cells (int, optional): 
            Minimum number of cells that must express a gene for it to be kept. Genes expressed in fewer cells will be filtered out. 
            Default is 3.
        datatype (str, required): 
            Specifies the type of spatial RNA-seq data. Must be one of the following:
            - `'FISH'`: Indicates spatial data is from FISH-based methods (e.g., MERFISH, smFISH).
            - `'Seq'`: Indicates spatial data is from sequencing-based methods (e.g., Slide-seq, Stero-seq).
        include_markers: also use scRNA data cell type marker genes for training.

    Returns:
        Tuple[AnnData, AnnData]: 
            Preprocessed single-cell RNA-seq data (`scRNA_data`) and spatial RNA-seq data (`spRNA_data`) as AnnData objects.

    Raises:
        AssertionError: 
            If `datatype` is not one of the allowed values (`'FISH'` or `'Seq'`).
        ValueError: 
            If an invalid `datatype` is provided.

    Notes:
        - This function preprocesses single-cell and spatial RNA-seq datasets separately using `preprocess_scRNA_data` and `preprocess_spRNA_data`.
        - The processed datasets are then concatenated based on shared genes, and the final dataset is reduced to `final_n_top_genes` highly variable genes.
        - Spatial data preprocessing can optionally include an enhancement step depending on the `datatype` value.
    """
    assert datatype in ['FISH', 'Seq']
    
    # Preprocess single-cell RNA-seq data
    scRNA_data = preprocess_scRNA_data(scRNA_data, min_genes_in_cells=min_genes_in_cells, min_counts=min_counts, min_genes=min_genes, hvg_num=hvg_num)
    
    # Preprocess spatial RNA-seq data based on datatype
    if datatype == 'FISH':
        spRNA_data = preprocess_spRNA_data(spRNA_data, min_genes_in_cells=3, hvg_num=hvg_num, Enhance=False)
    elif datatype == 'Seq':
        spRNA_data = preprocess_spRNA_data(spRNA_data, min_genes_in_cells=3, hvg_num=hvg_num, Enhance=True)
    else:
        raise ValueError(f"Invalid datatype. Expected 'FISH' or 'Seq'.")
    
    # Concatenate datasets
    print("Concatenating datasets...")
    scRNA_data, spRNA_data = concatenate_shared_genes(scRNA_data, spRNA_data, final_n_top_genes=final_n_top_genes, include_markers=include_markers)
    
    print("Preprocessing completed...")
    return scRNA_data, spRNA_data




