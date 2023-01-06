#!/usr/bin/env python3
"""
Run dimensionality reduction algorithms on a PLINK SNP data matrix with
0, 1, 2 genotype encodings (i.e. the number of minor alleles). Samples as rows
and SNPs as columns.

Arguments:
    path (str): Path to working directory
    file (str): Name of PLINK .raw file in working directory
    save (str): Name of file to save output matrix as
    alg (str): Algorithm to run (pca/svd)

Returns:
    A data matrix with reduced dimension for each method
    (PCA and SVD).
"""

import sys,os
import argparse
import datatable as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import itertools as IT
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.linalg import svd


def uniquify(path, sep = '_'):
    """
    Function to generate unique file names.
    Source: https://stackoverflow.com/questions/13852700/create-file-but-if-name-exists-add-number
    """
    def name_sequence():
        count = IT.count()
        yield ''
        while True:
            yield '{s}{n:d}'.format(s = sep, n = next(count))
    orig = tempfile._name_sequence 
    with tempfile._once_lock:
        tempfile._name_sequence = name_sequence()
        path = os.path.normpath(path)
        dirname, basename = os.path.split(path)
        filename, ext = os.path.splitext(basename)
        fd, filename = tempfile.mkstemp(dir = dirname, prefix = filename, suffix = ext)
        tempfile._name_sequence = orig
    return filename


def run_pca(dfs):
    """ 
    Principal Component Analysis
    Input:
        dfs (numpy array): centered or scaled numpy array
    Returns:
        [1] A matrix of n samples and k components.
        [2] A explained variance vs number of components curve.
    """
    print("Computing PCs...")
    pca = PCA(n_components = 0.99)
    pca.fit(dfs)
    out = pca.transform(dfs)
    print("No. components:", pca.components_.shape)
    print("Explained Var:", pca.explained_variance_ratio_)

    # Determine number of components
    plt.rcParams["figure.figsize"] = (12,6)
    fig, ax = plt.subplots()
    x = np.arange(1, pca.components_.shape[0]+1, step=1) # PCs
    y = np.cumsum(pca.explained_variance_ratio_) # cumulative explained variance
    plt.ylim(0.0,1.1)
    plt.plot(x, y, marker='o', linestyle='--', color='b')
    plt.xlabel('Number of Components')
    plt.xticks(np.arange(0, pca.components_.shape[0], step=1)) #change from 0-based array index to 1-based human-readable label
    plt.ylabel('Cumulative variance (%)')
    plt.title('The number of components needed to explain variance')
    plt.axhline(y=0.99, color='r', linestyle='-')
    plt.axhline(y=0.95, color='g', linestyle='-')
    plt.text(0.5, 0.1, '99% cut-off threshold', color = 'red', fontsize=12)
    plt.text(0.5, 0.2, '95% cut-off threshold', color = 'green', fontsize=12)
    ax.grid(axis='x')
    plt.savefig(uniquify('num_comp_curve.pdf'))

    return out


def run_svd(dfs):
    """ Full SVD """
    U, S, V = svd(dfs)
    out = np.dot(U, np.diag(S))
    return out, U, S, V


if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(
        description="Run dimension reduction techniques on genotype data")
    req_group = parser.add_argument_group(title="Required Input")
    req_group.add_argument(
        "-path", type=str, help="Path to working directory")
    req_group.add_argument(
        "-file", type=str, help="Name of PLINK .raw file in working directory")
    req_group.add_argument(
        "-save", type=str, help="Name of file to save output matrix as")
    req_group.add_argument(
        "-alg", type=str, help="Algorithm to run (pca/svd)")
    if len(sys.argv)==1:
        parser.print_help()
        sys.exist(0)
    args = parser.parse_args()

    # Set working directory
    os.chdir(args.path)

    # Read in genotype data
    df = dt.fread(args.file)
    df = df.to_pandas()

    # Drop extra columns and convert to numpy array
    df = df.drop(["FID", "PAT", "MAT", "SEX", "PHENOTYPE"], axis=1)
    IID = df.IID # save sample IDs
    df.set_index("IID", inplace=True) # set sample IDs as index
    SNPs = df.columns # save SNP IDs
    df_n = df.to_numpy() # convert to numpy array

    # Center genotypes
    scaler = StandardScaler(with_mean=True, with_std=False)
    dfc = scaler.fit_transform(df_n) # centered only
    
    ## Run PCA
    if args.alg=="pca":
        outc = run_pca(dfc)
        outc.to_csv(f"{args.save}_centered_PCA.csv", index=False)
    
    
    ## Run Full SVD
    if args.alg=="svd":
        out, U, S, V = run_svd(dfc)


