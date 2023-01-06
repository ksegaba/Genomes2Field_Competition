#!/usr/bin/env python3
"""
Generate subpopulations file to use with the autoencoder model
"""
__author__ = "Kenia Segura Abá"

import pandas as pd

if __name__=="__main__":
    # Read in maize hybrid names
    df = pd.read_csv("../Training_Data/All_hybrid_names_info.csv")

    # Subset training hybrids that have genotype data
    train = df[(df.train==True) & (df.vcf==True)]

    # Save to file
    train.Parent1.to_csv("../Training_Data/Hybrid_subpopulations", index=False)

    # Add subpopulation to PLINK .fam files' "Family ID (FID)" column
    beagle=pd.read_csv("../Training_Data/5_Genotype_Data_All_Years_maf005_maxmiss090_cleaned_BEAGLE_imputed.plink.fam", sep="\t", header=None)
    beagle[0] = beagle[1].str.split("/").str[0]
    beagle.to_csv("../Training_Data/5_Genotype_Data_All_Years_maf005_maxmiss090_cleaned_BEAGLE_imputed.plink.fam", sep="\t", index=False, header=False)
    
    linkimp=pd.read_csv("../Training_Data/5_Genotype_Data_All_Years_maf005_maxmiss090_cleaned_LinkImpute_imputed.plink.fam", sep="\t", header=None)
    linkimp[0] = linkimp[1].str.split("/").str[0]
    linkimp.to_csv("../Training_Data/5_Genotype_Data_All_Years_maf005_maxmiss090_cleaned_LinkImpute_imputed.plink.fam", sep="\t", index=False, header=False)