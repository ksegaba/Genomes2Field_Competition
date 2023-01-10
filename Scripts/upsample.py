"""
Upsample the combined data using SMOTE
"""
__author__ = "Kenia E. Segura Abá"

import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

pd.set_option("display.max_rows", None)
os.chdir("/mnt/home/seguraab/Shiu_Lab/Collabs/Maize_GxE_Competition_Data/Training_Data")

if __name__ == "__main__":
    # Read in data
    df = pd.read_csv("1_Training_Trait_Data_2014_2021_cleaned.csv")
    all = pd.read_csv("Merged_Trait_Weather_Soil_Data_imputed.csv")
    
    # Add missing Yield values (forgot to do in datawig_imputer.py)
    all.isna().sum()
    predictions = pd.read_csv("yield_imputed.csv") # predicted yield
    train = all[~all["Yield_Mg_ha"].isnull()]
    test = all[all["Yield_Mg_ha"].isnull()] # missing values in yield
    test['Yield_Mg_ha'] = predictions['Yield_Mg_ha_imputed'].values
    all = pd.concat([train, test])
    all.isna().sum() # all good!
    del train, test

    # Classify yield into low, mid, high
    l = df.Yield_Mg_ha.quantile(0.25)
    r = df.Yield_Mg_ha.quantile(0.75)
    df["yield_class"] = "low"
    df.loc[df["Yield_Mg_ha"]>=r, "yield_class"] = "high"
    df.loc[(df["Yield_Mg_ha"]>l) & (df["Yield_Mg_ha"]<r), "yield_class"] = "mid"
    all['yield_class'] = "low"
    all.loc[all["Yield_Mg_ha"]>=r, "yield_class"] = "high"
    all.loc[(all["Yield_Mg_ha"]>l) & (all["Yield_Mg_ha"]<r), "yield_class"] = "mid"

    # Check counts of each group
    fig = all.pivot_table(index="yield_class", aggfunc="size").plot(kind="bar")
    fig.set_title("before upsampling")
    plt.savefig("yield_class.pdf")
    plt.close()
    all.groupby('yield_class').count()
    all[all["Yield_Mg_ha"]<=l].shape # (33967, 71)
    all[all["Yield_Mg_ha"]>=r].shape # (33968, 71)
    all[(all["Yield_Mg_ha"]>l) & (all["Yield_Mg_ha"]<r)].shape # (67932, 71)

    # Split data using stratified sampling
    train, test = train_test_split(all, test_size = 0.2, stratify=all.yield_class, random_state=42)
    fig = train.pivot_table(index="yield_class", aggfunc="size").plot(kind="bar")
    fig.set_title("verify class distribution in train is same as input data")
    plt.savefig("yield_class_train.pdf")
    plt.close()
    fig = test.pivot_table(index="yield_class", aggfunc="size").plot(kind='bar')
    fig.set_title("verify class distribution in test is same as input data")
    plt.savefig("yield_class_test.pdf")
    plt.close()

    # Fit a baseline model to get benchmark performance for SMOTE
    train_num = train.select_dtypes(include=["float64"])
    test_num = test.select_dtypes(include=["float64"])
    #mod = LogisticRegression()
    #mod.fit(X=train_num[train_num.columns], y=train["yield_class"]) # train the model
    #preds = mod.predict(test[test_num.columns]) # evaluate the model

    #