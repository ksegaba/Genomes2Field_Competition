#!/usr/bin/env python3
"""
Combine all datasets (trait, soil, weather, meta, env. cov.) and perform
feature selection based on correlations between features and yield in bins,
which are based on the year, field location, and hybrid line.
"""
__author__ = "Kenia E. Segura Abá"

import pandas as pd
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

if __name__ == "__main__":
    # Read in imputed datasets
    trait = pd.read_csv("1_Training_Trait_Data_2014_2021_cleaned.csv")
    meta = pd.read_csv("2_Training_Meta_Data_2014_2021_cleaned.csv")
    soil = pd.read_csv("3_Training_Soil_Data_2015_2021.csv")
    weather = pd.read_csv("4_Training_Weather_Data_2014_2021_newdates.csv")
    ec = pd.read_csv("6_Training_EC_Data_2014_2021.csv")

    # Combine meta and trait
    trait.set_index(["Year", "Env"], inplace=True)
    meta.set_index(["Year", "Env"], inplace=True)
    all = meta.join(trait, how="outer")

    # Combine all and soil
    soil.set_index(["Year", "Env"], inplace=True)
    all = all.join(soil, how="outer", rsuffix="_soil")

    # Combine all and weather
    weather.set_index("Env", inplace=True)
    all = all.join(weather, on="Env", how="outer", rsuffix="_weather")

    # Combine all and ec
    # Note: the ec data has a lot of dates, we may only need a few of them so will need to be filtered
    # or if there is some way to generate features from the dates too, else, we can keep them all.
    # What do you think?
    ec.set_index("Env", inplace=True)
    all = all.join(ec, on="Env", how="outer", rsuffix="_ec")

    # Generate submatrices for each Env
    for grp in all["Env"].unique():
        sub = all[all["Env"]==grp]

        # Calculate correlation matrix of each matrix
        # Note: These heatmaps will help us see which variables are not correlated
        # with yield. We can discuss which variables we want to drop based on this.
        sub_num = sub.select_dtypes(include=["float64"])
        sub_num_cor = sub_num.corr(method="spearman")
        fig, ax = plt.subplots(figsize=(40,40))
        sns.heatmap(sub_num_cor, ax=ax, annot=True)
        plt.savefig(f"1-6_{grp}_spearman_heatmap.pdf")