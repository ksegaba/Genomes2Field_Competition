#!/usr/bin/env python3
""" 
Generate feature tables of training and testing data run models on.

Returns (to Training_Data/0_Master_Data/):
    [1] trait+soil merged (before split):   1_Training_Trait+Soil_Data_2014_2021_cleaned_no_duplicates.csv
    [2] trait & soil merged training set:   1_Training_Trait+Soil_Data_2014_2021_cleaned_no_duplicates_TRAIN.csv
    [3] trait & soil merged testing set:   1_Training_Trait+Soil_Data_2014_2021_cleaned_no_duplicates_VAL.csv
    [4] cross-validation scheme file of training set (after split) for rrBLUP:   cv_scheme_TRAIN.csv
    [5] cross-validation groups file of training set (after split) for sklearn models:   groups_cv_TRAIN.csv
    [6] meta, weather (PCs), & ec (PCs) merged (before split):   2_Training_Meta+Weather+EC_Data_2014_2021.csv
    [7] trait, soil, meta, weather (PCs), & ec (PCs) merged (before split):   0_Training_Trait+Soil+Meta+Weather+EC_Data_2014_2021.csv
    [8] yield pheno.csv for rrBLUP:   pheno.csv

Other outputs (to Training_Data/):
    [9] sample IDs file:   All_hybrid_names_info_new_IDs
   [10] cleaned soil data with new IDs:   3_Training_Soil_Data_2015_2021_cleaned_new_IDs.csv
   [11] weather PCA data with Env fixed:   4_Weather_centered_PCA.csv
   [12] ec PCA data with Env fixed:   6_Training_EC_centered_PCA.csv
   [13] one-hot encoded meta data:   2_Training_Meta_Data_2014_2021_cleaned_one-hot_encoded.csv

Testing_Data outputs (to Testing_Data/):

"""
__author__ = "Kenia E. Segura Abá"

import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
os.chdir("/mnt/home/seguraab/Shiu_Lab/Collabs/Maize_GxE_Competition_Data/Training_Data")

if __name__ == "__main__":
    ## pheno for each year or env
    # df = pd.read_csv('1_Training_Trait_Data_2014_2021_cleaned.csv')
    # years = df.Year.unique()
    # for yr in years:
    #     sub = df[df.Year==yr]
    #     sub['ID'] = df['Env'].str.cat(df['Hybrid'], sep='__')
    #     print(sub.head())
    #     sub[['ID', 'Yield_Mg_ha']].to_csv(f'pheno_{yr}.csv', index=False)
    #
    # envs = df.Env.unique() # pheno for each env
    # for env in envs:
    #     sub = df[df.Env==env]
    #     sub.rename({'Hybrid':'ID'}, inplace=True)
    #     print(sub.head())
    #     sub[['ID', 'Yield_Mg_ha']].to_csv(f'pheno_{env}.csv', index=False)
    
    ############################################################################
    # Training_Data feature tables 
    ############################################################################

    # Read in data and do minimal preprocessing
    soil = pd.read_csv("3_Training_Soil_Data_2015_2021_cleaned.csv") # soil
    soil.drop("Year_y", axis=1, inplace=True)

    meta = pd.read_csv("2_Training_Meta_Data_2014_2021_cleaned.csv")
    meta.drop(["Year", "Experiment_Code", "Farm", "Comments"], axis=1, inplace=True)

    weather = pd.read_csv("4_Training_Weather_Data_2014_2021_newdates.csv")
    wpc = pd.read_csv("4_Weather_centered_PCA.csv") # weather (PCs)
    len(weather.Env.str.split('_').str[0].unique()) # check for 45 fields
    len(weather.Env.unique())

    ec = pd.read_csv("6_Training_EC_Data_2014_2021.csv")
    ecpc = pd.read_csv("6_Training_EC_centered_PCA.csv") # ec (PCs)
    len(ec.Env.str.split('_').str[0].unique()) # check for 45 fields
    len(ec.Env.unique())

    df = pd.read_csv('1_Training_Trait_Data_2014_2021_cleaned.csv')
    soil.insert(1, "Hybrid", df.Hybrid.values)

    # One-hot encode meta categorical variables
    trt = pd.get_dummies(meta.Treatment)
    crp = pd.get_dummies(meta.Previous_Crop)
    til = pd.get_dummies(meta["Pre-plant_tillage_method(s)"])
    meta.drop(["Treatment", "Previous_Crop", "Pre-plant_tillage_method(s)"], 
        axis=1, inplace=True)
    meta = pd.concat([meta, trt, crp, til], axis=1)
    meta.to_csv("2_Training_Meta_Data_2014_2021_cleaned_one-hot_encoded.csv")

    # Fix Env on wpc and ecpc
    wpc.dropna(inplace=True)
    wpc.shape[0]==len(weather.Env.unique()) # True
    wpc.Env = weather.Env.unique()
    wpc.to_csv("4_Weather_centered_PCA.csv", index=False)

    ecpc.dropna(inplace=True)
    ecpc.shape[0]==len(ec.Env.unique()) # True
    ecpc.Env = ec.Env.unique()
    ecpc.to_csv("6_Training_EC_centered_PCA.csv", index=False)

    # Add age feature
    df["Date_Harvested"] = df["Date_Harvested"].astype("datetime64")
    df["Date_Planted"] = df["Date_Planted"].astype("datetime64")
    df["Age"] = (df["Date_Harvested"] - df["Date_Planted"]).dt.days

    # Take the median of replicate hybrids in each Env
    ## trait data + soil data
    # df.Replicate.unique() # check how many replicates per plant
    df = df.sort_values(by=['Env', 'Hybrid'])
    soil = soil.sort_values(by=['Env', 'Hybrid'])
    df.set_index(['Env', 'Hybrid'], inplace=True)
    soil.set_index(['Env', 'Hybrid'], inplace=True)
    trait_soil = pd.concat([df, soil], axis=1, join="outer")
    reduced = trait_soil.groupby(['Env', 'Hybrid'])[['Plot_Area_ha', 'Pollen_DAP_days', 
        'Silk_DAP_days', 'Plant_Height_cm', 'Ear_Height_cm', 'Yield_Mg_ha', 
        'Grain_Moisture', 'Twt_kg_m3', 'Age', '1:1 Soil pH', 'WDRF Buffer pH', 
        'Organic Matter LOI %', 'Nitrate-N ppm N', 'lbs N/A', 'Sulfate-S ppm S', 
        'Calcium ppm Ca', 'Magnesium ppm Mg', 'Sodium ppm Na', 
        'CEC/Sum of Cations me/100g', '%H Sat', '%K Sat', '%Ca Sat', '%Mg Sat', 
        '%Na Sat', '% Sand', '% Silt', '% Clay']].median()
    reduced.reset_index(inplace=True)
    reduced.insert(0, "ID", reduced.Env.str.cat(reduced.Hybrid, "-")) # set new ID
    reduced.drop(["Env", "Hybrid"], axis=1, inplace=True)

    ### Generate the sample IDs file
    all = pd.read_csv('All_hybrid_names_info.csv')
    ID = pd.DataFrame(reduced.reset_index()[['Env', 'Hybrid']], columns=['Env', 'Hybrid'])
    ID["ID"] = ID.Env.str.cat(ID.Hybrid, "-")
    out = ID.merge(all, on="Hybrid", how="left")
    out.isna().sum # NAs are in Parent2, that's ok
    out.to_csv('All_hybrid_names_info_new_IDs.csv', index=False)

    ## Add IDs to soil
    soil.reset_index(inplace=True)
    soil.insert(0, "ID", soil.Env.str.cat(soil.Hybrid, "-").values)
    soil.drop(["Env", "Hybrid"], axis=1, inplace=True)
    soil.to_csv("3_Training_Soil_Data_2015_2021_cleaned_new_IDs.csv", index=False)

    ## Generate merged feature tables by field
    # To Do: 
    #   1. Generate the trait training and validation feature tables
    #   2. Merge trait to meta + weather (PCs) + ec (PCs) + soil (Huan merged these)

    # add yield class to generate stratified train and val tables for each field
    l = reduced.Yield_Mg_ha.quantile(0.25)
    r = reduced.Yield_Mg_ha.quantile(0.75)
    reduced["yield_class"] = "low"
    reduced.loc[reduced["Yield_Mg_ha"]>=r, "yield_class"] = "high"
    reduced.loc[(reduced["Yield_Mg_ha"]>l) & (reduced["Yield_Mg_ha"]<r), "yield_class"] = "mid"

    # trait + soil train-test split!
    train, test = train_test_split(reduced, test_size = 0.2, stratify=reduced.yield_class, random_state=42)
    len(train.ID.str.split("_").str[0].unique()) # check for 45 fields
    len(test.ID.str.split("_").str[0].unique()) # check for 45 fields

    # one-hot encode yield_class after split
    yield_class_train = pd.get_dummies(train["yield_class"]) 
    yield_class_test = pd.get_dummies(test["yield_class"])
    train = pd.concat([train, yield_class_train], axis=1, ignore_index=True)
    test = pd.concat([test, yield_class_test], axis=1, ignore_index=True)
    cols = reduced.columns.to_list()
    cols.append("yield_class_high")
    cols.append("yield_class_low")
    cols.append("yield_class_mid")
    train.columns = cols
    test.columns = cols
    train.drop("yield_class", axis=1, inplace=True)
    test.drop("yield_class", axis=1, inplace=True)
    train.to_csv("0_Master_Data/1_Training_Trait+Soil_Data_2014_2021_cleaned_no_duplicates_TRAIN.csv")
    test.to_csv("0_Master_Data/1_Training_Trait+Soil_Data_2014_2021_cleaned_no_duplicates_VAL.csv")

    # one-hot encode yield_class for trait+soil before split
    yield_class = pd.get_dummies(reduced["yield_class"])
    reduced = pd.concat([reduced, yield_class], axis=1)
    reduced.drop("yield_class", axis=1, inplace=True)
    reduced.to_csv("0_Master_Data/1_Training_Trait+Soil_Data_2014_2021_cleaned_no_duplicates.csv")

    # Generate 8-fold CV Scheme and Groups files
    cvs = train["ID"].to_frame() # cvs file for rrBLUP
    cvs.reset_index(inplace=True)
    cvs.drop("index", axis=1, inplace=True)
    for i in range(1,9): # add 8 columns (there are 8 years)
        cvs[f"Fold_{i}"] = 0
    years = [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021]
    for i in range(0,8): # fill in columns
        cvs.loc[cvs.ID.str.contains(f"{years[i]}", na=False), f"Fold_{i+1}"] = 1
    # check for duplicates, just in case
    cvs.sum(axis=1)[cvs.sum(axis=1)>1] # 3 IDs appeared two folds
    cvs.iloc[19145,3] = 0 # correct
    cvs.iloc[52698,3] = 0
    cvs.iloc[65818,3] = 0
    cvs.to_csv("0_Master_Data/cv_scheme_TRAIN.csv", index=False)

    groups = pd.melt(cvs, id_vars="ID") # groups for sklearn.cross_validate
    groups["group"] = ""
    for i in range(1,9):
        groups.loc[(groups.variable==f"Fold_{i}") & (groups.value==1), "group"] = i
    groups = groups[groups.group!=""]
    groups.drop(["variable", "value"], axis=1, inplace=True)
    groups.to_csv("0_Master_Data/groups_cv_TRAIN.csv", index=False)

    # Merge meta + weather (PCs) + ec (PCs)
    meta.set_index("Env", inplace=True)
    wpc.set_index("Env", inplace=True)
    ecpc.set_index("Env", inplace=True)
    mwe = pd.concat([meta, wpc, ecpc], axis=1, join="outer")
    mwe.isna().sum() # check missing values
    mwe = mwe.dropna()
    len(mwe.index.str.split("-").str[0].unique()) # 157 envs
    len(mwe.index.str.split("_").str[0].unique()) # 40 fields
    mwe.to_csv("0_Master_Data/2_Training_Meta+Weather+EC_Data_2014_2021.csv")

    # Merge trait + soil + meta + weather (PCs) + ec (PCs)
    reduced.insert(1, "Env", reduced.ID.str.split("-").str[0])
    mwe.reset_index(inplace=True)
    out = reduced.merge(mwe, how="inner", on="Env")
    out.drop("Env", axis=1, inplace=True)
    out.to_csv("0_Master_Data/0_Training_Trait+Soil+Meta+Weather+EC_Data_2014_2021.csv", index=False)

    # The remaining steps are done per field
    # fields=df.Field_Location.unique()
    # for f in fields:
    #     sub_f = reduced[reduced.ID.str.contains(f)] # only this field
    #     sub_no_f = reduced[~reduced.ID.str.contains(f)] # all other fields
    #     
    #     # Step 1b. generate train (all other fields not f) and val for each field f
    #     s_f_train, s_f_test = train_test_split(sub_f, test_size = 0.2, stratify=sub_f.yield_class, random_state=42)
    #     s_no_f_train, s_f_no_test = train_test_split(sub_no_f, test_size = 0.2, stratify=sub_no_f.yield_class, random_state=42)
    #     # need to fix lines below    
    #     # Step 2. check yield distributions of both train and val are stratified
    #     ax = s_train.hist(figsize=(30,30), align="mid")
    #     plt.savefig(f"0_Validation_Data/1_{f}_Trait+Soil_Data_2014_2021_TRAIN.png")
    #     plt.close()
    #     ax = s_test.hist(figsize=(30,30), align="mid")
    #     plt.savefig(f"0_Validation_Data/1_{f}_Trait+Soil_Data_2014_2021_VAL.png")
    #     plt.close()

    #     # Step 3. save merged trait + soil datasets
    #     s_train.to_csv(
    #         f"0_Validation_Data/1_{f}_Trait+Soil_Data_2014_2021_TRAIN.csv", index=False)
    #     s_test.to_csv(
    #         f"0_Validation_Data/1_{f}_Trait+Soil_Data_2014_2021_VAL.csv", index=False)
    
    ############################################################################
    #  Testing_Data feature tables
    ############################################################################
    # clear up large created variables
    del reduced
    del train
    del test
    del out
    del soil
    del df
    del cvs 

    import os
    import pandas as pd
    os.chdir("/mnt/home/seguraab/Shiu_Lab/Collabs/Maize_GxE_Competition_Data/Testing_Data")

    # Read in data
    info = pd.read_csv("../Training_Data/All_hybrid_names_info_new_IDs.csv")
    template = pd.read_csv("1_Submission_Template_2022.csv")
    meta = pd.read_csv("2_Testing_Meta_Data_2022.csv")
    soil = pd.read_csv("3_Testing_Soil_Data_2022.csv")
    weather = pd.read_csv("4_Testing_Weather_Data_2022.csv")
    ec = pd.read_csv("6_Testing_EC_Data_2022.csv")
    trait = pd.read_csv("../Training_Data/1_Training_Trait_Data_2014_2021_cleaned_no_duplicates.csv")
    
    # generate pheno.csv for rrBLUP training
    trait[["ID", "Yield_Mg_ha"]].to_csv("0_Master_Data/pheno.csv", index=False)

    # Do test set hybrids have trait data? No.
    info_test = info[info.test==True]
    info_test.shape # (4902, 8)
    len(info_test.Hybrid.unique()) # 43
    len(info_test.Env.unique()) # 216
    len(template.Env.unique()) # 26
    len(template.Hybrid.unique()) # 548
    IDs = template.Env.str.cat(template.Hybrid, "-").to_frame() # get Env-Hybrid IDs
    IDs.rename(columns={"Env":"ID"}, inplace=True)
    trait_test = IDs.merge(trait, how="inner", on="ID")
    trait_test.shape # (0, 11)

    # useful commands: len(), df.shape, df.head()
    # Take averages or medians across envs and hybrids for each column in the training set trait data
    # to get putative test set trait values
    # use trait dataset
    # consider Field and Hybrid (not year)
    trait["Field"] = trait.ID.str.split("_").str[0]
    trait["Hybrid"] = trait.ID.str.split("-").str[1]
    # first, subset trait by field and hybrid that are in template
    trait.shape
    template.shape
    template["Field"] = template.Env.str.split("_").str[0]
    sub = trait[(trait.Field.isin(template.Field)) & (trait.Hybrid.isin(template.Hybrid))]
    sub.shape
    # second, take medians across all the columns 
    out = sub.groupby(["Field", "Hybrid"])[['Plot_Area_ha', 'Pollen_DAP_days', 'Silk_DAP_days',
       'Plant_Height_cm', 'Ear_Height_cm', 'Yield_Mg_ha', 'Grain_Moisture',
       'Twt_kg_m3', 'Age']].median()
    out.shape

    # third, merge with meta, weather (PCA), soil, and ec (PCA) data

    # fourth, check that the order of columns in training data is the same as in testing data
    training = pd.read_csv('../Training_Data/0_Master_Data/0_Training_Trait+Soil+Meta+Weather+EC_Data_2014_2021.csv')
    training_cols = training.columns

    testing = ...
    testing_cols = testing.columns

    training_cols.eq(testing_cols)