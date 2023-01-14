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
    #weather.rename(columns={"Env":"ID"}, inplace=True)
    #weather.drop("Date", axis=1, inplace=True)
    #weather.to_csv("4_Training_Weather_Data_2014_2021_cleaned.csv", index=False)
    #os.system("python ../Scripts/dimension_reduction.py -path /mnt/home/seguraab/Shiu_Lab/Collabs/Maize_GxE_Competition_Data/Training_Data -file 4_Training_Weather_Data_2014_2021_cleaned.csv -save 4_Training_Weather_Data_2014_2021 -alg pca")
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
    
    # generate pheno.csv for rrBLUP training (forgot to do this before)
    os.chdir("/mnt/ufs18/home-056/seguraab/Shiu_Lab/Collabs/Maize_GxE_Competition_Data/Testing_Data/")
    trait = pd.read_csv("../Training_Data/1_Training_Trait_Data_2014_2021_cleaned_no_duplicates.csv")
    sub = trait[["ID", "Yield_Mg_ha"]]
    l = sub.Yield_Mg_ha.quantile(0.25)
    r = sub.Yield_Mg_ha.quantile(0.75)
    sub["yield_class"] = "low"
    sub.loc[sub["Yield_Mg_ha"]>=r, "yield_class"] = "high"
    sub.loc[(sub["Yield_Mg_ha"]>l) & (sub["Yield_Mg_ha"]<r), "yield_class"] = "mid"
    sub.to_csv("0_Master_Data/pheno.csv", index=False)
    # now split by env
    envs = sub.ID.str.split("-").str[0].unique() # 212/217 envs
    all_envs = pd.read_csv('../Training_Data/env_list.txt', header=None) # 217 envs
    all_envs[~all_envs[0].isin(envs)] # the 6 envs that were lost. idk know how
    # TXH1-Dry_2017
    # TXH1-Early_2017
    # TXH1-Late_2017
    # TXH1-Dry_2018
    # TXH1-Early_2018
    # TXH1-Late_2018
    for env in envs:
        env_sub = sub.loc[sub.ID.str.contains(env),["ID", "Yield_Mg_ha"]]
        env_sub.insert(1, "Hybrid", env_sub.ID.str.split("-").str[1])
        # remove duplicates
        env_sub.drop("ID", axis=1, inplace=True)
        env_sub = env_sub.groupby("Hybrid")["Yield_Mg_ha"].median()
        env_sub = env_sub.to_frame()
        env_sub.reset_index(inplace=True)
        env_sub.to_csv(f"../Training_Data/0_Master_Data/pheno_{env}.csv", index=False)
    # make sure that all samples in 0_MASTER_Training_tsmwe_Data_2014_2021.csv are in pheno.csv
    training = pd.read_csv('../Training_Data/0_Master_Data/0_MASTER_Training_tsmwe_Data_2014_2021.csv')
    pheno = pd.read_csv('../Training_Data/0_Master_Data/pheno.csv')
    training.shape #(62115, 60)
    pheno.shape #(87356, 3)
    training[~training.ID.isin(pheno.ID)].shape # (0, 60) yay!

    import os
    import pandas as pd
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    os.chdir("/mnt/home/seguraab/Shiu_Lab/Collabs/Maize_GxE_Competition_Data/Testing_Data")

    # Read in data
    info = pd.read_csv("../Training_Data/All_hybrid_names_info_new_IDs.csv")
    template = pd.read_csv("1_Submission_Template_2022.csv")
    meta = pd.read_csv("2_Testing_Meta_Data_2022.csv")
    soil = pd.read_csv("3_Testing_Soil_Data_2022.csv")
    weather = pd.read_csv("4_Testing_Weather_Data_2022.csv")
    ec = pd.read_csv("6_Testing_EC_Data_2022.csv")
    
    # remove columns with missing data from weather and meta
    weather.isna().sum()
    weather.drop(["GWETTOP", "GWETROOT", "GWETPROF", 
        "ALLSKY_SFC_PAR_TOT", "ALLSKY_SFC_SW_DNI"], axis=1, inplace=True)
    # check the 130 missing values in each columns
    weather[weather.isna().any(axis=1)].shape # 283 rows total
    lost = weather[weather.isna().any(axis=1)].Env.unique() # Envs that would be lost
    keep = weather[~weather.isna().any(axis=1)].Env.unique()
    len(lost)==len(keep) # some days of each env will be lost
    weather.dropna(inplace=True)
    len(weather.Env.unique()) # no entire Env was lost. Hurray!

    meta.drop(["Year", "Experiment_Code", "Farm", "Comments"], axis=1, inplace=True)
    trt = pd.get_dummies(meta.Treatment) # one-hot encode meta columns
    crp = pd.get_dummies(meta.Previous_Crop)
    til = pd.get_dummies(meta["Pre-plant_tillage_method(s)"])
    meta.drop(["Treatment", "Previous_Crop", "Pre-plant_tillage_method(s)"], 
        axis=1, inplace=True)
    meta = pd.concat([meta, trt, crp, til], axis=1)
    meta.to_csv("2_Testing_Meta_Data_2022_cleaned_one-hot_encoded.csv")

    # run pca of weather and ec
    weather.rename(columns={"Env":"ID"}, inplace=True)
    #weather.drop(["Date"], axis=1, inplace=True)
    weather.to_csv("4_Testing_Weather_Data_2022_cleaned.csv", index=False) # Huan further processed this later
    os.system("python ../Scripts/dimension_reduction.py -path /mnt/home/seguraab/Shiu_Lab/Collabs/Maize_GxE_Competition_Data/Testing_Data -file 4_Testing_Weather_Data_2022_T_coulumNoNA.csv -save 4_Testing_Weather_Data_2022_T_coulumNoNA -alg pca")
    
    ec.rename(columns={"Env":"ID"}, inplace=True)
    ec.to_csv("6_Testing_EC_Data_2022_cleaned.csv", index=False)
    os.system("python ../Scripts/dimension_reduction.py -path /mnt/home/seguraab/Shiu_Lab/Collabs/Maize_GxE_Competition_Data/Testing_Data -file 6_Testing_EC_Data_2022_cleaned.csv -save 6_Testing_EC_Data_2022 -alg pca")

    # Merge meta, weather (PCs), and ec (PCs)
    # wpc = pd.read_csv("4_Testing_Weather_Data_2022_centered_PCA.csv") # this is just a place holder
    # wpc = wpc.groupby("ID")[['0', '1', '2', '3', '4', '5']].median() # this is just a place holder until huan runs here pipeline to get pca of weather dates as features
    # wpc.reset_index(inplace=True) # place holder
    wpc = pd.read_csv("4_Testing_Weather_Data_2022_T_coulumNoNA_centered_PCA.csv") # this is the correct dataset
    ecpc = pd.read_csv("6_Testing_EC_Data_2022_centered_PCA.csv")
    wpc.columns = ["Env", "WPCA0", "WPCA1", "WPCA2", "WPCA3", "WPCA4", "WPCA5", 'WPCA6', 'WPCA7', 'WPCA8', 'WPCA9', 'WPCA10', 'WPCA11', 'WPCA12', 'WPCA13', 'WPCA14', 'WPCA15', 'WPCA16']
    ecpc.columns = ["Env", "ECPCA0", "ECPCA1", "ECPCA2", "ECPCA3", "ECPCA4", "ECPCA5", "ECPCA6", "ECPCA7"]
    
    meta = pd.read_csv("2_Testing_Meta_Data_2022_cleaned_one-hot_encoded.csv")
    meta.drop("Unnamed: 0", axis=1, inplace=True)
    meta.set_index("Env", inplace=True)
    wpc.set_index("Env", inplace=True)
    ecpc.set_index("Env", inplace=True)
    mwe = pd.concat([meta, wpc, ecpc], axis=1, join="outer")
    mwe.isna().sum() # check missing values
    mwe.drop(["City", "Field", "Trial_ID (Assigned by collaborator for internal reference)",
        "Soil_Taxonomic_ID and horizon description, if known", 
        "In-season_tillage_method(s)", "System_Determining_Moisture",
        "Pounds_Needed_Soil_Moisture", "Latitude_of_Field_Corner_#2 (lower right)",
        "Cardinal_Heading_Pass_1", "Plot_Area_ha",
        "Weather_Station_Serial_Number (Last four digits, e.g. m2700s#####)", 
        "Date_weather_station_placed", "Date_weather_station_removed", 
        "Issue/comment_#1", "Issue/comment_#2", "Issue/comment_#3",
        "Issue/comment_#4", "Issue/comment_#5", "Issue/comment_#6", 
        "Date_Planted"], axis=1, inplace=True)
    mwe.isna().sum()
    mwe = mwe.dropna()
    len(mwe.index.str.split("-").str[0].unique()) # 17 envs
    len(mwe.index.str.split("_").str[0].unique()) # 17 fields
    # mwe.to_csv("2_Testing_Meta+Weather+EC_Data_2022_PLACEHOLDER.csv") # change index to ID manually in vim
    mwe.to_csv("2_Testing_Meta+Weather+EC_Data_2022.csv") # change index to ID manually in vim
    
    # Merge template, soil, mwe
    template.insert(0, "ID", template.Env.str.cat(template.Hybrid, "-"))
    soil["Date Received"] = pd.to_datetime(soil["Date Received"]).dt.strftime("%Y-%m-%d")
    soil["Date Reported"] = pd.to_datetime(soil["Date Reported"]).dt.strftime("%Y-%m-%d")
    soil.to_csv("3_Testing_Soil_Data_2022_newdates.csv", index=False)
    ts = pd.merge(template[["ID", "Env"]], soil, how="left", on="Env")
    ts.isna().sum()
    mwe.reset_index(inplace=True)
    mwe.rename(columns={"index":"Env"}, inplace=True)
    out = pd.merge(ts, mwe, how="left", on="Env") # cannot do inner bc we need to keep all the hybrids!
    out.isna().sum() # there are a lot of NAs
    ts.isna().sum() # no Env are missing, but WPCA and ECPCA have missing data in out
    mwe.isna().sum() # no Env are missing
    len(weather.Env.unique()) # 26
    len(template.Env.unique()) # 26
    len(ts.Env.unique()) # 26
    len(meta.index.unique()) # 26
    len(wpc.index.unique()) # 26
    len(ec.Env.unique()) # 24 the problem starts here!
    len(mwe.Env.unique()) #  17 this is the problem! cannot drop nas from mwe!
    # find which Env are missing in ec
    wpc[~wpc.index.isin(ec.Env)] # IAH1_2022 & NCH1_2022
    # Since these 2 are missing, we will have at least 2 Test Sets
    # one will be the hybrids in out
    # the rest will include the hybrids that were lost when merging soil with mwe
    # thus, we will have to retrain models based on the common features of this second Test set
    # to get the yield of the remaining hybrids.
    # So, instead of "left", let's do "inner"
    out = pd.merge(ts, mwe, how="inner", on="Env")
    out.shape # 7180 hybrids in the first Test set
    out.isna().sum() # cannot drop all rows with na bc it returns 0 rows. Instead I will impute this dataset
    out.drop(['LabID', 'Date Received', 'Date Reported', 'Texture No','Texture', 'Comments'], 
        axis=1, inplace=True) # drop columns irrelevant to imputation
    out.to_csv("0a_Testing_Trait+Soil+Meta+Weather+EC_2022.csv", index=False) #save it!!!!
    # Impute columns of out!!!!!!!
    # get spearman's rho heatmap to determine columns
    import matplotlib.pyplot as plt
    import seaborn as sns
    s = out.iloc[:,:25]
    s.to_csv("../0a_Testing_Trait+Soil+Meta+Weather+EC_2022_toImpute.csv", index=False)
    scor = s.select_dtypes("float64").corr()
    fig, ax = plt.subplots(figsize=(20,20))
    sns.heatmap(scor, ax=ax, annot=True)
    plt.savefig("../0a_Testing_Trait+Soil+Meta+Weather+EC_2022_toImpute_corr.pdf")
    plt.close()
    # submit imputation
    os.system("sbatch submit.sb")
    
    out.to_csv("0_Master_Data/0_Training_Trait+Soil+Meta+Weather+EC_Data_2014_2021.csv", index=False)
    # Keep only columns common to training and testing data
    # 1. generate testing set: done! see above
    test = pd.read_csv("0a_Testing_Trait+Soil+Meta+Weather+EC_2022.csv")
    train = pd.read_csv("../Training_Data/0_Master_Data/0_Training_Trait+Soil+Meta+Weather+EC_Data_2014_2021.csv")
    test.shape
    train.shape
    # 2. generate new training and testing sets:
    new_train = train.loc[:,train.columns.isin(test.columns)]
    new_train.shape
    new_test = test.loc[:,test.columns.isin(new_train.columns)]
    new_test.shape
    new_train.to_csv("../Training_Data/0_Master_Data/0_MASTER_Training_tsmwe_Data_2014_2021.csv", index=False)
    new_test.to_csv("0_Master_Data/0a_MASTER_Testing_tsmwe_Data_2022.csv", index=False)
    # 3. split new_train into training and validation sets
    test = pd.read_csv('../Training_Data/0_Master_Data/test.txt', header=None)
    new_train_train = new_train[~new_train.ID.isin(test[0])]
    new_train_val = new_train[new_train.ID.isin(test[0])]
    new_train_train.shape
    new_train_val.shape
    new_train_train.to_csv("../Training_Data/0_Master_Data/0_MASTER_Training_tsmwe_Data_2014_2021_TRAIN.csv", index=False)
    new_train_val.to_csv("../Training_Data/0_Master_Data/0_MASTER_Training_tsmwe_Data_2014_2021_VAL.csv", index=False)
    
    # Make test set for remaining hybrids that were left out
    # Keep only columns common to training and testing data
    # 1. generate new training st
    # 2. generate testing set
