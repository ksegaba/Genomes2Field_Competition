#!/usr/bin/env python3
import os
import datatable as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datawig
from datawig import CategoricalEncoder, NumericalEncoder, NumericalFeaturizer, EmbeddingFeaturizer
from sklearn.metrics import r2_score
os.chdir("/mnt/home/seguraab/Shiu_Lab/Collabs/Maize_GxE_Competition_Data/Training_Data")
pd.set_option("display.max_rows", None)

def impute(
    data_encoder_cols, data_featurizer_cols, label_encoder_cols, out, col_to_imp, 
    train, test, save, test_accuracy=False):
    
    """
    Datawig Imputer to impute missing values in the column of interest (label). 
    Parameters:
        data_encoder_cols (list): column encoder list
        data_featurizer_cols (list): column featurizer list
        label_encoder_cols (list):  label encoder list (length of 1)
        out (str): directory name to save imputer in
        col_to_imp (str): column to impute (label)
        train (pandas dataframe): training set
        test (pandas dataframe): testing set
        save (str): file name to save imputed dataframe as
        test_accuracy (bool): whether to run a test or not (True/False)
    Returns:
        Nothing. This function saves the imputer and the imputed dataframe
        to a directory/file.
    """
    
    imputer = datawig.Imputer(
        data_encoders=data_encoder_cols,
        data_featurizers=data_featurizer_cols,
        label_encoders=label_encoder_cols,
        output_path=out
    )
    imputer.fit(train_df=train, test_df=test) # fit the imputer model
    predictions = imputer.predict(test) # predict the label
    imputer.save() # save model
    if test_accuracy==True:
        print("R2:", r2_score(test[col_to_imp], predictions[f"{col_to_imp}_imputed"].round()))
    else:
        # update missing values in label
        test[col_to_imp] = predictions[f"{col_to_imp}_imputed"].round()
        imputed = pd.concat([train, test])
        imputed.to_csv(save) # save to file


if __name__ == "__main__":
    # Read in manually imputed data
    df1_imputed = pd.read_csv("1_Training_Trait_Data_2014_2021_imputed.csv")
    df1_imputed.isna().sum()

    # Read in weather data
    weather = pd.read_csv("4_Training_Weather_Data_2014_2021.csv") 

    # Standardize dates
    weather["Date"] = pd.to_datetime(weather["Date"], format="%Y%m%d").dt.strftime("%Y-%m-%d")
    weather.to_csv("4_Training_Weather_Data_2014_2021_newdates.csv", index=False)

    # Merge df1_imputed with weather data (filter by date first & date harvested)
    weather.set_index("Env", inplace=True)
    df1_imputed.set_index("Env", inplace=True)
    df1_imputed_weather = pd.merge(df1_imputed, weather, how="left", left_on=["Env", "Date_Harvested"], right_on=["Env", "Date"])
    df1_imputed_weather.reset_index(inplace=True)
    df1_imputed_weather.isna().sum()

    # Read in soil data
    soil = pd.read_csv("3_Training_Soil_Data_2015_2021.csv")

    # Standardize dates
    soil["Date Received"] = pd.to_datetime(soil["Date Received"]).dt.strftime("%Y-%m-%d")
    soil["Date Reported"] = pd.to_datetime(soil["Date Reported"]).dt.strftime("%Y-%m-%d")
    soil.to_csv("3_Training_Soil_Data_2015_2021_newdates.csv", index=False)

    # Merge df1_imputed_weather with soil data (filter by date first & date harvested)
    soil.set_index("Env", inplace=True)
    df1_imputed_weather.set_index("Env", inplace=True)
    df1_imp_wea_soil = pd.merge(df1_imputed_weather, soil, how="left", left_on="Env", right_on="Env")
    df1_imp_wea_soil.reset_index(inplace=True)
    df1_imp_wea_soil.isna().sum()
    df1_imp_wea_soil.drop("index", axis=1, inplace=True) # drop index column
    df1_imp_wea_soil.rename(columns={"Year_x": "Year"}, inplace=True) # rename Year_x

    # Save merged dataframe
    df1_imp_wea_soil.to_csv("Merged_Trait_Weather_Soil_Data.csv", index=False)

    ############################################################################
    # Impute Silk_DAP_days
    ############################################################################
    data_encoder_cols = [NumericalEncoder("Year"), 
        CategoricalEncoder("Field_Location"), 
        CategoricalEncoder("Hybrid"), 
        NumericalEncoder("Pollen_DAP_days"), 
        NumericalEncoder("Grain_Moisture"), 
        NumericalEncoder("Boron ppm B"), 
        NumericalEncoder("1:1 Soil pH")] # columns related to the label column to impute
    label_encoder_cols = [NumericalEncoder("Silk_DAP_days")] # column to impute
    data_featurizer_cols = [NumericalFeaturizer("Year"), 
        EmbeddingFeaturizer("Field_Location"), 
        EmbeddingFeaturizer("Hybrid"), 
        NumericalFeaturizer("Pollen_DAP_days"), 
        NumericalFeaturizer("Grain_Moisture"), 
        NumericalFeaturizer("Boron ppm B"), 
        NumericalFeaturizer("1:1 Soil pH")]
    
    # Check accuracy of imputer
    print("Checking accuracy of imputer for Silk_DAP_days...")
    sub = df1_imp_wea_soil[~df1_imp_wea_soil["Silk_DAP_days"].isnull()] # training set (no missing data in label)
    train, test = datawig.utils.random_split(sub, split_ratios=[0.9, 0.1])
    impute(data_encoder_cols, data_featurizer_cols, label_encoder_cols, 
        out="TEST_SilkDAPdays_imputer_model", col_to_imp="Silk_DAP_days", 
        train=train, test=test, save="Merged_Trait_Weather_Soil_Data_imputed.csv", 
        test_accuracy=True)
    
    # Back to imputing Silk_DAP_days
    print("Imputing Silk_DAP_days...")
    train = df1_imp_wea_soil[~df1_imp_wea_soil["Silk_DAP_days"].isnull()] # training set (no missing data in label)
    test = df1_imp_wea_soil[df1_imp_wea_soil["Silk_DAP_days"].isnull()] # testing set
    # impute(data_encoder_cols, data_featurizer_cols, label_encoder_cols, 
    #     out="df1_imp_wea_soil_SilkDAPdays_imputer_model", col_to_imp="Silk_DAP_days", 
    #     train=train, test=test, save="Merged_Trait_Weather_Soil_Data_imputed.csv")
    imputer = datawig.Imputer(
        data_encoders=data_encoder_cols,
        data_featurizers=data_featurizer_cols,
        label_encoders=label_encoder_cols,
        output_path="df1_imp_wea_soil_SilkDAPdays_imputer_model"
    )
    imputer.fit(train_df=train)
    predictions = imputer.predict(test)
    imputer.save() # save model
    test["Silk_DAP_days"] = predictions["Silk_DAP_days_imputed"].round()
    df1_iws_imputed = pd.concat([train, test])
    df1_iws_imputed.to_csv("Merged_Trait_Weather_Soil_Data_imputed.csv", index=False) # save df1_imp_wea_soil with imputed column

    # Delete datasets to clear up memory
    del df1_imp_wea_soil
    del df1_imputed
    del weather
    del soil
    
    ############################################################################
    # Impute Pollen_DAP_days
    ############################################################################
    data_encoder_cols = [NumericalEncoder("Year"), 
        CategoricalEncoder("Field_Location"), 
        CategoricalEncoder("Hybrid"), 
        NumericalEncoder("Silk_DAP_days"),
        NumericalEncoder("Grain_Moisture"), 
        NumericalEncoder("Boron ppm B"), 
        NumericalEncoder("1:1 Soil pH")] # columns related to the label column to impute
    label_encoder_cols = [NumericalEncoder("Pollen_DAP_days")] # column to impute
    data_featurizer_cols = [NumericalFeaturizer("Year"), 
        EmbeddingFeaturizer("Field_Location"), 
        EmbeddingFeaturizer("Hybrid"), 
        NumericalFeaturizer("Silk_DAP_days"), 
        NumericalFeaturizer("Grain_Moisture"), 
        NumericalFeaturizer("Boron ppm B"), 
        NumericalFeaturizer("1:1 Soil pH")]
    
    # Check accuracy of imputer
    print("Checking accuracy of imputer for Pollen_DAP_days...")
    df1_imp_wea_soil = pd.read_csv("Merged_Trait_Weather_Soil_Data_imputed.csv")
    sub = df1_imp_wea_soil[~df1_imp_wea_soil["Pollen_DAP_days"].isnull()] # training set (no missing data in label)
    train, test = datawig.utils.random_split(sub, split_ratios=[0.9, 0.1])
    impute(data_encoder_cols, data_featurizer_cols, label_encoder_cols, 
        out="TEST_PollenDAPdays_imputer_model", col_to_imp="Pollen_DAP_days", 
        train=train, test=test, save="Merged_Trait_Weather_Soil_Data_imputed.csv", 
        test_accuracy=True)
    
    # Back to imputing Pollen_DAP_days
    print("Imputing Pollen_DAP_days...")
    train = df1_imp_wea_soil[~df1_imp_wea_soil["Pollen_DAP_days"].isnull()] # training set (no missing data in label)
    test = df1_imp_wea_soil[df1_imp_wea_soil["Pollen_DAP_days"].isnull()] # testing set
    # impute(data_encoder_cols, data_featurizer_cols, label_encoder_cols, 
    #     out="df1_imp_wea_soil_PollenDAPdays_imputer_model", col_to_imp="Pollen_DAP_days", 
    #     train=train, test=test, save="Merged_Trait_Weather_Soil_Data_imputed.csv")
    imputer = datawig.Imputer(
        data_encoders=data_encoder_cols,
        data_featurizers=data_featurizer_cols,
        label_encoders=label_encoder_cols,
        output_path="df1_imp_wea_soil_PollenDAPdays_imputer_model"
    )
    imputer.fit(train_df=train)
    predictions = imputer.predict(test)
    imputer.save() # save model
    test["Pollen_DAP_days"] = predictions["Pollen_DAP_days_imputed"] # replace missing values with predictions
    df1_iws_imputed = pd.concat([train, test])
    df1_iws_imputed.to_csv("Merged_Trait_Weather_Soil_Data_imputed.csv", index=False) # save df1_imp_wea_soil with imputed column

    ###########################################################################
    # Impute Plant_Height_cm
    ############################################################################
    data_encoder_cols = [NumericalEncoder("Year"), 
        CategoricalEncoder("Field_Location"), 
        CategoricalEncoder("Hybrid"), 
        NumericalEncoder("Ear_Height_cm"), 
        NumericalEncoder("Yield_Mg_ha"), 
        NumericalEncoder("Grain_Moisture"), 
        NumericalEncoder("Organic Matter LOI %"), 
        NumericalEncoder("Nitrate-N ppm N"), 
        NumericalEncoder("% Silt")] # columns related to the label column to impute
    label_encoder_cols = [NumericalEncoder("Plant_Height_cm")] # column to impute
    data_featurizer_cols = [NumericalFeaturizer("Year"), 
        EmbeddingFeaturizer("Field_Location"), 
        EmbeddingFeaturizer("Hybrid"), 
        NumericalFeaturizer("Ear_Height_cm"), 
        NumericalFeaturizer("Yield_Mg_ha"), 
        NumericalFeaturizer("Grain_Moisture"), 
        NumericalFeaturizer("Organic Matter LOI %"), 
        NumericalFeaturizer("Nitrate-N ppm N"), 
        NumericalFeaturizer("% Silt")]

    # Check accuracy of imputer
    print("Checking accuracy of imputer for Plant_Height_cm...")
    df1_imp_wea_soil = pd.read_csv("Merged_Trait_Weather_Soil_Data_imputed.csv")
    sub = df1_imp_wea_soil[~df1_imp_wea_soil["Plant_Height_cm"].isnull()] # training set (no missing data in label)
    train, test = datawig.utils.random_split(sub, split_ratios=[0.9, 0.1])
    impute(data_encoder_cols, data_featurizer_cols, label_encoder_cols, 
        out="TEST_PlantHeightcm_imputer_model", col_to_imp="Plant_Height_cm", 
        train=train, test=test, save="Merged_Trait_Weather_Soil_Data_imputed.csv", 
        test_accuracy=True)
    
    # Back to imputing Plant_Height_cm
    print("Imputing Plant_Height_cm...")
    train = df1_imp_wea_soil[~df1_imp_wea_soil["Plant_Height_cm"].isnull()] # training set (no missing data in label)
    test = df1_imp_wea_soil[df1_imp_wea_soil["Plant_Height_cm"].isnull()] # testing set
    # impute(data_encoder_cols, data_featurizer_cols, label_encoder_cols, 
    #     out="df1_imp_wea_soil_PlantHeightcm_imputer_model", col_to_imp="Plant_Height_cm", 
    #     train=train, test=test, save="Merged_Trait_Weather_Soil_Data_imputed.csv")
    imputer = datawig.Imputer(
        data_encoders=data_encoder_cols,
        data_featurizers=data_featurizer_cols,
        label_encoders=label_encoder_cols,
        output_path="df1_imp_wea_soil_PlantHeightcm_imputer_model"
    )
    imputer.fit(train_df=train)
    predictions = imputer.predict(test)
    imputer.save() # save model
    test["Plant_Height_cm"] = predictions["Plant_Height_cm_imputed"] # replace missing values with predictions
    df1_iws_imputed = pd.concat([train, test])
    df1_iws_imputed.to_csv("Merged_Trait_Weather_Soil_Data_imputed.csv", index=False) # save df1_imp_wea_soil with imputed column
    
    ############################################################################
    # Impute Ear_Height_cm
    ############################################################################
    data_encoder_cols = [NumericalEncoder("Year"), 
        CategoricalEncoder("Field_Location"), 
        CategoricalEncoder("Hybrid"), 
        NumericalEncoder("Plant_Height_cm"), 
        NumericalEncoder("Yield_Mg_ha"), 
        NumericalEncoder("Grain_Moisture"), 
        NumericalEncoder("Organic Matter LOI %"), 
        NumericalEncoder("Nitrate-N ppm N"), 
        NumericalEncoder("RH2M"), 
        NumericalEncoder("%Mg Sat"), 
        NumericalEncoder("% Silt"), 
        NumericalEncoder("Zinc ppm Zn"), 
        NumericalEncoder("Iron ppm Fe"), 
        NumericalEncoder("Manganese ppm Mn"), 
        NumericalEncoder("Copper ppm Cu")] # columns related to the label column to impute
    label_encoder_cols = [NumericalEncoder("Ear_Height_cm")] # column to impute
    data_featurizer_cols = [NumericalFeaturizer("Year"), 
        EmbeddingFeaturizer("Field_Location"), 
        EmbeddingFeaturizer("Hybrid"), 
        NumericalFeaturizer("Plant_Height_cm"), 
        NumericalFeaturizer("Yield_Mg_ha"), 
        NumericalFeaturizer("Grain_Moisture"), 
        NumericalFeaturizer("Organic Matter LOI %"), 
        NumericalFeaturizer("Nitrate-N ppm N"), 
        NumericalFeaturizer("RH2M"), 
        NumericalFeaturizer("%Mg Sat"), 
        NumericalFeaturizer("% Silt"), 
        NumericalFeaturizer("Zinc ppm Zn"), 
        NumericalFeaturizer("Iron ppm Fe"), 
        NumericalFeaturizer("Manganese ppm Mn"), 
        NumericalFeaturizer("Copper ppm Cu")]
    
    # Check accuracy of imputer
    print("Checking accuracy of imputer for Ear_Height_cm...")
    df1_imp_wea_soil = pd.read_csv("Merged_Trait_Weather_Soil_Data_imputed.csv")
    sub = df1_imp_wea_soil[~df1_imp_wea_soil["Ear_Height_cm"].isnull()] # training set (no missing data in label)
    train, test = datawig.utils.random_split(sub, split_ratios=[0.9, 0.1])
    impute(data_encoder_cols, data_featurizer_cols, label_encoder_cols, 
        out="TEST_EarHeightcm_imputer_model", col_to_imp="Ear_Height_cm", 
        train=train, test=test, save="Merged_Trait_Weather_Soil_Data_imputed.csv", 
        test_accuracy=True)
    
    # Back to imputing Ear_Height_cm
    print("Imputing Ear_Height_cm...")
    train = df1_imp_wea_soil[~df1_imp_wea_soil["Ear_Height_cm"].isnull()] # training set (no missing data in label)
    test = df1_imp_wea_soil[df1_imp_wea_soil["Ear_Height_cm"].isnull()] # testing set
    # impute(data_encoder_cols, data_featurizer_cols, label_encoder_cols, 
    #     out="df1_imp_wea_soil_EarHeightcm_imputer_model", col_to_imp="Ear_Height_cm", 
    #     train=train, test=test, save="Merged_Trait_Weather_Soil_Data_imputed.csv")
    imputer = datawig.Imputer(
        data_encoders=data_encoder_cols,
        data_featurizers=data_featurizer_cols,
        label_encoders=label_encoder_cols,
        output_path="df1_imp_wea_soil_EarHeightcm_imputer_model"
    )
    imputer.fit(train_df=train, patience=10)
    predictions, metrics = imputer.transform_and_compute_metrics(test)
    imputer.save() # save model
    test["Ear_Height_cm"] = predictions["Ear_Height_cm"] # replace missing values with predictions
    df1_iws_imputed = pd.concat([train, test])
    df1_iws_imputed.to_csv("Merged_Trait_Weather_Soil_Data_imputed.csv", index=False)

    ############################################################################
    # Impute Root_Lodging_plants
    ############################################################################
    data_encoder_cols = [NumericalEncoder("Year"),
        CategoricalEncoder("Field_Location"), 
        CategoricalEncoder("Hybrid"), 
        NumericalEncoder("Plant_Height_cm"), 
        NumericalEncoder("1:1 S Salts mmho/cm"),
        NumericalEncoder("Magnesium ppm Mg"),
        NumericalEncoder("%Mg Sat")] # columns related to the label column to impute
    label_encoder_cols = [NumericalEncoder("Root_Lodging_plants")] # column to impute
    data_featurizer_cols = [NumericalFeaturizer("Year"),
        EmbeddingFeaturizer("Field_Location"), 
        EmbeddingFeaturizer("Hybrid"), 
        NumericalFeaturizer("Plant_Height_cm"), 
        NumericalFeaturizer("1:1 S Salts mmho/cm"), 
        NumericalFeaturizer("Magnesium ppm Mg"),
        NumericalFeaturizer("%Mg Sat")]

    # Check accuracy of imputer
    print("Checking accuracy of imputer for Root_Lodging_plants...")
    df1_imp_wea_soil = pd.read_csv("Merged_Trait_Weather_Soil_Data_imputed.csv")
    sub = df1_imp_wea_soil[~df1_imp_wea_soil["Root_Lodging_plants"].isnull()] # training set (no missing data in label)
    train, test = datawig.utils.random_split(sub, split_ratios=[0.9, 0.1])
    impute(data_encoder_cols, data_featurizer_cols, label_encoder_cols, 
        out="TEST_RootLodgingplants_imputer_model", col_to_imp="Root_Lodging_plants", 
        train=train, test=test, save="Merged_Trait_Weather_Soil_Data_imputed.csv", 
        test_accuracy=True)
    
    ############################################################################
    # Impute Stalk_Lodging_plants
    ############################################################################
    data_encoder_cols = [NumericalEncoder("Year"),
        CategoricalEncoder("Field_Location"), 
        CategoricalEncoder("Hybrid"), 
        NumericalEncoder("Stand_Count_plants"), 
        NumericalEncoder("Ear_Height_cm"), 
        NumericalEncoder("Zinc ppm Zn"),
        NumericalEncoder("Iron ppm Fe"),
        NumericalEncoder("Copper ppm Cu"),
        NumericalEncoder("Boron ppm B")] # columns related to the label column to impute
    label_encoder_cols = [NumericalEncoder("Stalk_Lodging_plants")] # column to impute
    data_featurizer_cols = [NumericalFeaturizer("Year"),
        EmbeddingFeaturizer("Field_Location"), 
        EmbeddingFeaturizer("Hybrid"), 
        NumericalFeaturizer("Stand_Count_plants"), 
        NumericalFeaturizer("Ear_Height_cm"), 
        NumericalFeaturizer("Zinc ppm Zn"),
        NumericalFeaturizer("Iron ppm Fe"),
        NumericalFeaturizer("Copper ppm Cu"),
        NumericalFeaturizer("Boron ppm B")]

    # Check accuracy of imputer
    print("Checking accuracy of imputer for Stalk_Lodging_plants...")
    df1_imp_wea_soil = pd.read_csv("Merged_Trait_Weather_Soil_Data_imputed.csv")
    sub = df1_imp_wea_soil[~df1_imp_wea_soil["Stalk_Lodging_plants"].isnull()] # training set (no missing data in label)
    train, test = datawig.utils.random_split(sub, split_ratios=[0.9, 0.1])
    impute(data_encoder_cols, data_featurizer_cols, label_encoder_cols, 
        out="TEST_StalkLodgingplants_imputer_model", col_to_imp="Stalk_Lodging_plants",
        train=train, test=test, save="Merged_Trait_Weather_Soil_Data_imputed.csv",
        test_accuracy=True)
    
    ############################################################################
    # Impute Grain_Moisture
    ############################################################################
    data_encoder_cols = [NumericalEncoder("Year"),
        CategoricalEncoder("Field_Location"), 
        CategoricalEncoder("Hybrid"), 
        NumericalEncoder("Plant_Height_cm"), 
        NumericalEncoder("Ear_Height_cm"), 
        NumericalEncoder("Yield_Mg_ha"), 
        NumericalEncoder("RH2M"), 
        NumericalEncoder("GWETTOP"), 
        NumericalEncoder("GWETROOT"), 
        NumericalEncoder("GWETPROF"), 
        NumericalEncoder("Organic Matter LOI %"), 
        NumericalEncoder("Magnesium ppm Mg"), 
        NumericalEncoder("%Mg Sat"), 
        NumericalEncoder("Mehlich P-III ppm P"), 
        NumericalEncoder("Zinc ppm Zn"), 
        NumericalEncoder("Iron ppm Fe"), 
        NumericalEncoder("Manganese ppm Mn"), 
        NumericalEncoder("Copper ppm Cu")] # columns related to the label column to impute
    label_encoder_cols = [NumericalEncoder("Grain_Moisture")] # column to impute
    data_featurizer_cols = [NumericalFeaturizer("Year"),
        EmbeddingFeaturizer("Field_Location"), 
        EmbeddingFeaturizer("Hybrid"), 
        NumericalFeaturizer("Plant_Height_cm"), 
        NumericalFeaturizer("Ear_Height_cm"), 
        NumericalFeaturizer("Yield_Mg_ha"), 
        NumericalFeaturizer("RH2M"), 
        NumericalFeaturizer("GWETTOP"), 
        NumericalFeaturizer("GWETROOT"), 
        NumericalFeaturizer("GWETPROF"), 
        NumericalFeaturizer("Organic Matter LOI %"), 
        NumericalFeaturizer("Magnesium ppm Mg"), 
        NumericalFeaturizer("%Mg Sat"), 
        NumericalFeaturizer("Mehlich P-III ppm P"), 
        NumericalFeaturizer("Zinc ppm Zn"), 
        NumericalFeaturizer("Iron ppm Fe"), 
        NumericalFeaturizer("Manganese ppm Mn"), 
        NumericalFeaturizer("Copper ppm Cu")]
        
    # Check accuracy of imputer
    print("Checking accuracy of imputer for Grain_Moisture...")
    df1_imp_wea_soil = pd.read_csv("Merged_Trait_Weather_Soil_Data_imputed.csv")
    sub = df1_imp_wea_soil[~df1_imp_wea_soil["Grain_Moisture"].isnull()] # training set (no missing data in label)
    train, test = datawig.utils.random_split(sub, split_ratios=[0.9, 0.1])
    impute(data_encoder_cols, data_featurizer_cols, label_encoder_cols, 
        out="TEST_GrainMoisture_imputer_model", col_to_imp="Grain_Moisture", 
        train=train, test=test, save="Merged_Trait_Weather_Soil_Data_imputed.csv", 
        test_accuracy=True)
    
    # Back to imputing Grain_Moisture
    print("Imputing Grain_Moisture...")
    train = df1_imp_wea_soil[~df1_imp_wea_soil["Grain_Moisture"].isnull()] # training set (no missing data in label)
    test = df1_imp_wea_soil[df1_imp_wea_soil["Grain_Moisture"].isnull()] # testing set
    # impute(data_encoder_cols, data_featurizer_cols, label_encoder_cols, 
    #     out="df1_imp_wea_soil_GrainMoisture_imputer_model", col_to_imp="Grain_Moisture",
    #     train=train, test=test, save="Merged_Trait_Weather_Soil_Data_imputed.csv")
    imputer = datawig.Imputer(
        data_encoders=data_encoder_cols,
        data_featurizers=data_featurizer_cols,
        label_encoders=label_encoder_cols,
        output_path="df1_imp_wea_soil_GrainMoisture_imputer_model"
    )
    imputer.fit(train_df=train)
    predictions = imputer.predict(test)
    imputer.save() # save model
    test["Grain_Moisture"] = predictions["Grain_Moisture_imputed"].round()
    df1_iws_imputed = pd.concat([train, test])
    df1_iws_imputed.to_csv("Merged_Trait_Weather_Soil_Data_imputed.csv", index=False) # save df1_imp_wea_soil with imputed column

    ############################################################################
    # Impute Twt_kg_m3
    ############################################################################
    data_encoder_cols = [NumericalEncoder("Year"), 
        CategoricalEncoder("Field_Location"), 
        CategoricalEncoder("Hybrid"), 
        NumericalEncoder("% Silt"), 
        NumericalEncoder("%Ca Sa"), 
        NumericalEncoder("Calcium ppm Ca"), 
        NumericalEncoder("T2M"), 
        NumericalEncoder("T2M_MAX"), 
        NumericalEncoder("T2M_MIN"), 
        NumericalEncoder("T2MWET"), 
        NumericalEncoder("ALLSKY_SFC_SW_DWN"), 
        NumericalEncoder("ALLSKY_SFC_PAR_TOT")] # columns related to the label column to impute
    label_encoder_cols = [NumericalEncoder("Twt_kg_m3")] # column to impute
    data_featurizer_cols = [NumericalFeaturizer("Year"), 
        EmbeddingFeaturizer("Field_Location"), 
        EmbeddingFeaturizer("Hybrid"), 
        NumericalFeaturizer("% Silt"), 
        NumericalFeaturizer("%Ca Sa"), 
        NumericalFeaturizer("Calcium ppm Ca"), 
        NumericalFeaturizer("T2M"), 
        NumericalFeaturizer("T2M_MAX"), 
        NumericalFeaturizer("T2M_MIN"), 
        NumericalFeaturizer("T2MWET"), 
        NumericalFeaturizer("ALLSKY_SFC_SW_DWN"), 
        NumericalFeaturizer("ALLSKY_SFC_PAR_TOT")]
        
    # Check accuracy of imputer
    print("Checking accuracy of imputer for Twt_kg_m3...")
    df1_imp_wea_soil = pd.read_csv("Merged_Trait_Weather_Soil_Data_imputed.csv")
    sub = df1_imp_wea_soil[~df1_imp_wea_soil["Twt_kg_m3"].isnull()] # training set (no missing data in label)
    train, test = datawig.utils.random_split(sub, split_ratios=[0.9, 0.1])
    impute(data_encoder_cols, data_featurizer_cols, label_encoder_cols, 
        out="TEST_Twtkgm3_imputer_model", col_to_imp="Twt_kg_m3", 
        train=train, test=test, save="Merged_Trait_Weather_Soil_Data_imputed.csv", 
        test_accuracy=True)
    
    # Back to imputing Twt_kg_m3
    print("Imputing Twt_kg_m3...")
    train = df1_imp_wea_soil[~df1_imp_wea_soil["Twt_kg_m3"].isnull()] # training set (no missing data in label)
    test = df1_imp_wea_soil[df1_imp_wea_soil["Twt_kg_m3"].isnull()] # testing set
    # impute(data_encoder_cols, data_featurizer_cols, label_encoder_cols, 
    #     out="df1_imp_wea_soil_Twtkgm3_imputer_model", col_to_imp="Twt_kg_m3",
    #     train=train, test=test, save="Merged_Trait_Weather_Soil_Data_imputed.csv")
    imputer = datawig.Imputer(
        data_encoders=data_encoder_cols,
        data_featurizers=data_featurizer_cols,
        label_encoders=label_encoder_cols,
        output_path="df1_imp_wea_soil_Twtkgm3_imputer_model"
    )
    imputer.fit(train_df=train)
    predictions = imputer.predict(test)
    imputer.save() # save model
    test["Twt_kg_m3"] = predictions["Twt_kg_m3_imputed"].round()
    df1_iws_imputed = pd.concat([train, test])
    df1_iws_imputed.to_csv("Merged_Trait_Weather_Soil_Data_imputed.csv", index=False) # save df1_imp_wea_soil with imputed column

    ############################################################################
    # Impute Yield_Mg_ha
    ############################################################################
    data_encoder_cols = [NumericalEncoder("Year"), 
        CategoricalEncoder("Field_Location"), 
        CategoricalEncoder("Hybrid"), 
        NumericalEncoder("Stand_Count_plants"), 
        NumericalEncoder("Plant_Height_cm"), 
        NumericalEncoder("Ear_Height_cm"), 
        NumericalEncoder("Grain_Moisture"), 
        NumericalEncoder("GWETTOP"),
        NumericalEncoder("GWETROOT"),
        NumericalEncoder("GWETPROF"),
        NumericalEncoder("Year_y"),
        NumericalEncoder("Organic Matter LOI %"),
        NumericalEncoder("Magnesium ppm Mg"),
        NumericalEncoder("%Mg Sat"),
        NumericalEncoder("Boron ppm B")] # columns related to the label column to impute
    label_encoder_cols = [NumericalEncoder("Yield_Mg_ha")] # column to impute
    data_featurizer_cols = [NumericalFeaturizer("Year"), 
        EmbeddingFeaturizer("Field_Location"), 
        EmbeddingFeaturizer("Hybrid"), 
        NumericalFeaturizer("Stand_Count_plants"), 
        NumericalFeaturizer("Plant_Height_cm"), 
        NumericalFeaturizer("Ear_Height_cm"), 
        NumericalFeaturizer("Grain_Moisture"), 
        NumericalFeaturizer("GWETPROF"), 
        NumericalFeaturizer("Year_y"), 
        NumericalFeaturizer("Organic Matter LOI %"),
        NumericalFeaturizer("Magnesium ppm Mg"), 
        NumericalFeaturizer("%Mg Sat"), 
        NumericalFeaturizer("Boron ppm B")]
    
    # Check accuracy of imputer
    print("Checking accuracy of imputer for Yield_Mg_ha...")
    df1_imp_wea_soil = pd.read_csv("Merged_Trait_Weather_Soil_Data_imputed.csv")
    sub = df1_imp_wea_soil[~df1_imp_wea_soil["Yield_Mg_ha"].isnull()] # training set (no missing data in label)
    train, test = datawig.utils.random_split(sub, split_ratios=[0.9, 0.1])
    impute(data_encoder_cols, data_featurizer_cols, label_encoder_cols, 
        out="TEST_YieldMgha_imputer_model", col_to_imp="Yield_Mg_ha", 
        train=train, test=test, save="Merged_Trait_Weather_Soil_Data_imputed.csv", 
        test_accuracy=True)
    
    # Back to imputing Yield_Mg_ha
    print("Imputing Yield_Mg_ha...")
    train = df1_imp_wea_soil[~df1_imp_wea_soil["Yield_Mg_ha"].isnull()] # training set (no missing data in label)
    test = df1_imp_wea_soil[df1_imp_wea_soil["Yield_Mg_ha"].isnull()] # testing set
    # impute(data_encoder_cols, data_featurizer_cols, label_encoder_cols, 
    #     out="df1_imp_wea_soil_YieldMgha_imputer_model", col_to_imp="Yield_Mg_ha",
    #     train=train, test=test, save="Merged_Trait_Weather_Soil_Data_imputed.csv")
    imputer = datawig.Imputer(
        data_encoders=data_encoder_cols,
        data_featurizers=data_featurizer_cols,
        label_encoders=label_encoder_cols,
        output_path="df1_imp_wea_soil_YieldMgha_imputer_model"
    )
    imputer.fit(train_df=train)
    predictions = imputer.predict(test)
    imputer.save() # save model
    test["Yield_Mg_ha"] = predictions["Yield_Mg_ha_imputed"].round()
    df1_iws_imputed = pd.concat([train, test])
    df1_iws_imputed.to_csv("Merged_Trait_Weather_Soil_Data_imputed.csv", index=False) # save df1_imp_wea_soil with imputed column

    # Drop columns with low imputation accuracy
    df1_imp_wea_soil.drop(["Root_Lodging_plants", "Stalk_Lodging_plants"], axis=1)
    df1_imp_wea_soil.to_csv("Merged_Trait_Weather_Soil_Data_imputed.csv") # save df1_imp_wea_soil with imputed column

    # Check missing data in final dataset
    df1_imp_wea_soil = pd.read_csv("Merged_Trait_Weather_Soil_Data_imputed.csv", index=False)
    print(df1_imp_wea_soil.iloc[:,0:20].isna().sum())
    