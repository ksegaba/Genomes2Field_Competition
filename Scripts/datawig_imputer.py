#!/usr/bin/env python3
import os
import datatable as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
        print("R2:", r2_score(test[col_to_imp], predictions[f"{col_to_imp}_imputed"]))
    else:
        # update missing values in label
        test[col_to_imp] = predictions[f"{col_to_imp}_imputed"]
        imputed = pd.concat([train, test])
        imputed.to_csv(save) # save to file


if __name__ == "__main__":
    # # Read in manually imputed data
    # df1_imputed = pd.read_csv("1_Training_Trait_Data_2014_2021_imputed.csv")
    # df1_imputed.isna().sum()

    # # Read in weather data
    # weather = pd.read_csv("4_Training_Weather_Data_2014_2021.csv") 

    # # Standardize dates
    # weather["Date"] = pd.to_datetime(weather["Date"], format="%Y%m%d").dt.strftime("%Y-%m-%d")
    # weather.to_csv("4_Training_Weather_Data_2014_2021_newdates.csv", index=False)

    # # Merge df1_imputed with weather data (filter by date first & date harvested)
    # weather.set_index("Env", inplace=True)
    # df1_imputed.set_index("Env", inplace=True)
    # df1_imputed_weather = pd.merge(df1_imputed, weather, how="left", left_on=["Env", "Date_Harvested"], right_on=["Env", "Date"])
    # df1_imputed_weather.reset_index(inplace=True)
    # df1_imputed_weather.isna().sum()

    # # Read in soil data
    # soil = pd.read_csv("3_Training_Soil_Data_2015_2021.csv")

    # # Standardize dates
    # soil["Date Received"] = pd.to_datetime(soil["Date Received"]).dt.strftime("%Y-%m-%d")
    # soil["Date Reported"] = pd.to_datetime(soil["Date Reported"]).dt.strftime("%Y-%m-%d")
    # soil.to_csv("3_Training_Soil_Data_2015_2021_newdates.csv", index=False)

    # # Merge df1_imputed_weather with soil data (filter by date first & date harvested)
    # soil.set_index("Env", inplace=True)
    # df1_imputed_weather.set_index("Env", inplace=True)
    # df1_imp_wea_soil = pd.merge(df1_imputed_weather, soil, how="left", left_on="Env", right_on="Env")
    # df1_imp_wea_soil.reset_index(inplace=True)
    # df1_imp_wea_soil.isna().sum()
    # df1_imp_wea_soil.drop("index", axis=1, inplace=True) # drop index column
    # df1_imp_wea_soil.rename(columns={"Year_x": "Year"}, inplace=True) # rename Year_x

    # # Save merged dataframe
    # df1_imp_wea_soil.to_csv("Merged_Trait_Weather_Soil_Data.csv", index=False)

    ############################################################################
    # Impute Silk_DAP_days
    ############################################################################
    # data_encoder_cols = [NumericalEncoder("Year"), 
    #     CategoricalEncoder("Field_Location"), 
    #     CategoricalEncoder("Hybrid"), 
    #     NumericalEncoder("Pollen_DAP_days"), 
    #     NumericalEncoder("Grain_Moisture"), 
    #     NumericalEncoder("Boron ppm B"), 
    #     NumericalEncoder("1:1 Soil pH")] # columns related to the label column to impute
    # label_encoder_cols = [NumericalEncoder("Silk_DAP_days")] # column to impute
    # data_featurizer_cols = [NumericalFeaturizer("Year"), 
    #     EmbeddingFeaturizer("Field_Location"), 
    #     EmbeddingFeaturizer("Hybrid"), 
    #     NumericalFeaturizer("Pollen_DAP_days"), 
    #     NumericalFeaturizer("Grain_Moisture"), 
    #     NumericalFeaturizer("Boron ppm B"), 
    #     NumericalFeaturizer("1:1 Soil pH")]
    
    # # Check accuracy of imputer
    # # print("Checking accuracy of imputer for Silk_DAP_days...")
    # # sub = df1_imp_wea_soil[~df1_imp_wea_soil["Silk_DAP_days"].isnull()] # training set (no missing data in label)
    # # train, test = datawig.utils.random_split(sub, split_ratios=[0.9, 0.1])
    # # impute(data_encoder_cols, data_featurizer_cols, label_encoder_cols, 
    # #     out="TEST_SilkDAPdays_imputer_model", col_to_imp="Silk_DAP_days", 
    # #     train=train, test=test, save="Merged_Trait_Weather_Soil_Data_imputed.csv", 
    # #     test_accuracy=True)
    
    # # Back to imputing Silk_DAP_days
    # print("Imputing Silk_DAP_days...")
    # train = df1_imp_wea_soil[~df1_imp_wea_soil["Silk_DAP_days"].isnull()] # training set (no missing data in label)
    # test = df1_imp_wea_soil[df1_imp_wea_soil["Silk_DAP_days"].isnull()] # testing set
    # # impute(data_encoder_cols, data_featurizer_cols, label_encoder_cols, 
    # #     out="df1_imp_wea_soil_SilkDAPdays_imputer_model", col_to_imp="Silk_DAP_days", 
    # #     train=train, test=test, save="Merged_Trait_Weather_Soil_Data_imputed.csv")
    # imputer = datawig.Imputer(
    #     data_encoders=data_encoder_cols,
    #     data_featurizers=data_featurizer_cols,
    #     label_encoders=label_encoder_cols,
    #     output_path="df1_imp_wea_soil_SilkDAPdays_imputer_model"
    # )
    # imputer.fit(train_df=train)
    # predictions = imputer.predict(test)
    # imputer.save() # save model
    # test["Silk_DAP_days"] = predictions["Silk_DAP_days_imputed"].round()
    # df1_iws_imputed = pd.concat([train, test])
    # df1_iws_imputed.to_csv("Merged_Trait_Weather_Soil_Data_imputed.csv", index=False) # save df1_imp_wea_soil with imputed column

    # # Delete datasets to clear up memory
    # del df1_imp_wea_soil
    # del df1_imputed
    # del weather
    # del soil
    
    # ############################################################################
    # # Impute Pollen_DAP_days
    # ############################################################################
    # data_encoder_cols = [NumericalEncoder("Year"), 
    #     CategoricalEncoder("Field_Location"), 
    #     CategoricalEncoder("Hybrid"), 
    #     NumericalEncoder("Silk_DAP_days"),
    #     NumericalEncoder("Grain_Moisture"), 
    #     NumericalEncoder("Boron ppm B"), 
    #     NumericalEncoder("1:1 Soil pH")] # columns related to the label column to impute
    # label_encoder_cols = [NumericalEncoder("Pollen_DAP_days")] # column to impute
    # data_featurizer_cols = [NumericalFeaturizer("Year"), 
    #     EmbeddingFeaturizer("Field_Location"), 
    #     EmbeddingFeaturizer("Hybrid"), 
    #     NumericalFeaturizer("Silk_DAP_days"), 
    #     NumericalFeaturizer("Grain_Moisture"), 
    #     NumericalFeaturizer("Boron ppm B"), 
    #     NumericalFeaturizer("1:1 Soil pH")]
    
    # # Check accuracy of imputer
    # # print("Checking accuracy of imputer for Pollen_DAP_days...")
    # df1_imp_wea_soil = pd.read_csv("Merged_Trait_Weather_Soil_Data_imputed.csv")
    # # sub = df1_imp_wea_soil[~df1_imp_wea_soil["Pollen_DAP_days"].isnull()] # training set (no missing data in label)
    # # train, test = datawig.utils.random_split(sub, split_ratios=[0.9, 0.1])
    # # impute(data_encoder_cols, data_featurizer_cols, label_encoder_cols, 
    # #     out="TEST_PollenDAPdays_imputer_model", col_to_imp="Pollen_DAP_days", 
    # #     train=train, test=test, save="Merged_Trait_Weather_Soil_Data_imputed.csv", 
    # #     test_accuracy=True)
    
    # # Back to imputing Pollen_DAP_days
    # print("Imputing Pollen_DAP_days...")
    # train = df1_imp_wea_soil[~df1_imp_wea_soil["Pollen_DAP_days"].isnull()] # training set (no missing data in label)
    # test = df1_imp_wea_soil[df1_imp_wea_soil["Pollen_DAP_days"].isnull()] # testing set
    # # impute(data_encoder_cols, data_featurizer_cols, label_encoder_cols, 
    # #     out="df1_imp_wea_soil_PollenDAPdays_imputer_model", col_to_imp="Pollen_DAP_days", 
    # #     train=train, test=test, save="Merged_Trait_Weather_Soil_Data_imputed.csv")
    # imputer = datawig.Imputer(
    #     data_encoders=data_encoder_cols,
    #     data_featurizers=data_featurizer_cols,
    #     label_encoders=label_encoder_cols,
    #     output_path="df1_imp_wea_soil_PollenDAPdays_imputer_model"
    # )
    # imputer.fit(train_df=train)
    # predictions = imputer.predict(test)
    # imputer.save() # save model
    # test["Pollen_DAP_days"] = predictions["Pollen_DAP_days_imputed"].round() # replace missing values with predictions
    # df1_iws_imputed = pd.concat([train, test])
    # df1_iws_imputed.to_csv("Merged_Trait_Weather_Soil_Data_imputed.csv", index=False) # save df1_imp_wea_soil with imputed column

    # # Delete datasets to clear up memory
    # del df1_imp_wea_soil
    # del train
    # del test
    # del df1_iws_imputed

    # ###########################################################################
    # # Impute Plant_Height_cm
    # ############################################################################
    # data_encoder_cols = [NumericalEncoder("Year"), 
    #     CategoricalEncoder("Field_Location"), 
    #     CategoricalEncoder("Hybrid"), 
    #     NumericalEncoder("Ear_Height_cm"), 
    #     NumericalEncoder("Yield_Mg_ha"), 
    #     NumericalEncoder("Grain_Moisture"), 
    #     NumericalEncoder("Organic Matter LOI %"), 
    #     NumericalEncoder("Nitrate-N ppm N"), 
    #     NumericalEncoder("% Silt")] # columns related to the label column to impute
    # label_encoder_cols = [NumericalEncoder("Plant_Height_cm")] # column to impute
    # data_featurizer_cols = [NumericalFeaturizer("Year"), 
    #     EmbeddingFeaturizer("Field_Location"), 
    #     EmbeddingFeaturizer("Hybrid"), 
    #     NumericalFeaturizer("Ear_Height_cm"), 
    #     NumericalFeaturizer("Yield_Mg_ha"), 
    #     NumericalFeaturizer("Grain_Moisture"), 
    #     NumericalFeaturizer("Organic Matter LOI %"), 
    #     NumericalFeaturizer("Nitrate-N ppm N"), 
    #     NumericalFeaturizer("% Silt")]

    # # Check accuracy of imputer
    # # print("Checking accuracy of imputer for Plant_Height_cm...")
    # df1_imp_wea_soil = pd.read_csv("Merged_Trait_Weather_Soil_Data_imputed.csv")
    # # sub = df1_imp_wea_soil[~df1_imp_wea_soil["Plant_Height_cm"].isnull()] # training set (no missing data in label)
    # # train, test = datawig.utils.random_split(sub, split_ratios=[0.9, 0.1])
    # # impute(data_encoder_cols, data_featurizer_cols, label_encoder_cols, 
    # #     out="TEST_PlantHeightcm_imputer_model", col_to_imp="Plant_Height_cm", 
    # #     train=train, test=test, save="Merged_Trait_Weather_Soil_Data_imputed.csv", 
    # #     test_accuracy=True)
    
    # # Back to imputing Plant_Height_cm
    # print("Imputing Plant_Height_cm...")
    # train = df1_imp_wea_soil[~df1_imp_wea_soil["Plant_Height_cm"].isnull()] # training set (no missing data in label)
    # test = df1_imp_wea_soil[df1_imp_wea_soil["Plant_Height_cm"].isnull()] # testing set
    # # impute(data_encoder_cols, data_featurizer_cols, label_encoder_cols, 
    # #     out="df1_imp_wea_soil_PlantHeightcm_imputer_model", col_to_imp="Plant_Height_cm", 
    # #     train=train, test=test, save="Merged_Trait_Weather_Soil_Data_imputed.csv")
    # imputer = datawig.Imputer(
    #     data_encoders=data_encoder_cols,
    #     data_featurizers=data_featurizer_cols,
    #     label_encoders=label_encoder_cols,
    #     output_path="df1_imp_wea_soil_PlantHeightcm_imputer_model"
    # )
    # imputer.fit(train_df=train)
    # predictions = imputer.predict(test)
    # imputer.save() # save model
    # test["Plant_Height_cm"] = predictions["Plant_Height_cm_imputed"] # replace missing values with predictions
    # df1_iws_imputed = pd.concat([train, test])
    # df1_iws_imputed.to_csv("Merged_Trait_Weather_Soil_Data_imputed.csv", index=False) # save df1_imp_wea_soil with imputed column
    
    # # Delete datasets to clear up memory
    # del df1_imp_wea_soil
    # del train
    # del test
    # del df1_iws_imputed

    # ############################################################################
    # # Impute Ear_Height_cm
    # ############################################################################
    # data_encoder_cols = [NumericalEncoder("Year"), 
    #     CategoricalEncoder("Field_Location"), 
    #     CategoricalEncoder("Hybrid"), 
    #     NumericalEncoder("Plant_Height_cm"), 
    #     NumericalEncoder("Yield_Mg_ha"), 
    #     NumericalEncoder("Grain_Moisture"), 
    #     NumericalEncoder("Organic Matter LOI %"), 
    #     NumericalEncoder("Nitrate-N ppm N"), 
    #     NumericalEncoder("RH2M"), 
    #     NumericalEncoder("%Mg Sat"), 
    #     NumericalEncoder("% Silt"), 
    #     NumericalEncoder("Zinc ppm Zn"), 
    #     NumericalEncoder("Iron ppm Fe"), 
    #     NumericalEncoder("Manganese ppm Mn"), 
    #     NumericalEncoder("Copper ppm Cu")] # columns related to the label column to impute
    # label_encoder_cols = [NumericalEncoder("Ear_Height_cm")] # column to impute
    # data_featurizer_cols = [NumericalFeaturizer("Year"), 
    #     EmbeddingFeaturizer("Field_Location"), 
    #     EmbeddingFeaturizer("Hybrid"), 
    #     NumericalFeaturizer("Plant_Height_cm"), 
    #     NumericalFeaturizer("Yield_Mg_ha"), 
    #     NumericalFeaturizer("Grain_Moisture"), 
    #     NumericalFeaturizer("Organic Matter LOI %"), 
    #     NumericalFeaturizer("Nitrate-N ppm N"), 
    #     NumericalFeaturizer("RH2M"), 
    #     NumericalFeaturizer("%Mg Sat"), 
    #     NumericalFeaturizer("% Silt"), 
    #     NumericalFeaturizer("Zinc ppm Zn"), 
    #     NumericalFeaturizer("Iron ppm Fe"), 
    #     NumericalFeaturizer("Manganese ppm Mn"), 
    #     NumericalFeaturizer("Copper ppm Cu")]
    
    # # Check accuracy of imputer
    # # print("Checking accuracy of imputer for Ear_Height_cm...")
    # df1_imp_wea_soil = pd.read_csv("Merged_Trait_Weather_Soil_Data_imputed.csv")
    # # sub = df1_imp_wea_soil[~df1_imp_wea_soil["Ear_Height_cm"].isnull()] # training set (no missing data in label)
    # # train, test = datawig.utils.random_split(sub, split_ratios=[0.9, 0.1])
    # # impute(data_encoder_cols, data_featurizer_cols, label_encoder_cols, 
    # #     out="TEST_EarHeightcm_imputer_model", col_to_imp="Ear_Height_cm", 
    # #     train=train, test=test, save="Merged_Trait_Weather_Soil_Data_imputed.csv", 
    # #     test_accuracy=True)
    
    # # Back to imputing Ear_Height_cm
    # print("Imputing Ear_Height_cm...")
    # train = df1_imp_wea_soil[~df1_imp_wea_soil["Ear_Height_cm"].isnull()] # training set (no missing data in label)
    # test = df1_imp_wea_soil[df1_imp_wea_soil["Ear_Height_cm"].isnull()] # testing set
    # # impute(data_encoder_cols, data_featurizer_cols, label_encoder_cols, 
    # #     out="df1_imp_wea_soil_EarHeightcm_imputer_model", col_to_imp="Ear_Height_cm", 
    # #     train=train, test=test, save="Merged_Trait_Weather_Soil_Data_imputed.csv")
    # imputer = datawig.Imputer(
    #     data_encoders=data_encoder_cols,
    #     data_featurizers=data_featurizer_cols,
    #     label_encoders=label_encoder_cols,
    #     output_path="df1_imp_wea_soil_EarHeightcm_imputer_model"
    # )
    # imputer.fit(train_df=train, patience=10)
    # predictions, metrics = imputer.transform_and_compute_metrics(test)
    # imputer.save() # save model
    # test["Ear_Height_cm"] = predictions["Ear_Height_cm"] # replace missing values with predictions
    # df1_iws_imputed = pd.concat([train, test])
    # df1_iws_imputed.to_csv("Merged_Trait_Weather_Soil_Data_imputed.csv", index=False)

    # # Delete datasets to clear up memory
    # del df1_imp_wea_soil
    # del train
    # del test
    # del df1_iws_imputed

    # ############################################################################
    # # Impute Root_Lodging_plants
    # ############################################################################
    # # data_encoder_cols = [NumericalEncoder("Year"),
    # #     CategoricalEncoder("Field_Location"), 
    # #     CategoricalEncoder("Hybrid"), 
    # #     NumericalEncoder("Plant_Height_cm"), 
    # #     NumericalEncoder("1:1 S Salts mmho/cm"),
    # #     NumericalEncoder("Magnesium ppm Mg"),
    # #     NumericalEncoder("%Mg Sat")] # columns related to the label column to impute
    # # label_encoder_cols = [NumericalEncoder("Root_Lodging_plants")] # column to impute
    # # data_featurizer_cols = [NumericalFeaturizer("Year"),
    # #     EmbeddingFeaturizer("Field_Location"), 
    # #     EmbeddingFeaturizer("Hybrid"), 
    # #     NumericalFeaturizer("Plant_Height_cm"), 
    # #     NumericalFeaturizer("1:1 S Salts mmho/cm"), 
    # #     NumericalFeaturizer("Magnesium ppm Mg"),
    # #     NumericalFeaturizer("%Mg Sat")]

    # # # Check accuracy of imputer
    # # print("Checking accuracy of imputer for Root_Lodging_plants...")
    # # df1_imp_wea_soil = pd.read_csv("Merged_Trait_Weather_Soil_Data_imputed.csv")
    # # sub = df1_imp_wea_soil[~df1_imp_wea_soil["Root_Lodging_plants"].isnull()] # training set (no missing data in label)
    # # train, test = datawig.utils.random_split(sub, split_ratios=[0.9, 0.1])
    # # impute(data_encoder_cols, data_featurizer_cols, label_encoder_cols, 
    # #     out="TEST_RootLodgingplants_imputer_model", col_to_imp="Root_Lodging_plants", 
    # #     train=train, test=test, save="Merged_Trait_Weather_Soil_Data_imputed.csv", 
    # #     test_accuracy=True)
    
    # ############################################################################
    # # Impute Stalk_Lodging_plants
    # ############################################################################
    # # data_encoder_cols = [NumericalEncoder("Year"),
    # #     CategoricalEncoder("Field_Location"), 
    # #     CategoricalEncoder("Hybrid"), 
    # #     NumericalEncoder("Stand_Count_plants"), 
    # #     NumericalEncoder("Ear_Height_cm"), 
    # #     NumericalEncoder("Zinc ppm Zn"),
    # #     NumericalEncoder("Iron ppm Fe"),
    # #     NumericalEncoder("Copper ppm Cu"),
    # #     NumericalEncoder("Boron ppm B")] # columns related to the label column to impute
    # # label_encoder_cols = [NumericalEncoder("Stalk_Lodging_plants")] # column to impute
    # # data_featurizer_cols = [NumericalFeaturizer("Year"),
    # #     EmbeddingFeaturizer("Field_Location"), 
    # #     EmbeddingFeaturizer("Hybrid"), 
    # #     NumericalFeaturizer("Stand_Count_plants"), 
    # #     NumericalFeaturizer("Ear_Height_cm"), 
    # #     NumericalFeaturizer("Zinc ppm Zn"),
    # #     NumericalFeaturizer("Iron ppm Fe"),
    # #     NumericalFeaturizer("Copper ppm Cu"),
    # #     NumericalFeaturizer("Boron ppm B")]

    # # # Check accuracy of imputer
    # # print("Checking accuracy of imputer for Stalk_Lodging_plants...")
    # # df1_imp_wea_soil = pd.read_csv("Merged_Trait_Weather_Soil_Data_imputed.csv")
    # # sub = df1_imp_wea_soil[~df1_imp_wea_soil["Stalk_Lodging_plants"].isnull()] # training set (no missing data in label)
    # # train, test = datawig.utils.random_split(sub, split_ratios=[0.9, 0.1])
    # # impute(data_encoder_cols, data_featurizer_cols, label_encoder_cols, 
    # #     out="TEST_StalkLodgingplants_imputer_model", col_to_imp="Stalk_Lodging_plants",
    # #     train=train, test=test, save="Merged_Trait_Weather_Soil_Data_imputed.csv",
    # #     test_accuracy=True)
    
    # ############################################################################
    # # Impute Grain_Moisture
    # ############################################################################
    # data_encoder_cols = [NumericalEncoder("Year"),
    #     CategoricalEncoder("Field_Location"), 
    #     CategoricalEncoder("Hybrid"), 
    #     NumericalEncoder("Plant_Height_cm"), 
    #     NumericalEncoder("Ear_Height_cm"), 
    #     NumericalEncoder("Yield_Mg_ha"), 
    #     NumericalEncoder("RH2M"), 
    #     NumericalEncoder("GWETTOP"), 
    #     NumericalEncoder("GWETROOT"), 
    #     NumericalEncoder("GWETPROF"), 
    #     NumericalEncoder("Organic Matter LOI %"), 
    #     NumericalEncoder("Magnesium ppm Mg"), 
    #     NumericalEncoder("%Mg Sat"), 
    #     NumericalEncoder("Mehlich P-III ppm P"), 
    #     NumericalEncoder("Zinc ppm Zn"), 
    #     NumericalEncoder("Iron ppm Fe"), 
    #     NumericalEncoder("Manganese ppm Mn"), 
    #     NumericalEncoder("Copper ppm Cu")] # columns related to the label column to impute
    # label_encoder_cols = [NumericalEncoder("Grain_Moisture")] # column to impute
    # data_featurizer_cols = [NumericalFeaturizer("Year"),
    #     EmbeddingFeaturizer("Field_Location"), 
    #     EmbeddingFeaturizer("Hybrid"), 
    #     NumericalFeaturizer("Plant_Height_cm"), 
    #     NumericalFeaturizer("Ear_Height_cm"), 
    #     NumericalFeaturizer("Yield_Mg_ha"), 
    #     NumericalFeaturizer("RH2M"), 
    #     NumericalFeaturizer("GWETTOP"), 
    #     NumericalFeaturizer("GWETROOT"), 
    #     NumericalFeaturizer("GWETPROF"), 
    #     NumericalFeaturizer("Organic Matter LOI %"), 
    #     NumericalFeaturizer("Magnesium ppm Mg"), 
    #     NumericalFeaturizer("%Mg Sat"), 
    #     NumericalFeaturizer("Mehlich P-III ppm P"), 
    #     NumericalFeaturizer("Zinc ppm Zn"), 
    #     NumericalFeaturizer("Iron ppm Fe"), 
    #     NumericalFeaturizer("Manganese ppm Mn"), 
    #     NumericalFeaturizer("Copper ppm Cu")]
        
    # # Check accuracy of imputer
    # # print("Checking accuracy of imputer for Grain_Moisture...")
    # df1_imp_wea_soil = pd.read_csv("Merged_Trait_Weather_Soil_Data_imputed.csv")
    # # sub = df1_imp_wea_soil[~df1_imp_wea_soil["Grain_Moisture"].isnull()] # training set (no missing data in label)
    # # train, test = datawig.utils.random_split(sub, split_ratios=[0.9, 0.1])
    # # impute(data_encoder_cols, data_featurizer_cols, label_encoder_cols, 
    # #     out="TEST_GrainMoisture_imputer_model", col_to_imp="Grain_Moisture", 
    # #     train=train, test=test, save="Merged_Trait_Weather_Soil_Data_imputed.csv", 
    # #     test_accuracy=True)
    
    # # Back to imputing Grain_Moisture
    # print("Imputing Grain_Moisture...")
    # train = df1_imp_wea_soil[~df1_imp_wea_soil["Grain_Moisture"].isnull()] # training set (no missing data in label)
    # test = df1_imp_wea_soil[df1_imp_wea_soil["Grain_Moisture"].isnull()] # testing set
    # # impute(data_encoder_cols, data_featurizer_cols, label_encoder_cols, 
    # #     out="df1_imp_wea_soil_GrainMoisture_imputer_model", col_to_imp="Grain_Moisture",
    # #     train=train, test=test, save="Merged_Trait_Weather_Soil_Data_imputed.csv")
    # imputer = datawig.Imputer(
    #     data_encoders=data_encoder_cols,
    #     data_featurizers=data_featurizer_cols,
    #     label_encoders=label_encoder_cols,
    #     output_path="df1_imp_wea_soil_GrainMoisture_imputer_model"
    # )
    # imputer.fit(train_df=train)
    # predictions = imputer.predict(test)
    # imputer.save() # save model
    # test["Grain_Moisture"] = predictions["Grain_Moisture_imputed"]
    # df1_iws_imputed = pd.concat([train, test])
    # df1_iws_imputed.to_csv("Merged_Trait_Weather_Soil_Data_imputed.csv", index=False) # save df1_imp_wea_soil with imputed column

    # # Delete datasets to clear up memory
    # del df1_imp_wea_soil
    # del train
    # del test
    # del df1_iws_imputed

    # ############################################################################
    # # Impute Twt_kg_m3
    # ############################################################################
    # data_encoder_cols = [NumericalEncoder("Year"), 
    #     CategoricalEncoder("Field_Location"), 
    #     CategoricalEncoder("Hybrid"), 
    #     NumericalEncoder("% Silt"), 
    #     NumericalEncoder("Calcium ppm Ca"), 
    #     NumericalEncoder("T2M"), 
    #     NumericalEncoder("T2M_MAX"), 
    #     NumericalEncoder("T2M_MIN"), 
    #     NumericalEncoder("T2MWET"), 
    #     NumericalEncoder("ALLSKY_SFC_SW_DWN"), 
    #     NumericalEncoder("ALLSKY_SFC_PAR_TOT")] # columns related to the label column to impute
    # label_encoder_cols = [NumericalEncoder("Twt_kg_m3")] # column to impute
    # data_featurizer_cols = [NumericalFeaturizer("Year"), 
    #     EmbeddingFeaturizer("Field_Location"), 
    #     EmbeddingFeaturizer("Hybrid"), 
    #     NumericalFeaturizer("% Silt"), 
    #     NumericalFeaturizer("Calcium ppm Ca"), 
    #     NumericalFeaturizer("T2M"), 
    #     NumericalFeaturizer("T2M_MAX"), 
    #     NumericalFeaturizer("T2M_MIN"), 
    #     NumericalFeaturizer("T2MWET"), 
    #     NumericalFeaturizer("ALLSKY_SFC_SW_DWN"), 
    #     NumericalFeaturizer("ALLSKY_SFC_PAR_TOT")]
        
    # # Check accuracy of imputer
    # # print("Checking accuracy of imputer for Twt_kg_m3...")
    # df1_imp_wea_soil = pd.read_csv("Merged_Trait_Weather_Soil_Data_imputed.csv")
    # # sub = df1_imp_wea_soil[~df1_imp_wea_soil["Twt_kg_m3"].isnull()] # training set (no missing data in label)
    # # train, test = datawig.utils.random_split(sub, split_ratios=[0.9, 0.1])
    # # impute(data_encoder_cols, data_featurizer_cols, label_encoder_cols, 
    # #     out="TEST_Twtkgm3_imputer_model", col_to_imp="Twt_kg_m3", 
    # #     train=train, test=test, save="Merged_Trait_Weather_Soil_Data_imputed.csv", 
    # #     test_accuracy=True)
    
    # # Back to imputing Twt_kg_m3
    # print("Imputing Twt_kg_m3...")
    # train = df1_imp_wea_soil[~df1_imp_wea_soil["Twt_kg_m3"].isnull()] # training set (no missing data in label)
    # test = df1_imp_wea_soil[df1_imp_wea_soil["Twt_kg_m3"].isnull()] # testing set
    # # impute(data_encoder_cols, data_featurizer_cols, label_encoder_cols, 
    # #     out="df1_imp_wea_soil_Twtkgm3_imputer_model", col_to_imp="Twt_kg_m3",
    # #     train=train, test=test, save="Merged_Trait_Weather_Soil_Data_imputed.csv")
    # imputer = datawig.Imputer(
    #     data_encoders=data_encoder_cols,
    #     data_featurizers=data_featurizer_cols,
    #     label_encoders=label_encoder_cols,
    #     output_path="df1_imp_wea_soil_Twtkgm3_imputer_model"
    # )
    # imputer.fit(train_df=train)
    # predictions = imputer.predict(test)
    # imputer.save() # save model
    # test["Twt_kg_m3"] = predictions["Twt_kg_m3_imputed"]
    # df1_iws_imputed = pd.concat([train, test])
    # df1_iws_imputed.to_csv("Merged_Trait_Weather_Soil_Data_imputed.csv", index=False) # save df1_imp_wea_soil with imputed column

    # # Delete datasets to clear up memory
    # del df1_imp_wea_soil
    # del train
    # del test
    # del df1_iws_imputed

    # ############################################################################
    # # Impute Yield_Mg_ha
    # ############################################################################
    # data_encoder_cols = [NumericalEncoder("Year"), 
    #     CategoricalEncoder("Field_Location"), 
    #     CategoricalEncoder("Hybrid"), 
    #     NumericalEncoder("Stand_Count_plants"), 
    #     NumericalEncoder("Plant_Height_cm"), 
    #     NumericalEncoder("Ear_Height_cm"), 
    #     NumericalEncoder("Grain_Moisture"), 
    #     NumericalEncoder("GWETTOP"), 
    #     NumericalEncoder("GWETROOT"), 
    #     NumericalEncoder("GWETPROF"), 
    #     NumericalEncoder("Organic Matter LOI %"),
    #     NumericalEncoder("Magnesium ppm Mg"),
    #     NumericalEncoder("%Mg Sat")] # columns related to the label column to impute
    # label_encoder_cols = [NumericalEncoder("Yield_Mg_ha")] # column to impute
    # data_featurizer_cols = [NumericalFeaturizer("Year"), 
    #     EmbeddingFeaturizer("Field_Location"), 
    #     EmbeddingFeaturizer("Hybrid"), 
    #     NumericalFeaturizer("Stand_Count_plants"), 
    #     NumericalFeaturizer("Plant_Height_cm"), 
    #     NumericalFeaturizer("Ear_Height_cm"), 
    #     NumericalFeaturizer("Grain_Moisture"), 
    #     NumericalFeaturizer("GWETTOP"), 
    #     NumericalFeaturizer("GWETROOT"), 
    #     NumericalFeaturizer("GWETPROF"), 
    #     NumericalFeaturizer("Organic Matter LOI %"),
    #     NumericalFeaturizer("Magnesium ppm Mg"), 
    #     NumericalFeaturizer("%Mg Sat")] 
    # # Check accuracy of imputer
    # # print("Checking accuracy of imputer for Yield_Mg_ha...")
    # df1_imp_wea_soil = pd.read_csv("Merged_Trait_Weather_Soil_Data_imputed.csv")
    # # sub = df1_imp_wea_soil[~df1_imp_wea_soil["Yield_Mg_ha"].isnull()] # training set (no missing data in label)
    # # train, test = datawig.utils.random_split(sub, split_ratios=[0.9, 0.1])
    # # impute(data_encoder_cols, data_featurizer_cols, label_encoder_cols, 
    # #     out="TEST_YieldMgha_imputer_model", col_to_imp="Yield_Mg_ha", 
    # #     train=train, test=test, save="Merged_Trait_Weather_Soil_Data_yield.csv", 
    # #     test_accuracy=True)
    
    # # Back to imputing Yield_Mg_ha
    # print("Imputing Yield_Mg_ha...")
    # train = df1_imp_wea_soil[~df1_imp_wea_soil["Yield_Mg_ha"].isnull()] # training set (no missing data in label)
    # train.to_csv("yield_train.csv", index=False)
    # test = df1_imp_wea_soil[df1_imp_wea_soil["Yield_Mg_ha"].isnull()] # testing set
    # test.to_csv("yield_test.csv", index=False)
    # # impute(data_encoder_cols, data_featurizer_cols, label_encoder_cols, 
    # #     out="df1_imp_wea_soil_YieldMgha_imputer_model", col_to_imp="Yield_Mg_ha",
    # #     train=train, test=test, save="Merged_Trait_Weather_Soil_Data_imputed.csv")
    # imputer = datawig.Imputer(
    #     data_encoders=data_encoder_cols,
    #     data_featurizers=data_featurizer_cols,
    #     label_encoders=label_encoder_cols,
    #     output_path="df1_imp_wea_soil_YieldMgha_imputer_model"
    # )
    # imputer.fit(train_df=train)
    # predictions = imputer.predict(test)
    # imputer.save() # save model
    # predictions["Yield_Mg_ha_imputed"].to_csv("yield_imputed.csv", index=False)    # save yield predictions
    # #test["Yield_Mg_ha"] = predictions["Yield_Mg_ha_imputed"] # for some reason this is not working

    # # Drop columns with low imputation accuracy and save the merged data
    # df1_imp_wea_soil.drop(["Stand_Count_plants", "Root_Lodging_plants", "Stalk_Lodging_plants"], axis=1, inplace=True)
    # df1_imp_wea_soil.to_csv("Merged_Trait_Weather_Soil_Data_imputed.csv", index=False) # save df1_imp_wea_soil with imputed column

    # # Replace missing values with predicted values manually for yield
    # predictions = pd.read_csv("yield_imputed.csv")
    # df1_imp_wea_soil = pd.read_csv("Merged_Trait_Weather_Soil_Data_imputed.csv")
    # train = df1_imp_wea_soil[~df1_imp_wea_soil["Yield_Mg_ha"].isnull()] # training set (no missing data in label)
    # test = df1_imp_wea_soil[df1_imp_wea_soil["Yield_Mg_ha"].isnull()]
    # test["Yield_Mg_ha"] = predictions["Yield_Mg_ha_imputed"].values
    # df1_iws_imputed = pd.concat([train, test])
    # df1_iws_imputed.iloc[:,0:19].to_csv("1_Training_Trait_Data_2014_2021_cleaned.csv", index=False)
    # print(df1_iws_imputed.iloc[:,0:19].isna().sum())

    # # Delete datasets to clear up memory
    # del df1_imp_wea_soil
    # del train
    # del test
    # del df1_iws_imputed

    ############################################################################
    ################ Impute 3_Training_Soil_Data_2015_2021.csv #################
    # Impute 1:1 Soil pH
    ############################################################################
    data_encoder_cols = [NumericalEncoder("Year"), 
        CategoricalEncoder("Field_Location"), 
        NumericalEncoder("Pollen_DAP_days"), 
        NumericalEncoder("Silk_DAP_days"), 
        NumericalEncoder("E Depth"), 
        NumericalEncoder("WDRF Buffer pH"), 
        NumericalEncoder("Calcium ppm Ca"), 
        NumericalEncoder("Magnesium ppm Mg"), 
        NumericalEncoder("Sodium ppm Na"), 
        NumericalEncoder("CEC/Sum of Cations me/100g"), 
        NumericalEncoder("%Ca Sat"), 
        NumericalEncoder("% Clay"), 
        NumericalEncoder("BpH")]
    label_encoder_cols = [NumericalEncoder("1:1 Soil pH")]
    data_featurizer_cols = [NumericalFeaturizer("Year"), 
        EmbeddingFeaturizer("Field_Location"), 
        NumericalFeaturizer("Pollen_DAP_days"), 
        NumericalFeaturizer("Silk_DAP_days"), 
        NumericalFeaturizer("E Depth"), 
        NumericalFeaturizer("WDRF Buffer pH"), 
        NumericalFeaturizer("Calcium ppm Ca"), 
        NumericalFeaturizer("Magnesium ppm Mg"), 
        NumericalFeaturizer("Sodium ppm Na"), 
        NumericalFeaturizer("CEC/Sum of Cations me/100g"), 
        NumericalFeaturizer("%Ca Sat"), 
        NumericalFeaturizer("% Clay"), 
        NumericalFeaturizer("BpH")]
    # Check accuracy of imputer
    print("Checking accuracy of imputer for 1:1 Soil pH...")
    df1_imp_wea_soil = pd.read_csv("Merged_Trait_Weather_Soil_Data_imputed.csv")
    sub = df1_imp_wea_soil[~df1_imp_wea_soil["1:1 Soil pH"].isnull()] # training set (no missing data in label)
    train, test = datawig.utils.random_split(sub, split_ratios=[0.9, 0.1])
    impute(data_encoder_cols, data_featurizer_cols, label_encoder_cols, 
        out="TEST_1:1SoilpH_imputer_model", col_to_imp="1:1 Soil pH", 
        train=train, test=test, save="Merged_Trait_Weather_Soil_Data_imputed.csv", 
        test_accuracy=True)
    
    # Back to imputing 1:1 Soil pH
    print("Imputing 1:1 Soil pH...")
    train = df1_imp_wea_soil[~df1_imp_wea_soil["1:1 Soil pH"].isnull()] # training set (no missing data in label)
    test = df1_imp_wea_soil[df1_imp_wea_soil["1:1 Soil pH"].isnull()] # testing set
    imputer = datawig.Imputer(
        data_encoders=data_encoder_cols,
        data_featurizers=data_featurizer_cols,
        label_encoders=label_encoder_cols,
        output_path="df1_imp_wea_soil_1:1SoilpH_imputer_model"
    )
    imputer.fit(train_df=train)
    predictions = imputer.predict(test)
    imputer.save() # save model
    test["1:1 Soil pH"] = predictions["1:1 Soil pH_imputed"].values # replace missing values with predictions
    df1_iws_imputed = pd.concat([train, test])
    df1_iws_imputed.to_csv("Merged_Trait_Weather_Soil_Data_imputed.csv", index=False) # save df1_imp_wea_soil with imputed column

    ############################################################################
    # Impute WDRF Buffer pH
    ############################################################################
    data_encoder_cols = [NumericalEncoder("Year"), 
        CategoricalEncoder("Field_Location"), 
        NumericalEncoder("Plot_Area_ha"), 
        NumericalEncoder("E Depth"), 
        NumericalEncoder("1:1 Soil pH"), 
        NumericalEncoder("Calcium ppm Ca"), 
        NumericalEncoder("Sodium ppm Na"), 
        NumericalEncoder("%Ca Sat"), 
        NumericalEncoder("%Mg Sat"), 
        NumericalEncoder("% Silt"), 
        NumericalEncoder("%Na Sat")]
    label_encoder_cols = [NumericalEncoder("WDRF Buffer pH")]
    data_featurizer_cols = [NumericalFeaturizer("Year"), 
        EmbeddingFeaturizer("Field_Location"), 
        NumericalFeaturizer("Plot_Area_ha"), 
        NumericalFeaturizer("E Depth"), 
        NumericalFeaturizer("1:1 Soil pH"), 
        NumericalFeaturizer("Calcium ppm Ca"), 
        NumericalFeaturizer("Sodium ppm Na"), 
        NumericalFeaturizer("%Ca Sat"), 
        NumericalFeaturizer("%Mg Sat"), 
        NumericalFeaturizer("% Silt"), 
        NumericalFeaturizer("%Na Sat")]
    # Check accuracy of imputer
    print("Checking accuracy of imputer for WDRF Buffer pH...")
    df1_imp_wea_soil = pd.read_csv("Merged_Trait_Weather_Soil_Data_imputed.csv")
    sub = df1_imp_wea_soil[~df1_imp_wea_soil["WDRF Buffer pH"].isnull()] # training set (no missing data in label)
    train, test = datawig.utils.random_split(sub, split_ratios=[0.9, 0.1])
    impute(data_encoder_cols, data_featurizer_cols, label_encoder_cols, 
        out="TEST_WDRFBufferpH_imputer_model", col_to_imp="WDRF Buffer pH", 
        train=train, test=test, save="Merged_Trait_Weather_Soil_Data_imputed.csv", 
        test_accuracy=True)
    
    # Back to imputing WDRF Buffer pH
    print("Imputing WDRF Buffer pH...")
    train = df1_imp_wea_soil[~df1_imp_wea_soil["WDRF Buffer pH"].isnull()] # training set (no missing data in label)
    test = df1_imp_wea_soil[df1_imp_wea_soil["WDRF Buffer pH"].isnull()] # testing set
    imputer = datawig.Imputer(
        data_encoders=data_encoder_cols,
        data_featurizers=data_featurizer_cols,
        label_encoders=label_encoder_cols,
        output_path="df1_imp_wea_soil_WDRFBufferpH_imputer_model"
    )
    imputer.fit(train_df=train)
    predictions = imputer.predict(test)
    imputer.save() # save model
    test["WDRF Buffer pH"] = predictions["WDRF Buffer pH_imputed"].values # replace missing values with predictions
    df1_iws_imputed = pd.concat([train, test])
    df1_iws_imputed.to_csv("Merged_Trait_Weather_Soil_Data_imputed.csv", index=False) # save df1_imp_wea_soil with imputed column

    ############################################################################
    # Impute Organic Matter LOI %
    ############################################################################
    data_encoder_cols = [NumericalEncoder("Year"), 
        CategoricalEncoder("Field_Location"), 
        NumericalEncoder("Plot_Area_ha"), 
        NumericalEncoder("Plant_Height_cm"), 
        NumericalEncoder("Ear_Height_cm"), 
        NumericalEncoder("Yield_Mg_ha"), 
        NumericalEncoder("Grain_Moisture"), 
        NumericalEncoder("RH2M"), 
        NumericalEncoder("WS2M"), 
        NumericalEncoder("GWETTOP"), 
        NumericalEncoder("GWETROOT"), 
        NumericalEncoder("GWETPROF"), 
        NumericalEncoder("1:1 S Salts mmho/cm"), 
        NumericalEncoder("Nitrate-N ppm N"), 
        NumericalEncoder("Calcium ppm Ca"), 
        NumericalEncoder("Magnesium ppm Mg"), 
        NumericalEncoder("CEC/Sum of Cations me/100g"), 
        NumericalEncoder("%Mg Sat"), 
        NumericalEncoder("% Silt"), 
        NumericalEncoder("% Clay")]
    label_encoder_cols = [NumericalEncoder("Organic Matter LOI %")]
    data_featurizer_cols = [NumericalFeaturizer("Year"), 
        EmbeddingFeaturizer("Field_Location"), 
        NumericalFeaturizer("Plot_Area_ha"), 
        NumericalFeaturizer("Plant_Height_cm"), 
        NumericalFeaturizer("Ear_Height_cm"), 
        NumericalFeaturizer("Yield_Mg_ha"), 
        NumericalFeaturizer("Grain_Moisture"), 
        NumericalFeaturizer("RH2M"), 
        NumericalFeaturizer("WS2M"), 
        NumericalFeaturizer("GWETTOP"), 
        NumericalFeaturizer("GWETROOT"), 
        NumericalFeaturizer("GWETPROF"), 
        NumericalFeaturizer("1:1 S Salts mmho/cm"), 
        NumericalFeaturizer("Nitrate-N ppm N"), 
        NumericalFeaturizer("Calcium ppm Ca"), 
        NumericalFeaturizer("Magnesium ppm Mg"), 
        NumericalFeaturizer("CEC/Sum of Cations me/100g"), 
        NumericalFeaturizer("%Mg Sat"), 
        NumericalFeaturizer("% Silt"), 
        NumericalFeaturizer("% Clay")]
    # Check accuracy of imputer
    print("Checking accuracy of imputer for Organic Matter LOI %...")
    df1_imp_wea_soil = pd.read_csv("Merged_Trait_Weather_Soil_Data_imputed.csv")
    sub = df1_imp_wea_soil[~df1_imp_wea_soil["Organic Matter LOI %"].isnull()] # training set (no missing data in label)
    train, test = datawig.utils.random_split(sub, split_ratios=[0.9, 0.1])
    impute(data_encoder_cols, data_featurizer_cols, label_encoder_cols, 
        out="TEST_OrganicMatterLOI%_imputer_model", col_to_imp="Organic Matter LOI %", 
        train=train, test=test, save="Merged_Trait_Weather_Soil_Data_imputed.csv", 
        test_accuracy=True)
    
    # Back to imputing Organic Matter LOI %
    print("Imputing Organic Matter LOI %...")
    train = df1_imp_wea_soil[~df1_imp_wea_soil["Organic Matter LOI %"].isnull()] # training set (no missing data in label)
    test = df1_imp_wea_soil[df1_imp_wea_soil["Organic Matter LOI %"].isnull()] # testing set
    imputer = datawig.Imputer(
        data_encoders=data_encoder_cols,
        data_featurizers=data_featurizer_cols,
        label_encoders=label_encoder_cols,
        output_path="df1_imp_wea_soil_OrganicMatterLOI%_imputer_model"
    )
    imputer.fit(train_df=train)
    predictions = imputer.predict(test)
    imputer.save() # save model
    test["Organic Matter LOI %"] = predictions["Organic Matter LOI %_imputed"].values # replace missing values with predictions
    df1_iws_imputed = pd.concat([train, test])
    df1_iws_imputed.to_csv("Merged_Trait_Weather_Soil_Data_imputed.csv", index=False) # save df1_imp_wea_soil with imputed column

    ############################################################################
    # Impute Nitrate-N ppm N
    ############################################################################
    data_encoder_cols = [NumericalEncoder("Year"), 
        CategoricalEncoder("Field_Location"), 
        NumericalEncoder("Plant_Height_cm"), 
        NumericalEncoder("Ear_Height_cm"), 
        NumericalEncoder("WS2M"), 
        NumericalEncoder("Year_y"), 
        NumericalEncoder("1:1 S Salts mmho/cm"), 
        NumericalEncoder("Organic Matter LOI %"), 
        NumericalEncoder("lbs N/A"), 
        NumericalEncoder("Potassium ppm K"), 
        NumericalEncoder("Sulfate-S ppm S"), 
        NumericalEncoder("Mehlich P-III ppm P"), 
        NumericalEncoder("% Silt")]
    label_encoder_cols = [NumericalEncoder("Nitrate-N ppm N")]
    data_featurizer_cols = [NumericalFeaturizer("Year"), 
        EmbeddingFeaturizer("Field_Location"), 
        NumericalFeaturizer("Plant_Height_cm"), 
        NumericalFeaturizer("Ear_Height_cm"), 
        NumericalFeaturizer("WS2M"), 
        NumericalFeaturizer("Year_y"), 
        NumericalFeaturizer("1:1 S Salts mmho/cm"), 
        NumericalFeaturizer("Organic Matter LOI %"), 
        NumericalFeaturizer("lbs N/A"), 
        NumericalFeaturizer("Potassium ppm K"), 
        NumericalFeaturizer("Sulfate-S ppm S"), 
        NumericalFeaturizer("Mehlich P-III ppm P"), 
        NumericalFeaturizer("% Silt")]
    # Check accuracy of imputer
    print("Checking accuracy of imputer for Nitrate-N ppm N...")
    df1_imp_wea_soil = pd.read_csv("Merged_Trait_Weather_Soil_Data_imputed.csv")
    sub = df1_imp_wea_soil[~df1_imp_wea_soil["Nitrate-N ppm N"].isnull()] # training set (no missing data in label)
    train, test = datawig.utils.random_split(sub, split_ratios=[0.9, 0.1])
    impute(data_encoder_cols, data_featurizer_cols, label_encoder_cols, 
        out="TEST_Nitrate-NppmN_imputer_model", col_to_imp="Nitrate-N ppm N", 
        train=train, test=test, save="Merged_Trait_Weather_Soil_Data_imputed.csv", 
        test_accuracy=True)
    
    # Back to imputing Nitrate-N ppm N
    print("Imputing Nitrate-N ppm N...")
    train = df1_imp_wea_soil[~df1_imp_wea_soil["Nitrate-N ppm N"].isnull()] # training set (no missing data in label)
    test = df1_imp_wea_soil[df1_imp_wea_soil["Nitrate-N ppm N"].isnull()] # testing set
    imputer = datawig.Imputer(
        data_encoders=data_encoder_cols,
        data_featurizers=data_featurizer_cols,
        label_encoders=label_encoder_cols,
        output_path="df1_imp_wea_soil_Nitrate-NppmN_imputer_model"
    )
    imputer.fit(train_df=train)
    predictions = imputer.predict(test)
    imputer.save() # save model
    test["Nitrate-N ppm N"] = predictions["Nitrate-N ppm N_imputed"].values # replace missing values with predictions
    df1_iws_imputed = pd.concat([train, test])
    df1_iws_imputed.to_csv("Merged_Trait_Weather_Soil_Data_imputed.csv", index=False) # save df1_imp_wea_soil with imputed column

    ############################################################################
    # Impute lbs N/A
    ############################################################################
    data_encoder_cols = [NumericalEncoder("Year"), 
        CategoricalEncoder("Field_Location"), 
        NumericalEncoder("Plant_Height_cm"), 
        NumericalEncoder("Ear_Height_cm"), 
        NumericalEncoder("WS2M"), 
        NumericalEncoder("Year_y"), 
        NumericalEncoder("1:1 S Salts mmho/cm"), 
        NumericalEncoder("Organic Matter LOI %"), 
        NumericalEncoder("Nitrate-N ppm N"), 
        NumericalEncoder("Potassium ppm K"), 
        NumericalEncoder("Sulfate-S ppm S"), 
        NumericalEncoder("%Na Sat"), 
        NumericalEncoder("% Silt")]
    label_encoder_cols = [NumericalEncoder("lbs N/A")]
    data_featurizer_cols = [NumericalFeaturizer("Year"), 
        EmbeddingFeaturizer("Field_Location"), 
        NumericalFeaturizer("Plant_Height_cm"), 
        NumericalFeaturizer("Ear_Height_cm"), 
        NumericalFeaturizer("WS2M"), 
        NumericalFeaturizer("Year_y"), 
        NumericalFeaturizer("1:1 S Salts mmho/cm"), 
        NumericalFeaturizer("Organic Matter LOI %"), 
        NumericalFeaturizer("Nitrate-N ppm N"), 
        NumericalFeaturizer("Potassium ppm K"), 
        NumericalFeaturizer("Sulfate-S ppm S"), 
        NumericalFeaturizer("%Na Sat"), 
        NumericalFeaturizer("% Silt")]
    # Check accuracy of imputer
    print("Checking accuracy of imputer for lbs N/A...")
    df1_imp_wea_soil = pd.read_csv("Merged_Trait_Weather_Soil_Data_imputed.csv")
    sub = df1_imp_wea_soil[~df1_imp_wea_soil["lbs N/A"].isnull()] # training set (no missing data in label)
    train, test = datawig.utils.random_split(sub, split_ratios=[0.9, 0.1])
    impute(data_encoder_cols, data_featurizer_cols, label_encoder_cols, 
        out="TEST_lbsN/A_imputer_model", col_to_imp="lbs N/A", 
        train=train, test=test, save="Merged_Trait_Weather_Soil_Data_imputed.csv", 
        test_accuracy=True)
    
    # Back to imputing lbs N/A
    print("Imputing lbs N/A...")
    train = df1_imp_wea_soil[~df1_imp_wea_soil["lbs N/A"].isnull()] # training set (no missing data in label)
    test = df1_imp_wea_soil[df1_imp_wea_soil["lbs N/A"].isnull()] # testing set
    imputer = datawig.Imputer(
        data_encoders=data_encoder_cols,
        data_featurizers=data_featurizer_cols,
        label_encoders=label_encoder_cols,
        output_path="df1_imp_wea_soil_lbsN/A_imputer_model"
    )
    imputer.fit(train_df=train)
    predictions = imputer.predict(test)
    imputer.save() # save model
    test["lbs N/A"] = predictions["lbs N/A_imputed"].values # replace missing values with predictions
    df1_iws_imputed = pd.concat([train, test])
    df1_iws_imputed.to_csv("Merged_Trait_Weather_Soil_Data_imputed.csv", index=False) # save df1_imp_wea_soil with imputed column

    ############################################################################
    # Impute Sulfate-S ppm S
    ############################################################################
    data_encoder_cols = [NumericalEncoder("Year"), 
        CategoricalEncoder("Field_Location"), 
        NumericalEncoder("1:1 S Salts mmho/cm"), 
        NumericalEncoder("Nitrate-N ppm N"), 
        NumericalEncoder("lbs N/A"), 
        NumericalEncoder("Potassium ppm K"), 
        NumericalEncoder("Calcium ppm Ca"), 
        NumericalEncoder("Magnesium ppm Mg"), 
        NumericalEncoder("Sodium ppm Na"), 
        NumericalEncoder("CEC/Sum of Cations me/100g"), 
        NumericalEncoder("%Na Sat"), 
        NumericalEncoder("% Clay")]
    label_encoder_cols = [NumericalEncoder("Sulfate-S ppm S")]
    data_featurizer_cols = [NumericalFeaturizer("Year"), 
        EmbeddingFeaturizer("Field_Location"), 
        NumericalFeaturizer("1:1 S Salts mmho/cm"), 
        NumericalFeaturizer("Nitrate-N ppm N"), 
        NumericalFeaturizer("lbs N/A"), 
        NumericalFeaturizer("Potassium ppm K"), 
        NumericalFeaturizer("Calcium ppm Ca"), 
        NumericalFeaturizer("Magnesium ppm Mg"), 
        NumericalFeaturizer("Sodium ppm Na"), 
        NumericalFeaturizer("CEC/Sum of Cations me/100g"), 
        NumericalFeaturizer("%Na Sat"), 
        NumericalFeaturizer("% Clay")]
    # Check accuracy of imputer
    print("Checking accuracy of imputer for Sulfate-S ppm S...")
    df1_imp_wea_soil = pd.read_csv("Merged_Trait_Weather_Soil_Data_imputed.csv")
    sub = df1_imp_wea_soil[~df1_imp_wea_soil["Sulfate-S ppm S"].isnull()] # training set (no missing data in label)
    train, test = datawig.utils.random_split(sub, split_ratios=[0.9, 0.1])
    impute(data_encoder_cols, data_featurizer_cols, label_encoder_cols, 
        out="TEST_Sulfate-SppmS_imputer_model", col_to_imp="Sulfate-S ppm S", 
        train=train, test=test, save="Merged_Trait_Weather_Soil_Data_imputed.csv", 
        test_accuracy=True)
    
    # Back to imputing Sulfate-S ppm S
    print("Imputing Sulfate-S ppm S...")
    train = df1_imp_wea_soil[~df1_imp_wea_soil["Sulfate-S ppm S"].isnull()] # training set (no missing data in label)
    test = df1_imp_wea_soil[df1_imp_wea_soil["Sulfate-S ppm S"].isnull()] # testing set
    imputer = datawig.Imputer(
        data_encoders=data_encoder_cols,
        data_featurizers=data_featurizer_cols,
        label_encoders=label_encoder_cols,
        output_path="df1_imp_wea_soil_Sulfate-SppmS_imputer_model"
    )
    imputer.fit(train_df=train)
    predictions = imputer.predict(test)
    imputer.save() # save model
    test["Sulfate-S ppm S"] = predictions["Sulfate-S ppm S_imputed"].values # replace missing values with predictions
    df1_iws_imputed = pd.concat([train, test])
    df1_iws_imputed.to_csv("Merged_Trait_Weather_Soil_Data_imputed.csv", index=False) # save df1_imp_wea_soil with imputed column

    ############################################################################
    # Impute Calcium ppm Ca
    ############################################################################
    data_encoder_cols = [NumericalEncoder("Year"), 
        CategoricalEncoder("Field_Location"), 
        NumericalEncoder("Pollen_DAP_days"), 
        NumericalEncoder("Silk_DAP_days"), 
        NumericalEncoder("WS2M"), 
        NumericalEncoder("E Depth"), 
        NumericalEncoder("1:1 Soil pH"), 
        NumericalEncoder("WDRF Buffer pH"), 
        NumericalEncoder("1:1 S Salts mmho/cm"), 
        NumericalEncoder("Texture No"), 
        NumericalEncoder("Organic Matter LOI %"), 
        NumericalEncoder("Potassium ppm K"), 
        NumericalEncoder("Sulfate-S ppm S"), 
        NumericalEncoder("Magnesium ppm Mg"), 
        NumericalEncoder("Sodium ppm Na"), 
        NumericalEncoder("CEC/Sum of Cations me/100g"), 
        NumericalEncoder("%Ca Sat"), 
        NumericalEncoder("% Silt"), 
        NumericalEncoder("% Clay")]
    label_encoder_cols = [NumericalEncoder("Calcium ppm Ca")]
    data_featurizer_cols = [NumericalFeaturizer("Year"), 
        EmbeddingFeaturizer("Field_Location"), 
        NumericalFeaturizer("Pollen_DAP_days"), 
        NumericalFeaturizer("Silk_DAP_days"), 
        NumericalFeaturizer("WS2M"), 
        NumericalFeaturizer("E Depth"), 
        NumericalFeaturizer("1:1 Soil pH"), 
        NumericalFeaturizer("WDRF Buffer pH"), 
        NumericalFeaturizer("1:1 S Salts mmho/cm"), 
        NumericalFeaturizer("Texture No"), 
        NumericalFeaturizer("Organic Matter LOI %"), 
        NumericalFeaturizer("Potassium ppm K"), 
        NumericalFeaturizer("Sulfate-S ppm S"), 
        NumericalFeaturizer("Magnesium ppm Mg"), 
        NumericalFeaturizer("Sodium ppm Na"), 
        NumericalFeaturizer("CEC/Sum of Cations me/100g"), 
        NumericalFeaturizer("%Ca Sat"), 
        NumericalFeaturizer("% Silt"), 
        NumericalFeaturizer("% Clay")]
    # Check accuracy of imputer
    print("Checking accuracy of imputer for Calcium ppm Ca...")
    df1_imp_wea_soil = pd.read_csv("Merged_Trait_Weather_Soil_Data_imputed.csv")
    sub = df1_imp_wea_soil[~df1_imp_wea_soil["Calcium ppm Ca"].isnull()] # training set (no missing data in label)
    train, test = datawig.utils.random_split(sub, split_ratios=[0.9, 0.1])
    impute(data_encoder_cols, data_featurizer_cols, label_encoder_cols, 
        out="TEST_CalciumppmCa_imputer_model", col_to_imp="Calcium ppm Ca", 
        train=train, test=test, save="Merged_Trait_Weather_Soil_Data_imputed.csv", 
        test_accuracy=True)
    
    # Back to imputing Calcium ppm Ca
    print("Imputing Calcium ppm Ca...")
    train = df1_imp_wea_soil[~df1_imp_wea_soil["Calcium ppm Ca"].isnull()] # training set (no missing data in label)
    test = df1_imp_wea_soil[df1_imp_wea_soil["Calcium ppm Ca"].isnull()] # testing set
    imputer = datawig.Imputer(
        data_encoders=data_encoder_cols,
        data_featurizers=data_featurizer_cols,
        label_encoders=label_encoder_cols,
        output_path="df1_imp_wea_soil_CalciumppmCa_imputer_model"
    )
    imputer.fit(train_df=train)
    predictions = imputer.predict(test)
    imputer.save() # save model
    test["Calcium ppm Ca"] = predictions["Calcium ppm Ca_imputed"].values # replace missing values with predictions
    df1_iws_imputed = pd.concat([train, test])
    df1_iws_imputed.to_csv("Merged_Trait_Weather_Soil_Data_imputed.csv", index=False) # save df1_imp_wea_soil with imputed column

    ############################################################################
    # Impute Magnesium ppm Mg
    ############################################################################
    data_encoder_cols = [NumericalEncoder("Year"), 
        CategoricalEncoder("Field_Location"), 
        NumericalEncoder("Plot_Area_ha"), 
        NumericalEncoder("Grain_Moisture"), 
        NumericalEncoder("WS2M"), 
        NumericalEncoder("1:1 S Salts mmho/cm"), 
        NumericalEncoder("Organic Matter LOI %"), 
        NumericalEncoder("Potassium ppm K"), 
        NumericalEncoder("Sulfate-S ppm S"), 
        NumericalEncoder("Calcium ppm Ca"), 
        NumericalEncoder("Sodium ppm Na"), 
        NumericalEncoder("CEC/Sum of Cations me/100g"), 
        NumericalEncoder("%Mg Sat"), 
        NumericalEncoder("%Ca Sat"), 
        NumericalEncoder("% Silt"), 
        NumericalEncoder("% Clay")]
    label_encoder_cols = [NumericalEncoder("Magnesium ppm Mg")]
    data_featurizer_cols = [NumericalFeaturizer("Year"), 
        EmbeddingFeaturizer("Field_Location"), 
        NumericalFeaturizer("Plot_Area_ha"), 
        NumericalFeaturizer("Grain_Moisture"), 
        NumericalFeaturizer("WS2M"), 
        NumericalFeaturizer("1:1 S Salts mmho/cm"), 
        NumericalFeaturizer("Organic Matter LOI %"), 
        NumericalFeaturizer("Potassium ppm K"), 
        NumericalFeaturizer("Sulfate-S ppm S"), 
        NumericalFeaturizer("Calcium ppm Ca"), 
        NumericalFeaturizer("Sodium ppm Na"), 
        NumericalFeaturizer("CEC/Sum of Cations me/100g"), 
        NumericalFeaturizer("%Mg Sat"), 
        NumericalFeaturizer("%Ca Sat"), 
        NumericalFeaturizer("% Silt"), 
        NumericalFeaturizer("% Clay")]
    # Check accuracy of imputer
    print("Checking accuracy of imputer for Magnesium ppm Mg...")
    df1_imp_wea_soil = pd.read_csv("Merged_Trait_Weather_Soil_Data_imputed.csv")
    sub = df1_imp_wea_soil[~df1_imp_wea_soil["Magnesium ppm Mg"].isnull()] # training set (no missing data in label)
    train, test = datawig.utils.random_split(sub, split_ratios=[0.9, 0.1])
    impute(data_encoder_cols, data_featurizer_cols, label_encoder_cols, 
        out="TEST_MagnesiumppmMg_imputer_model", col_to_imp="Magnesium ppm Mg", 
        train=train, test=test, save="Merged_Trait_Weather_Soil_Data_imputed.csv", 
        test_accuracy=True)
    
    # Back to imputing Magnesium ppm Mg
    print("Imputing Magnesium ppm Mg...")
    train = df1_imp_wea_soil[~df1_imp_wea_soil["Magnesium ppm Mg"].isnull()] # training set (no missing data in label)
    test = df1_imp_wea_soil[df1_imp_wea_soil["Magnesium ppm Mg"].isnull()] # testing set
    imputer = datawig.Imputer(
        data_encoders=data_encoder_cols,
        data_featurizers=data_featurizer_cols,
        label_encoders=label_encoder_cols,
        output_path="df1_imp_wea_soil_MagnesiumppmMg_imputer_model"
    )
    imputer.fit(train_df=train)
    predictions = imputer.predict(test)
    imputer.save() # save model
    test["Magnesium ppm Mg"] = predictions["Magnesium ppm Mg_imputed"].values # replace missing values with predictions
    df1_iws_imputed = pd.concat([train, test])
    df1_iws_imputed.to_csv("Merged_Trait_Weather_Soil_Data_imputed.csv", index=False) # save df1_imp_wea_soil with imputed column

    ############################################################################
    # Impute Sodium ppm Na
    ############################################################################
    data_encoder_cols = [NumericalEncoder("Year"), 
        CategoricalEncoder("Field_Location"), 
        NumericalEncoder("QV2M"), 
        NumericalEncoder("T2M_MAX"), 
        NumericalEncoder("T2M_MIN"), 
        NumericalEncoder("T2M"), 
        NumericalEncoder("1:1 Soil pH"), 
        NumericalEncoder("WDRF Buffer pH"), 
        NumericalEncoder("1:1 S Salts mmho/cm"), 
        NumericalEncoder("Potassium ppm K"), 
        NumericalEncoder("Sulfate-S ppm S"), 
        NumericalEncoder("Calcium ppm Ca"), 
        NumericalEncoder("Magnesium ppm Mg"), 
        NumericalEncoder("CEC/Sum of Cations me/100g"), 
        NumericalEncoder("%Ca Sat"), 
        NumericalEncoder("%Na Sat"), 
        NumericalEncoder("% Clay")]
    label_encoder_cols = [NumericalEncoder("Sodium ppm Na")]
    data_featurizer_cols = [NumericalFeaturizer("Year"), 
        EmbeddingFeaturizer("Field_Location"), 
        NumericalFeaturizer("QV2M"), 
        NumericalFeaturizer("T2M_MAX"), 
        NumericalFeaturizer("T2M_MIN"), 
        NumericalFeaturizer("T2M"), 
        NumericalFeaturizer("1:1 Soil pH"), 
        NumericalFeaturizer("WDRF Buffer pH"), 
        NumericalFeaturizer("1:1 S Salts mmho/cm"), 
        NumericalFeaturizer("Potassium ppm K"), 
        NumericalFeaturizer("Sulfate-S ppm S"), 
        NumericalFeaturizer("Calcium ppm Ca"), 
        NumericalFeaturizer("Magnesium ppm Mg"), 
        NumericalFeaturizer("CEC/Sum of Cations me/100g"), 
        NumericalFeaturizer("%Ca Sat"), 
        NumericalFeaturizer("%Na Sat"), 
        NumericalFeaturizer("% Clay")]
    # Check accuracy of imputer
    print("Checking accuracy of imputer for Sodium ppm Na...")
    df1_imp_wea_soil = pd.read_csv("Merged_Trait_Weather_Soil_Data_imputed.csv")
    sub = df1_imp_wea_soil[~df1_imp_wea_soil["Sodium ppm Na"].isnull()] # training set (no missing data in label)
    train, test = datawig.utils.random_split(sub, split_ratios=[0.9, 0.1])
    impute(data_encoder_cols, data_featurizer_cols, label_encoder_cols, 
        out="TEST_SodiumppmNa_imputer_model", col_to_imp="Sodium ppm Na", 
        train=train, test=test, save="Merged_Trait_Weather_Soil_Data_imputed.csv", 
        test_accuracy=True)
    
    # Back to imputing Sodium ppm Na
    print("Imputing Sodium ppm Na...")
    train = df1_imp_wea_soil[~df1_imp_wea_soil["Sodium ppm Na"].isnull()] # training set (no missing data in label)
    test = df1_imp_wea_soil[df1_imp_wea_soil["Sodium ppm Na"].isnull()] # testing set
    imputer = datawig.Imputer(
        data_encoders=data_encoder_cols,
        data_featurizers=data_featurizer_cols,
        label_encoders=label_encoder_cols,
        output_path="df1_imp_wea_soil_SodiumppmNa_imputer_model"
    )
    imputer.fit(train_df=train)
    predictions = imputer.predict(test)
    imputer.save() # save model
    test["Sodium ppm Na"] = predictions["Sodium ppm Na_imputed"].values # replace missing values with predictions
    df1_iws_imputed = pd.concat([train, test])
    df1_iws_imputed.to_csv("Merged_Trait_Weather_Soil_Data_imputed.csv", index=False) # save df1_imp_wea_soil with imputed column

    ############################################################################
    # Impute CEC/Sum of Cations me/100g
    ############################################################################
    data_encoder_cols = [NumericalEncoder("Year"), 
        CategoricalEncoder("Field_Location"), 
        NumericalEncoder("WS2M"), 
        NumericalEncoder("1:1 Soil pH"), 
        NumericalEncoder("1:1 S Salts mmho/cm"), 
        NumericalEncoder("Organic Matter LOI %"), 
        NumericalEncoder("Potassium ppm K"), 
        NumericalEncoder("Sulfate-S ppm S"), 
        NumericalEncoder("Calcium ppm Ca"), 
        NumericalEncoder("Magnesium ppm Mg"), 
        NumericalEncoder("Sodium ppm Na"), 
        NumericalEncoder("%Ca Sat"), 
        NumericalEncoder("% Silt"), 
        NumericalEncoder("% Clay")]
    label_encoder_cols = [NumericalEncoder("CEC/Sum of Cations me/100g")]
    data_featurizer_cols = [NumericalFeaturizer("Year"), 
        EmbeddingFeaturizer("Field_Location"), 
        NumericalFeaturizer("WS2M"), 
        NumericalFeaturizer("1:1 Soil pH"), 
        NumericalFeaturizer("1:1 S Salts mmho/cm"), 
        NumericalFeaturizer("Organic Matter LOI %"), 
        NumericalFeaturizer("Potassium ppm K"), 
        NumericalFeaturizer("Sulfate-S ppm S"), 
        NumericalFeaturizer("Calcium ppm Ca"), 
        NumericalFeaturizer("Magnesium ppm Mg"), 
        NumericalFeaturizer("Sodium ppm Na"), 
        NumericalFeaturizer("%Ca Sat"), 
        NumericalFeaturizer("% Silt"), 
        NumericalFeaturizer("% Clay")]
    # Check accuracy of imputer
    print("Checking accuracy of imputer for CEC/Sum of Cations me/100g...")
    df1_imp_wea_soil = pd.read_csv("Merged_Trait_Weather_Soil_Data_imputed.csv")
    sub = df1_imp_wea_soil[~df1_imp_wea_soil["CEC/Sum of Cations me/100g"].isnull()] # training set (no missing data in label)
    train, test = datawig.utils.random_split(sub, split_ratios=[0.9, 0.1])
    impute(data_encoder_cols, data_featurizer_cols, label_encoder_cols, 
        out="TEST_CEC/SumofCationsme/100g_imputer_model", col_to_imp="CEC/Sum of Cations me/100g", 
        train=train, test=test, save="Merged_Trait_Weather_Soil_Data_imputed.csv", 
        test_accuracy=True)
    
    # Back to imputing CEC/Sum of Cations me/100g
    print("Imputing CEC/Sum of Cations me/100g...")
    train = df1_imp_wea_soil[~df1_imp_wea_soil["CEC/Sum of Cations me/100g"].isnull()] # training set (no missing data in label)
    test = df1_imp_wea_soil[df1_imp_wea_soil["CEC/Sum of Cations me/100g"].isnull()] # testing set
    imputer = datawig.Imputer(
        data_encoders=data_encoder_cols,
        data_featurizers=data_featurizer_cols,
        label_encoders=label_encoder_cols,
        output_path="df1_imp_wea_soil_CEC/SumofCationsme/100g_imputer_model"
    )
    imputer.fit(train_df=train)
    predictions = imputer.predict(test)
    imputer.save() # save model
    test["CEC/Sum of Cations me/100g"] = predictions["CEC/Sum of Cations me/100g_imputed"].values # replace missing values with predictions
    df1_iws_imputed = pd.concat([train, test])
    df1_iws_imputed.to_csv("Merged_Trait_Weather_Soil_Data_imputed.csv", index=False) # save df1_imp_wea_soil with imputed column

    ############################################################################
    # Impute %H Sat
    ############################################################################
    data_encoder_cols = [NumericalEncoder("Year"), 
        CategoricalEncoder("Field_Location"), 
        NumericalEncoder("QV2M"), 
        NumericalEncoder("T2MDEW"), 
        NumericalEncoder("PS"), 
        NumericalEncoder("ALLSKY_SFC_SW_DWN"), 
        NumericalEncoder("ALLSKY_SFC_PAR_TOT"), 
        NumericalEncoder("T2M_MAX"), 
        NumericalEncoder("T2M_MIN"), 
        NumericalEncoder("T2MWET"), 
        NumericalEncoder("T2M"), 
        NumericalEncoder("Year_y"), 
        NumericalEncoder("%K Sat"), 
        NumericalEncoder("%Na Sat"), 
        NumericalEncoder("Mehlich P-III ppm P"), 
        NumericalEncoder("% Sand")]
    label_encoder_cols = [NumericalEncoder("%H Sat")]
    data_featurizer_cols = [NumericalFeaturizer("Year"), 
        EmbeddingFeaturizer("Field_Location"), 
        NumericalFeaturizer("QV2M"), 
        NumericalFeaturizer("T2MDEW"), 
        NumericalFeaturizer("PS"), 
        NumericalFeaturizer("ALLSKY_SFC_SW_DWN"), 
        NumericalFeaturizer("ALLSKY_SFC_PAR_TOT"), 
        NumericalFeaturizer("T2M_MAX"), 
        NumericalFeaturizer("T2M_MIN"), 
        NumericalFeaturizer("T2MWET"), 
        NumericalFeaturizer("T2M"), 
        NumericalFeaturizer("Year_y"), 
        NumericalFeaturizer("%K Sat"), 
        NumericalFeaturizer("%Na Sat"), 
        NumericalFeaturizer("Mehlich P-III ppm P"), 
        NumericalFeaturizer("% Sand")]
    # Check accuracy of imputer
    print("Checking accuracy of imputer for %H Sat...")
    df1_imp_wea_soil = pd.read_csv("Merged_Trait_Weather_Soil_Data_imputed.csv")
    sub = df1_imp_wea_soil[~df1_imp_wea_soil["%H Sat"].isnull()] # training set (no missing data in label)
    train, test = datawig.utils.random_split(sub, split_ratios=[0.9, 0.1])
    impute(data_encoder_cols, data_featurizer_cols, label_encoder_cols, 
        out="TEST_%HSat_imputer_model", col_to_imp="%H Sat", 
        train=train, test=test, save="Merged_Trait_Weather_Soil_Data_imputed.csv", 
        test_accuracy=True)
    
    # Back to imputing %H Sat
    print("Imputing %H Sat...")
    train = df1_imp_wea_soil[~df1_imp_wea_soil["%H Sat"].isnull()] # training set (no missing data in label)
    test = df1_imp_wea_soil[df1_imp_wea_soil["%H Sat"].isnull()] # testing set
    imputer = datawig.Imputer(
        data_encoders=data_encoder_cols,
        data_featurizers=data_featurizer_cols,
        label_encoders=label_encoder_cols,
        output_path="df1_imp_wea_soil_%HSat_imputer_model"
    )
    imputer.fit(train_df=train)
    predictions = imputer.predict(test)
    imputer.save() # save model
    test["%H Sat"] = predictions["%H Sat_imputed"].values # replace missing values with predictions
    df1_iws_imputed = pd.concat([train, test])
    df1_iws_imputed.to_csv("Merged_Trait_Weather_Soil_Data_imputed.csv", index=False) # save df1_imp_wea_soil with imputed column

    ############################################################################
    # Impute %K Sat
    ############################################################################
    data_encoder_cols = [NumericalEncoder("Year"), 
        CategoricalEncoder("Field_Location"), 
        NumericalEncoder("QV2M"), 
        NumericalEncoder("T2MDEW"), 
        NumericalEncoder("PS"), 
        NumericalEncoder("ALLSKY_SFC_SW_DWN"), 
        NumericalEncoder("ALLSKY_SFC_PAR_TOT"), 
        NumericalEncoder("T2M_MAX"), 
        NumericalEncoder("T2M_MIN"), 
        NumericalEncoder("T2MWET"), 
        NumericalEncoder("T2M"), 
        NumericalEncoder("Potassium ppm K"), 
        NumericalEncoder("%H Sat"), 
        NumericalEncoder("%Na Sat"), 
        NumericalEncoder("Mehlich P-III ppm P"), 
        NumericalEncoder("% Sand")]
    label_encoder_cols = [NumericalEncoder("%K Sat")]
    data_featurizer_cols = [NumericalFeaturizer("Year"), 
        EmbeddingFeaturizer("Field_Location"), 
        NumericalFeaturizer("QV2M"), 
        NumericalFeaturizer("T2MDEW"), 
        NumericalFeaturizer("PS"), 
        NumericalFeaturizer("ALLSKY_SFC_SW_DWN"), 
        NumericalFeaturizer("ALLSKY_SFC_PAR_TOT"), 
        NumericalFeaturizer("T2M_MAX"), 
        NumericalFeaturizer("T2M_MIN"), 
        NumericalFeaturizer("T2MWET"), 
        NumericalFeaturizer("T2M"), 
        NumericalFeaturizer("Potassium ppm K"), 
        NumericalFeaturizer("%H Sat"), 
        NumericalFeaturizer("%Na Sat"), 
        NumericalFeaturizer("Mehlich P-III ppm P"), 
        NumericalFeaturizer("% Sand")]
    # Check accuracy of imputer
    print("Checking accuracy of imputer for %K Sat...")
    df1_imp_wea_soil = pd.read_csv("Merged_Trait_Weather_Soil_Data_imputed.csv")
    sub = df1_imp_wea_soil[~df1_imp_wea_soil["%K Sat"].isnull()] # training set (no missing data in label)
    train, test = datawig.utils.random_split(sub, split_ratios=[0.9, 0.1])
    impute(data_encoder_cols, data_featurizer_cols, label_encoder_cols, 
        out="TEST_%KSat_imputer_model", col_to_imp="%K Sat", 
        train=train, test=test, save="Merged_Trait_Weather_Soil_Data_imputed.csv", 
        test_accuracy=True)
    
    # Back to imputing %K Sat
    print("Imputing %K Sat...")
    train = df1_imp_wea_soil[~df1_imp_wea_soil["%K Sat"].isnull()] # training set (no missing data in label)
    test = df1_imp_wea_soil[df1_imp_wea_soil["%K Sat"].isnull()] # testing set
    imputer = datawig.Imputer(
        data_encoders=data_encoder_cols,
        data_featurizers=data_featurizer_cols,
        label_encoders=label_encoder_cols,
        output_path="df1_imp_wea_soil_%KSat_imputer_model"
    )
    imputer.fit(train_df=train)
    predictions = imputer.predict(test)
    imputer.save() # save model
    test["%K Sat"] = predictions["%K Sat_imputed"].values # replace missing values with predictions
    df1_iws_imputed = pd.concat([train, test])
    df1_iws_imputed.to_csv("Merged_Trait_Weather_Soil_Data_imputed.csv", index=False) # save df1_imp_wea_soil with imputed column

    ############################################################################
    # Impute %Ca Sat
    ############################################################################
    data_encoder_cols = [NumericalEncoder("Year"), 
        CategoricalEncoder("Field_Location"), 
        NumericalEncoder("Pollen_DAP_days"), 
        NumericalEncoder("Silk_DAP_days"), 
        NumericalEncoder("Twt_kg_m3"), 
        NumericalEncoder("E Depth"), 
        NumericalEncoder("1:1 Soil pH"), 
        NumericalEncoder("WDRF Buffer pH"), 
        NumericalEncoder("1:1 S Salts mmho/cm"), 
        NumericalEncoder("Calcium ppm Ca"), 
        NumericalEncoder("Magnesium ppm Mg"), 
        NumericalEncoder("Sodium ppm Na"), 
        NumericalEncoder("CEC/Sum of Cations me/100g"), 
        NumericalEncoder("% Silt"), 
        NumericalEncoder("% Clay")] # columns related to the label column to impute
    label_encoder_cols = [NumericalEncoder("%Ca Sat")] # column to impute
    data_featurizer_cols = [NumericalFeaturizer("Year"), 
        EmbeddingFeaturizer("Field_Location"), 
        NumericalFeaturizer("Pollen_DAP_days"), 
        NumericalFeaturizer("Silk_DAP_days"), 
        NumericalFeaturizer("Twt_kg_m3"), 
        NumericalFeaturizer("E Depth"), 
        NumericalFeaturizer("1:1 Soil pH"), 
        NumericalFeaturizer("WDRF Buffer pH"), 
        NumericalFeaturizer("1:1 S Salts mmho/cm"), 
        NumericalFeaturizer("Calcium ppm Ca"), 
        NumericalFeaturizer("Magnesium ppm Mg"), 
        NumericalFeaturizer("Sodium ppm Na"), 
        NumericalFeaturizer("CEC/Sum of Cations me/100g"), 
        NumericalFeaturizer("% Silt"), 
        NumericalFeaturizer("% Clay")]

    # Check accuracy of imputer
    print("Checking accuracy of imputer for %Ca Sat...")
    df1_imp_wea_soil = pd.read_csv("Merged_Trait_Weather_Soil_Data_imputed.csv")
    sub = df1_imp_wea_soil[~df1_imp_wea_soil["%Ca Sat"].isnull()] # training set (no missing data in label)
    train, test = datawig.utils.random_split(sub, split_ratios=[0.9, 0.1])
    impute(data_encoder_cols, data_featurizer_cols, label_encoder_cols, 
        out="TEST_%CaSat_imputer_model", col_to_imp="%Ca Sat", 
        train=train, test=test, save="Merged_Trait_Weather_Soil_Data_imputed.csv", 
        test_accuracy=True)
    
    # Back to imputing %Ca Sat
    print("Imputing %Ca Sat...")
    train = df1_imp_wea_soil[~df1_imp_wea_soil["%Ca Sat"].isnull()] # training set (no missing data in label)
    test = df1_imp_wea_soil[df1_imp_wea_soil["%Ca Sat"].isnull()] # testing set
    imputer = datawig.Imputer(
        data_encoders=data_encoder_cols,
        data_featurizers=data_featurizer_cols,
        label_encoders=label_encoder_cols,
        output_path="df1_imp_wea_soil_%CaSat_imputer_model"
    )
    imputer.fit(train_df=train)
    predictions = imputer.predict(test)
    imputer.save() # save model
    test["%Ca Sat"] = predictions["%Ca Sat_imputed"].values # replace missing values with predictions
    df1_iws_imputed = pd.concat([train, test])
    df1_iws_imputed.to_csv("Merged_Trait_Weather_Soil_Data_imputed.csv", index=False) # save df1_imp_wea_soil with imputed column

    ############################################################################
    # Impute %Mg Sat
    ############################################################################
    data_encoder_cols = [NumericalEncoder("Year"), 
        CategoricalEncoder("Field_Location"), 
        NumericalEncoder("Plot_Area_ha"), 
        NumericalEncoder("Plant_Height_cm"), 
        NumericalEncoder("Ear_Height_cm"), 
        NumericalEncoder("Yield_Mg_ha"), 
        NumericalEncoder("Grain_Moisture"), 
        NumericalEncoder("RH2M"), 
        NumericalEncoder("WS2M"), 
        NumericalEncoder("GWETTOP"), 
        NumericalEncoder("GWETROOT"), 
        NumericalEncoder("GWETPROF"), 
        NumericalEncoder("WDRF Buffer pH"), 
        NumericalEncoder("Organic Matter LOI %"), 
        NumericalEncoder("Magnesium ppm Mg"), 
        NumericalEncoder("% Silt"), 
        NumericalEncoder("% Clay")] # columns related to the label column to impute
    label_encoder_cols = [NumericalEncoder("%Mg Sat")] # column to impute
    data_featurizer_cols = [NumericalFeaturizer("Year"), 
        EmbeddingFeaturizer("Field_Location"), 
        NumericalFeaturizer("Plot_Area_ha"), 
        NumericalFeaturizer("Plant_Height_cm"), 
        NumericalFeaturizer("Ear_Height_cm"), 
        NumericalFeaturizer("Yield_Mg_ha"), 
        NumericalFeaturizer("Grain_Moisture"), 
        NumericalFeaturizer("RH2M"), 
        NumericalFeaturizer("WS2M"), 
        NumericalFeaturizer("GWETTOP"), 
        NumericalFeaturizer("GWETROOT"), 
        NumericalFeaturizer("GWETPROF"), 
        NumericalFeaturizer("WDRF Buffer pH"), 
        NumericalFeaturizer("Organic Matter LOI %"), 
        NumericalFeaturizer("Magnesium ppm Mg"), 
        NumericalFeaturizer("% Silt"), 
        NumericalFeaturizer("% Clay")]

    # Check accuracy of imputer
    print("Checking accuracy of imputer for %Mg Sat...")
    df1_imp_wea_soil = pd.read_csv("Merged_Trait_Weather_Soil_Data_imputed.csv")
    sub = df1_imp_wea_soil[~df1_imp_wea_soil["%Mg Sat"].isnull()] # training set (no missing data in label)
    train, test = datawig.utils.random_split(sub, split_ratios=[0.9, 0.1])
    impute(data_encoder_cols, data_featurizer_cols, label_encoder_cols, 
        out="TEST_%MgSat_imputer_model", col_to_imp="%Mg Sat", 
        train=train, test=test, save="Merged_Trait_Weather_Soil_Data_imputed.csv", 
        test_accuracy=True)
    
    # Back to imputing %Mg Sat
    print("Imputing %Mg Sat...")
    train = df1_imp_wea_soil[~df1_imp_wea_soil["%Mg Sat"].isnull()] # training set (no missing data in label)
    test = df1_imp_wea_soil[df1_imp_wea_soil["%Mg Sat"].isnull()] # testing set
    imputer = datawig.Imputer(
        data_encoders=data_encoder_cols,
        data_featurizers=data_featurizer_cols,
        label_encoders=label_encoder_cols,
        output_path="df1_imp_wea_soil_%MgSat_imputer_model"
    )
    imputer.fit(train_df=train)
    predictions = imputer.predict(test)
    imputer.save() # save model
    test["%Mg Sat"] = predictions["%Mg Sat_imputed"].values # replace missing values with predictions
    df1_iws_imputed = pd.concat([train, test])
    df1_iws_imputed.to_csv("Merged_Trait_Weather_Soil_Data_imputed.csv", index=False) # save df1_imp_wea_soil with imputed column

    ############################################################################
    # Impute %Na Sat
    ############################################################################
    data_encoder_cols = [NumericalEncoder("Year"), 
        CategoricalEncoder("Field_Location"), 
        NumericalEncoder("QV2M"), 
        NumericalEncoder("T2MDEW"), 
        NumericalEncoder("ALLSKY_SFC_SW_DWN"), 
        NumericalEncoder("ALLSKY_SFC_PAR_TOT"), 
        NumericalEncoder("T2M_MAX"), 
        NumericalEncoder("T2M_MIN"), 
        NumericalEncoder("T2MWET"), 
        NumericalEncoder("T2M"), 
        NumericalEncoder("PRECTOTCORR"), 
        NumericalEncoder("Year_y"), 
        NumericalEncoder("lbs N/A"), 
        NumericalEncoder("Sulfate-S ppm S"), 
        NumericalEncoder("Sodium ppm Na"), 
        NumericalEncoder("%H Sat"), 
        NumericalEncoder("%K Sat"), 
        NumericalEncoder("Mehlich P-III ppm P"), 
        NumericalEncoder("% Sand")] # columns related to the label column to impute
    label_encoder_cols = [NumericalEncoder("%Na Sat")] # column to impute
    data_featurizer_cols = [NumericalFeaturizer("Year"), 
        EmbeddingFeaturizer("Field_Location"), 
        NumericalFeaturizer("QV2M"), 
        NumericalFeaturizer("T2MDEW"), 
        NumericalFeaturizer("ALLSKY_SFC_SW_DWN"), 
        NumericalFeaturizer("ALLSKY_SFC_PAR_TOT"), 
        NumericalFeaturizer("T2M_MAX"), 
        NumericalFeaturizer("T2M_MIN"), 
        NumericalFeaturizer("T2MWET"), 
        NumericalFeaturizer("T2M"), 
        NumericalFeaturizer("PRECTOTCORR"), 
        NumericalFeaturizer("Year_y"), 
        NumericalFeaturizer("lbs N/A"), 
        NumericalFeaturizer("Sulfate-S ppm S"), 
        NumericalFeaturizer("Sodium ppm Na"), 
        NumericalFeaturizer("%H Sat"), 
        NumericalFeaturizer("%K Sat"), 
        NumericalFeaturizer("Mehlich P-III ppm P"),
        NumericalFeaturizer("% Sand")]

    # Check accuracy of imputer
    print("Checking accuracy of imputer for %Na Sat...")
    df1_imp_wea_soil = pd.read_csv("Merged_Trait_Weather_Soil_Data_imputed.csv")
    sub = df1_imp_wea_soil[~df1_imp_wea_soil["%Na Sat"].isnull()] # training set (no missing data in label)
    train, test = datawig.utils.random_split(sub, split_ratios=[0.9, 0.1])
    impute(data_encoder_cols, data_featurizer_cols, label_encoder_cols, 
        out="TEST_%NaSat_imputer_model", col_to_imp="%Na Sat", 
        train=train, test=test, save="Merged_Trait_Weather_Soil_Data_imputed.csv", 
        test_accuracy=True)
    
    # Back to imputing %Na Sat
    print("Imputing %Na Sat...")
    train = df1_imp_wea_soil[~df1_imp_wea_soil["%Na Sat"].isnull()] # training set (no missing data in label)
    test = df1_imp_wea_soil[df1_imp_wea_soil["%Na Sat"].isnull()] # testing set
    imputer = datawig.Imputer(
        data_encoders=data_encoder_cols,
        data_featurizers=data_featurizer_cols,
        label_encoders=label_encoder_cols,
        output_path="df1_imp_wea_soil_%NaSat_imputer_model"
    )
    imputer.fit(train_df=train)
    predictions = imputer.predict(test)
    imputer.save() # save model
    test["%Na Sat"] = predictions["%Na Sat_imputed"].values # replace missing values with predictions
    df1_iws_imputed = pd.concat([train, test])
    df1_iws_imputed.to_csv("Merged_Trait_Weather_Soil_Data_imputed.csv", index=False) # save df1_imp_wea_soil with imputed column

    ############################################################################
    # Impute % Sand
    ############################################################################
    data_encoder_cols = [NumericalEncoder("Year"), 
        CategoricalEncoder("Field_Location"), 
        NumericalEncoder("QV2M"), 
        NumericalEncoder("T2MDEW"), 
        NumericalEncoder("PS"), 
        NumericalEncoder("ALLSKY_SFC_SW_DWN"), 
        NumericalEncoder("ALLSKY_SFC_PAR_TOT"), 
        NumericalEncoder("T2M_MAX"), 
        NumericalEncoder("T2M_MIN"), 
        NumericalEncoder("T2MWET"), 
        NumericalEncoder("GWETROOT"), 
        NumericalEncoder("T2M"), 
        NumericalEncoder("PRECTOTCORR"), 
        NumericalEncoder("WS2M"), 
        NumericalEncoder("%H Sat"), 
        NumericalEncoder("%K Sat"), 
        NumericalEncoder("%Na Sat"), 
        NumericalEncoder("Mehlich P-III ppm P"), 
        NumericalEncoder("% Clay")] # columns related to the label column to impute
    label_encoder_cols = [NumericalEncoder("% Sand")] # column to impute
    data_featurizer_cols = [NumericalFeaturizer("Year"), 
        EmbeddingFeaturizer("Field_Location"), 
        NumericalFeaturizer("QV2M"), 
        NumericalFeaturizer("T2MDEW"), 
        NumericalFeaturizer("PS"), 
        NumericalFeaturizer("ALLSKY_SFC_SW_DWN"), 
        NumericalFeaturizer("ALLSKY_SFC_PAR_TOT"), 
        NumericalFeaturizer("T2M_MAX"), 
        NumericalFeaturizer("T2M_MIN"), 
        NumericalFeaturizer("T2MWET"), 
        NumericalFeaturizer("GWETROOT"), 
        NumericalFeaturizer("T2M"), 
        NumericalFeaturizer("PRECTOTCORR"), 
        NumericalFeaturizer("WS2M"), 
        NumericalFeaturizer("%H Sat"), 
        NumericalFeaturizer("%K Sat"), 
        NumericalFeaturizer("%Na Sat"), 
        NumericalFeaturizer("Mehlich P-III ppm P"),
        NumericalFeaturizer("% Clay")]

    # Check accuracy of imputer
    print("Checking accuracy of imputer for % Sand...")
    df1_imp_wea_soil = pd.read_csv("Merged_Trait_Weather_Soil_Data_imputed.csv")
    sub = df1_imp_wea_soil[~df1_imp_wea_soil["% Sand"].isnull()] # training set (no missing data in label)
    train, test = datawig.utils.random_split(sub, split_ratios=[0.9, 0.1])
    impute(data_encoder_cols, data_featurizer_cols, label_encoder_cols, 
        out="TEST_%Sand_imputer_model", col_to_imp="% Sand", 
        train=train, test=test, save="Merged_Trait_Weather_Soil_Data_imputed.csv", 
        test_accuracy=True)
    
    # Back to imputing % Sand
    print("Imputing % Sand...")
    train = df1_imp_wea_soil[~df1_imp_wea_soil["% Sand"].isnull()] # training set (no missing data in label)
    test = df1_imp_wea_soil[df1_imp_wea_soil["% Sand"].isnull()] # testing set
    imputer = datawig.Imputer(
        data_encoders=data_encoder_cols,
        data_featurizers=data_featurizer_cols,
        label_encoders=label_encoder_cols,
        output_path="df1_imp_wea_soil_%Sand_imputer_model"
    )
    imputer.fit(train_df=train)
    predictions = imputer.predict(test)
    imputer.save() # save model
    test["% Sand"] = predictions["% Sand_imputed"].values # replace missing values with predictions
    df1_iws_imputed = pd.concat([train, test])
    df1_iws_imputed.to_csv("Merged_Trait_Weather_Soil_Data_imputed.csv", index=False) # save df1_imp_wea_soil with imputed column

    ############################################################################
    # Impute % Silt
    ############################################################################
    data_encoder_cols = [NumericalEncoder("Year"), 
        CategoricalEncoder("Field_Location"), 
        NumericalEncoder("Plot_Area_ha"), 
        NumericalEncoder("Plant_Height_cm"), 
        NumericalEncoder("Ear_Height_cm"), 
        NumericalEncoder("Twt_kg_m3"), 
        NumericalEncoder("WS2M"), 
        NumericalEncoder("E Depth"), 
        NumericalEncoder("1:1 Soil pH"), 
        NumericalEncoder("WDRF Buffer pH"), 
        NumericalEncoder("1:1 S Salts mmho/cm"), 
        NumericalEncoder("Texture No"), 
        NumericalEncoder("Nitrate-N ppm N"), 
        NumericalEncoder("lbs N/A"), 
        NumericalEncoder("Organic Matter LOI %"), 
        NumericalEncoder("Calcium ppm Ca"), 
        NumericalEncoder("Magnesium ppm Mg"), 
        NumericalEncoder("%Ca Sat"), 
        NumericalEncoder("%Mg Sat"), 
        NumericalEncoder("% Clay")] # columns related to the label column to impute
    label_encoder_cols = [NumericalEncoder("% Silt")] # column to impute
    data_featurizer_cols = [NumericalFeaturizer("Year"), 
        EmbeddingFeaturizer("Field_Location"), 
        NumericalFeaturizer("Plot_Area_ha"), 
        NumericalFeaturizer("Plant_Height_cm"), 
        NumericalFeaturizer("Ear_Height_cm"), 
        NumericalFeaturizer("Twt_kg_m3"), 
        NumericalFeaturizer("WS2M"), 
        NumericalFeaturizer("E Depth"), 
        NumericalFeaturizer("1:1 Soil pH"), 
        NumericalFeaturizer("WDRF Buffer pH"), 
        NumericalFeaturizer("1:1 S Salts mmho/cm"), 
        NumericalFeaturizer("Texture No"), 
        NumericalFeaturizer("Nitrate-N ppm N"), 
        NumericalFeaturizer("lbs N/A"), 
        NumericalFeaturizer("Organic Matter LOI %"), 
        NumericalFeaturizer("Calcium ppm Ca"), 
        NumericalFeaturizer("Magnesium ppm Mg"), 
        NumericalFeaturizer("%Ca Sat"), 
        NumericalFeaturizer("%Mg Sat"), 
        NumericalFeaturizer("% Clay")]

    # Check accuracy of imputer
    print("Checking accuracy of imputer for % Silt...")
    df1_imp_wea_soil = pd.read_csv("Merged_Trait_Weather_Soil_Data_imputed.csv")
    sub = df1_imp_wea_soil[~df1_imp_wea_soil["% Silt"].isnull()] # training set (no missing data in label)
    train, test = datawig.utils.random_split(sub, split_ratios=[0.9, 0.1])
    impute(data_encoder_cols, data_featurizer_cols, label_encoder_cols, 
        out="TEST_%Silt_imputer_model", col_to_imp="% Silt", 
        train=train, test=test, save="Merged_Trait_Weather_Soil_Data_imputed.csv", 
        test_accuracy=True)
    
    # Back to imputing % Silt
    print("Imputing % Silt...")
    train = df1_imp_wea_soil[~df1_imp_wea_soil["% Silt"].isnull()] # training set (no missing data in label)
    test = df1_imp_wea_soil[df1_imp_wea_soil["% Silt"].isnull()] # testing set
    imputer = datawig.Imputer(
        data_encoders=data_encoder_cols,
        data_featurizers=data_featurizer_cols,
        label_encoders=label_encoder_cols,
        output_path="df1_imp_wea_soil_%Silt_imputer_model"
    )
    imputer.fit(train_df=train)
    predictions = imputer.predict(test)
    imputer.save() # save model
    test["% Silt"] = predictions["% Silt_imputed"].values # replace missing values with predictions
    df1_iws_imputed = pd.concat([train, test])
    df1_iws_imputed.to_csv("Merged_Trait_Weather_Soil_Data_imputed.csv", index=False) # save df1_imp_wea_soil with imputed column

    ############################################################################
    # Impute % Clay
    ############################################################################
    data_encoder_cols = [NumericalEncoder("Year"), 
        CategoricalEncoder("Field_Location"), 
        NumericalEncoder("Plot_Area_ha"), 
        NumericalEncoder("Stand_count_plants"), 
        NumericalEncoder("Pollen_DAP_days"), 
        NumericalEncoder("Silk_DAP_days"), 
        NumericalEncoder("Plant_Height_cm"), 
        NumericalEncoder("Ear_Height_cm"), 
        NumericalEncoder("WS2M"), 
        NumericalEncoder("1:1 Soil pH"), 
        NumericalEncoder("Organic Matter LOI %"), 
        NumericalEncoder("Potassium ppm K"), 
        NumericalEncoder("Sulfate-S ppm S"), 
        NumericalEncoder("Calcium ppm Ca"), 
        NumericalEncoder("Magnesium ppm Mg"), 
        NumericalEncoder("Sodium ppm Na"), 
        NumericalEncoder("%Ca Sat"), 
        NumericalEncoder("%Mg Sat"), 
        NumericalEncoder("% Silt")] # columns related to the label column to impute
    label_encoder_cols = [NumericalEncoder("% Clay")] # column to impute
    data_featurizer_cols = [NumericalFeaturizer("Year"), 
        EmbeddingFeaturizer("Field_Location"), 
        NumericalFeaturizer("Plot_Area_ha"), 
        NumericalFeaturizer("Stand_count_plants"), 
        NumericalFeaturizer("Pollen_DAP_days"), 
        NumericalFeaturizer("Silk_DAP_days"), 
        NumericalFeaturizer("Plant_Height_cm"), 
        NumericalFeaturizer("Ear_Height_cm"), 
        NumericalFeaturizer("WS2M"), 
        NumericalFeaturizer("1:1 Soil pH"), 
        NumericalFeaturizer("Organic Matter LOI %"), 
        NumericalFeaturizer("Potassium ppm K"), 
        NumericalFeaturizer("Sulfate-S ppm S"), 
        NumericalFeaturizer("Calcium ppm Ca"), 
        NumericalFeaturizer("Magnesium ppm Mg"), 
        NumericalFeaturizer("Sodium ppm Na"), 
        NumericalFeaturizer("%Ca Sat"), 
        NumericalFeaturizer("%Mg Sat"), 
        NumericalFeaturizer("% Silt")]

    # Check accuracy of imputer
    print("Checking accuracy of imputer for % Clay...")
    df1_imp_wea_soil = pd.read_csv("Merged_Trait_Weather_Soil_Data_imputed.csv")
    sub = df1_imp_wea_soil[~df1_imp_wea_soil["% Clay"].isnull()] # training set (no missing data in label)
    train, test = datawig.utils.random_split(sub, split_ratios=[0.9, 0.1])
    impute(data_encoder_cols, data_featurizer_cols, label_encoder_cols, 
        out="TEST_%Clay_imputer_model", col_to_imp="% Clay", 
        train=train, test=test, save="Merged_Trait_Weather_Soil_Data_imputed.csv", 
        test_accuracy=True)
    
    # Back to imputing % Clay
    print("Imputing % Clay...")
    train = df1_imp_wea_soil[~df1_imp_wea_soil["% Clay"].isnull()] # training set (no missing data in label)
    test = df1_imp_wea_soil[df1_imp_wea_soil["% Clay"].isnull()] # testing set
    imputer = datawig.Imputer(
        data_encoders=data_encoder_cols,
        data_featurizers=data_featurizer_cols,
        label_encoders=label_encoder_cols,
        output_path="df1_imp_wea_soil_%Clay_imputer_model"
    )
    imputer.fit(train_df=train)
    predictions = imputer.predict(test)
    imputer.save() # save model
    test["% Clay"] = predictions["% Clay_imputed"].values # replace missing values with predictions
    df1_iws_imputed = pd.concat([train, test])
    df1_iws_imputed.to_csv("Merged_Trait_Weather_Soil_Data_imputed.csv", index=False) # save df1_imp_wea_soil with imputed column
    print(df1_iws_imputed.iloc[:,36:].isna().sum())
    df1_iws_imputed.iloc[:,36:].to_csv("3_Training_Soil_Data_2015_2021_cleaned.csv", index=False)