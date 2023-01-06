"""
Imputing missing values using Datawig

"""

import os
import datatable as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datawig
from datawig import CategoricalEncoder, NumericalEncoder, NumericalFeaturizer, EmbeddingFeaturizer
os.chdir("/mnt/home/seguraab/Shiu_Lab/Collabs/Maize_GxE_Competition_Data/Training_Data")


if __name__ == "__main__":
    df1_imp_wea_soil = pd.read_csv("Merged_Trait_Weather_Soil_Data.csv")