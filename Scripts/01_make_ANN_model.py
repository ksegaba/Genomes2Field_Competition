#!/usr/bin/env python3
import tensorflow as tf
import pandas as pd
from tensorflow import keras
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import r2_score
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import r2_score
import eli5
from eli5.sklearn import PermutationImportance
import argparse
tf.__version__
keras.__version__
######
parser = argparse.ArgumentParser(description='This is a code to run ANN model')
parser.add_argument('-tr', '--tr_file',required=True, help="Training_dataset")
parser.add_argument('-vl', '--vl_file', required=True, help="Validation_data")
parser.add_argument('-lb', '--lb_name', required=True, help="Validation_data")
#
args = parser.parse_args()
###############
#input args
train = args.tr_file  
validation = args.vl_file
lable_class = args.lb_name
#loading df
df_train = pd.read_csv(train,sep =",",header =0,index_col =0)
df_validation = pd.read_csv(validation,sep =",",header =0,index_col =0)
validation_intances = df_validation.index.array
#output_file
out = open("%s_results.txt"%(validation.replace(".csv","")), "w")
#out = open(f"{validation.replace('.csv','')}_results.txt","w")

#traning and testing sets

df_train_X = df_train.drop(columns=[lable_class]).values
df_train_y = df_train[lable_class].values.astype(np.float32)
df_validation_X = df_validation.drop(columns=[lable_class]).values
df_validation_y = df_validation[lable_class].values.astype(np.float32)

# create ANN model
model = Sequential()
# Defining the Input layer and FIRST hidden layer, both are same!
model.add(Dense(units=10, input_dim=df_train_X.shape[1:][0], kernel_initializer='normal', activation='relu'))
# Defining the Second layer of the model
# after the first layer we don't have to specify input_dim as keras configure it automatically
model.add(Dense(units=10, kernel_initializer='normal', activation='relu'))
# The output neuron is a single fully connected node 
# Since we will be predicting a single number
model.add(Dense(1, kernel_initializer='normal'))
# Compiling the model
model.compile(loss='mean_squared_error', optimizer='adam')
# Fitting the ANN to the Training set
#model.fit(df_train_X, df_train_y ,batch_size = 20, epochs = 50, verbose=1)

#parameter tuning
def make_regression_ann(Optimizer_trial):
    from keras.models import Sequential
    from keras.layers import Dense
    
    model = Sequential()
    model.add(Dense(units=30, input_dim=df_train_X.shape[1], kernel_initializer='normal', activation='relu'))
    model.add(Dense(units=30, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer=Optimizer_trial)
    return model
 
###########################################
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasRegressor
 
# Listing all the parameters to try
Parameter_Trials={'batch_size':[30,40,50],
                      'epochs':[30,40,50],
                    'Optimizer_trial':['Adam', 'Adamax', 'Nadam'],
                 }
 
# Creating the regression ANN model
RegModel=KerasRegressor(make_regression_ann, verbose=0)

###########################################
from sklearn.metrics import make_scorer
 
# Defining a custom function to calculate accuracy
def Accuracy_Score(orig,pred):
    MAPE = np.mean(100 * (np.abs(orig-pred)/orig))
    print('#'*70,'Accuracy:', 100-MAPE)
    return(100-MAPE)
 
custom_Scoring=make_scorer(Accuracy_Score, greater_is_better=True)
#########################################
# Creating the Grid search space
# See different scoring methods by using sklearn.metrics.SCORERS.keys()
grid_search=GridSearchCV(estimator=RegModel, 
                         param_grid=Parameter_Trials, 
                         scoring=custom_Scoring)
 
#########################################
# Measuring how much time it took to find the best params
import time
StartTime=time.time()
 
# Running Grid Search for different paramenters
grid_search.fit(df_train_X, df_train_y, verbose=1)

EndTime=time.time()
out.write("########## Total Time Taken for grid: %f Minutes\n"%(round(EndTime-StartTime)/60))
#out.write(f"########## Total Time Taken for grid: {round(EndTime-StartTime)/60} Minutes\n")

#using best parameters to fit the model
model.compile(loss="mean_squared_error" ,optimizer =grid_search.best_params_["Optimizer_trial"] )
model.fit(df_train_X, df_train_y ,batch_size = grid_search.best_params_["batch_size"], epochs = grid_search.best_params_["epochs"], verbose=0)

# Generating Predictions on testing data
Predictions=model.predict(df_validation_X)

#generate R2 values
r2 = r2_score( df_validation_y, Predictions )
out.write("r2 is %f"%r2)
#out.write(f"r2 is {r2}")
#saving_pred_vals
dataset = pd.DataFrame({'genotype': validation_intances, 'predicted_yeild': pd.Series(Predictions.T[0])}, columns=['genotype', 'predicted_yeild'])
dataset.to_csv('%s_predicted_vals.csv'%(validation.replace(".csv","")),sep = ",",index = False)
#dataset.to_csv('%s_predicted_vals.csv'%(validation.replace(".csv","")),sep = ",",index = False)
#saving_model
model.save('%s_model'%(validation.replace(".csv","")))
out.close()
#outputting premuation importance scores
perm = PermutationImportance(model,scoring="r2", random_state=1).fit(df_train_X, df_train_y )
importance_df = eli5.show_weights(perm, feature_names = df_train.drop(columns=[lable_class]).columns.tolist())
df_fi = pd.DataFrame(dict(feature_names=df_train.drop(columns=[lable_class]).columns.tolist(),
                          feat_imp=perm.feature_importances_, 
                          std=perm.feature_importances_std_,
                          ))
df_fi.sort_values('feat_imp', ascending=False)
df_fi.to_csv('%s_premutation_importance.csv'%(train.replace(".csv","")),sep = ",",index = False)



