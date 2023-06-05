from tensorflow import keras
import sys
import pandas as pd
import numpy as np
mod = sys.argv[1]
test = sys.argv[2] 
model = keras.models.load_model(mod)
df_test = pd.read_csv(test,sep =",",header =0,index_col =0)
test_index = df_test.index.array
Predictions=model.predict(df_test.values.astype(np.float32))
#saving_pred_vals
dataset = pd.DataFrame({'genotype': test_index, 'predicted_yeild': pd.Series(Predictions.T[0])}, columns=['genotype', 'predicted_yeild'])
dataset.to_csv(f'{test.replace(".csv","")}_TEST_predicted_vals.csv',sep = ",",index = False)


