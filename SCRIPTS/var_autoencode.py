## apply VAE to VCF data
## input: "DATA/VCF/{chrom}/{section}_haplo.vcf"
## output: "DATA/ENCODED/{chrom}/{section}_haplo_var_encoded.csv", model?

import glob
import re
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras.callbacks as kcb
# import os

## import custom scripts (the ones form github)
import tfmodels

## get file paths
inp_data = sys.argv[1]
out_data_enc = sys.argv[2]
out_data_dec = sys.argv[3]
out_model_enc = sys.argv[4]
out_model_dec = sys.argv[5]
code_ratio = 100

## read input as data frame and create sub dfs for train and test data , 
df_data = pd.read_csv(inp_data, index_col=0, sep=';').T.astype(dtype=np.float32)
df_rand = df_data.sample(frac=1)
train_index = int(0.8*len(df_data))
df_train = df_rand[0:train_index]
df_test = df_rand[train_index:]

x_test_t = tf.convert_to_tensor(df_test)
x_train_t = tf.convert_to_tensor(df_train)

## create and train model
my_auto, my_enc, my_dec = tfmodels.variational_autoencoder(df_data.shape[1], code_ratio)

my_auto.compile(optimizer='adam', metrics=[tf.keras.metrics.BinaryAccuracy()])

history = my_auto.fit( x_train_t, x_train_t, epochs=1000, shuffle=True, validation_data=(x_test_t, x_test_t), ## need to check test data, it's weird
                 callbacks=kcb.EarlyStopping(monitor="val_loss", patience=30, restore_best_weights=True) )

## find epoch at which model found best losses
bestiter = history.history["val_loss"].index(min(history.history["val_loss"]))
print(bestiter)
print(inp_data+"\t"+str(code_ratio)+"\t"
      +str(history.history["loss"][bestiter])+"\t"
      +str(history.history["val_loss"][bestiter]))

## save encoder and decoder
my_enc.save(out_model_enc)
my_dec.save(out_model_dec)

## output data 
encoded_data = my_enc.predict(df_data)
np.savetxt(out_data_enc + "mean.csv", encoded_data[0], delimiter=";")
np.savetxt(out_data_enc + "logvar.csv", encoded_data[1], delimiter=";")
np.savetxt(out_data_enc + "sampling.csv", encoded_data[2], delimiter=";")
df_decoded_mean = pd.DataFrame(my_dec.predict(encoded_data[0]), columns=df_data.columns, index=df_data.index).T
df_decoded_mean.to_csv(out_data_dec + "mean.csv", sep=";")
df_decoded_sampling = pd.DataFrame(my_dec.predict(encoded_data[2]), columns=df_data.columns, index=df_data.index).T
df_decoded_sampling.to_csv(out_data_dec + "sampling.csv", sep=";")


