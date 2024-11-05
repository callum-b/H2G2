## apply regular autoencoder to VCF data
## input: "DATA/VCF/{chrom}/{section}_haplo.vcf"
## output: "DATA/ENCODED/{chrom}/{section}_haplo_encoded.csv", model?

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
pruning = False

## read input as data frame and create sub dfs for train and test data , 
df_data = pd.read_csv(inp_data, index_col=0, sep=';').T.astype(dtype=np.float64)
df_rand = df_data.sample(frac=1)
train_index = int(0.8*len(df_data))
df_train = df_rand[0:train_index]
df_test = df_rand[train_index:]


## create and train model
my_auto, my_enc, my_dec = tfmodels.autoencoder(df_data.shape[1], code_ratio)

my_auto.compile(optimizer='adam', loss='binary_crossentropy', metrics=[tf.keras.metrics.BinaryAccuracy()])

history = my_auto.fit( df_train, df_train, epochs=100, shuffle=True, validation_data=(df_test, df_test), 
                 callbacks=kcb.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True) )

## find epoch at which model found best losses
bestiter = history.history["val_loss"].index(min(history.history["val_loss"]))
print(bestiter)
print(inp_data+"\t"+str(code_ratio)+"\t"
      +str(history.history["loss"][bestiter])+"\t"+str(history.history["binary_accuracy"][bestiter])+"\t"
      +str(history.history["val_loss"][bestiter])+"\t"+str(history.history["val_binary_accuracy"][bestiter]))

## save encoder and decoder
my_enc.save(out_model_enc)
my_dec.save(out_model_dec)

## output data 
encoded_data = my_enc.predict(df_data)
np.savetxt(out_data_enc, encoded_data, delimiter=";")
decoded_data = my_dec.predict(encoded_data)
df_decoded_data = pd.DataFrame(decoded_data, columns=df_data.columns, index=df_data.index).T
df_decoded_data.to_csv(out_data_dec, sep=";")


### PRUNING AND SURGERY ###

if pruning:
    import math
    ## requires Pix2Pix_Auto_Prune
    import kerassurgeon
    from kerassurgeon import identify
    from kerassurgeon.operations import delete_layer, insert_layer, delete_channels
    import tensorflow_model_optimization as tfmot
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Dense, Input, Embedding, Flatten, Reshape, Dropout
    from tensorflow.keras.regularizers import l1
    from tensorflow.keras.optimizers import Adam

    input_size = df_data.shape[1]
    ratios = np.geomspace(1,code_ratio,5)
    code_size = math.floor(input_size/code_ratio)
    hidden_1_size = math.floor(input_size/ratios[1])
    hidden_2_size = math.floor(input_size/ratios[2])
    hidden_3_size = math.floor(input_size/ratios[3])
    
    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.10,
                                                                final_sparsity=0.50,
                                                                begin_step=0,
                                                                end_step=10)
    }

    model_for_pruning = prune_low_magnitude(my_auto, **pruning_params)

    # `prune_low_magnitude` requires a recompile.
    model_for_pruning.compile(optimizer='adam',
                loss=tf.keras.losses.BinaryCrossentropy(),
                metrics=[tf.keras.metrics.BinaryAccuracy()])


    callbacks = [
        tfmot.sparsity.keras.UpdatePruningStep(),
        tfmot.sparsity.keras.PruningSummaries(log_dir="/".join(out_data_enc.split("/")[0:-1])),
        kcb.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ]

    history_pruning = model_for_pruning.fit(df_train, df_train, epochs=100, 
                        shuffle=True, validation_data=(df_test, df_test),
                        callbacks=callbacks)


    bestiter_pruning = history_pruning.history["val_loss"].index(min(history_pruning.history["val_loss"]))
    print(bestiter_pruning)
    print(inp_data+"\t"+str(code_size)+"\t"
        +str(history_pruning.history["loss"][bestiter_pruning])+"\t"+str(history_pruning.history["binary_accuracy"][bestiter_pruning])+"\t"
        +str(history_pruning.history["val_loss"][bestiter_pruning])+"\t"+str(history_pruning.history["val_binary_accuracy"][bestiter_pruning]))



    encoded_data = my_enc.predict(df_train)
    zeroes = list()
    for dim in range(encoded_data.shape[1]):
        if np.std(encoded_data[:,dim]) == 0:
            zeroes.append(dim)
    print(zeroes)

    smaller_encoded_data = np.delete(encoded_data, zeroes, 1)

    encoded_val_data = my_enc.predict(df_test)
    smaller_encoded_val_data = np.delete(encoded_val_data, zeroes, 1)


    smaller_encoded_inputs = Input(shape=(code_size-len(zeroes),))

    smaller_hidden_3_rev = Dense(hidden_2_size, activation='relu', name='hidden_3_rev')(smaller_encoded_inputs) ### NEED TO FIX THESE SIZES HERE, GET THEM FROM DECODER MODEL?
    smaller_d_drop_1 = Dropout(0.2, name='d_drop_1')(smaller_hidden_3_rev)
    smaller_hidden_2_rev = Dense(hidden_2_size, activation='relu', name='hidden_2_rev')(smaller_d_drop_1)
    smaller_d_drop_2 = Dropout(0.2, name='d_drop_2')(smaller_hidden_2_rev)
    smaller_hidden_1_rev = Dense(hidden_1_size, activation='relu', name='hidden_1_rev')(smaller_d_drop_2)
    smaller_output_data = Dense(input_size, activation='relu', name='output_data')(smaller_hidden_1_rev)

    smaller_decoder = Model(smaller_encoded_inputs, smaller_output_data)

    smaller_decoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=[tf.keras.metrics.BinaryAccuracy()])

    history_smaller_decoder = smaller_decoder.fit( smaller_encoded_data, df_train, epochs=100, shuffle=True, validation_data=(smaller_encoded_val_data, df_test), 
                    callbacks=kcb.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True) )


    bestiter = history_smaller_decoder.history["val_loss"].index(min(history_smaller_decoder.history["val_loss"]))
    print(bestiter)
    print(inp_data+"\t"+str(code_size-len(zeroes))+"\t"
        +str(history_smaller_decoder.history["loss"][bestiter])+"\t"+str(history_smaller_decoder.history["binary_accuracy"][bestiter])+"\t"
        +str(history_smaller_decoder.history["val_loss"][bestiter])+"\t"+str(history_smaller_decoder.history["val_binary_accuracy"][bestiter]))



