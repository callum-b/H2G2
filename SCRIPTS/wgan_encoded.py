## apply WGEN to encoded VCF data
## input: a whole bunch of CSV files
## output: simulated CSV files (*_haplo_wgan_generated.csv), model weights in h5 format

import glob
import re
import sys
import os
import numpy as np
from pathlib import Path


## import custom scripts (the ones form github)
import tfmodels

## get training params then input files, first is model wieghts output path, then list of input csvs
train_steps = int(sys.argv[1])
train_save_steps = int(sys.argv[2])
train_check_steps = int(sys.argv[3])
output_samples = int(sys.argv[4])
out_model_weights_path = sys.argv[5]
input_csvs = sys.argv[6:]

out_model_checkpoints_dir = "/".join(out_model_weights_path.split("/")[0:-1]) + "/"
if not os.path.isdir(out_model_checkpoints_dir):
    Path(out_model_checkpoints_dir).mkdir(parents=True, exist_ok=True)
if not os.path.isdir(out_model_checkpoints_dir.replace("MODELS", "GENERATED")):
    Path(out_model_checkpoints_dir.replace("MODELS", "GENERATED")).mkdir(parents=True, exist_ok=True)
load=""
if os.path.isfile(out_model_weights_path + "WGAN.weights.h5"):
    print("Previous save file found in " + out_model_weights_path + ", weights will be loaded from that dir.")
    load = out_model_weights_path
    out_model_checkpoints_dir = out_model_weights_path



## read input files and concatenate them into a single array (rows are samples, columns are variables to concat)
arrays = []
lengths = []
for filepath in input_csvs:
    arrays.append(np.loadtxt(filepath, dtype=np.float64, delimiter=";"))
    lengths.append(arrays[-1].shape[1])

x_train = np.concatenate(arrays, axis=1)


## create wgan model
my_wgan, my_crit, my_gen = tfmodels.wgan(x_train)
print("Model created, preparing for training...")

## train it using custom loop in tfmodels
tfmodels.train_wgan(my_wgan, my_crit, my_gen, x_train, out_model_checkpoints_dir, n_steps=train_steps, save=train_save_steps, check=train_check_steps, load=load, verbose=True) # check=50 doesn't crash, but 100 does

## save model weights for loading
if load:
    my_wgan.save_weights(out_model_weights_path + "WGAN.weights.h5")
else:
    my_wgan.save_weights(out_model_weights_path)

## generate 1000 (default) new samples
x_sim = tfmodels.generate_fake_samples(my_gen, 10000, output_samples)[0] ## 10000 is the number of latent dimensions

for mylength in lengths:
    to_output = x_sim[:, 0:mylength]
    x_sim = np.delete(x_sim, list(range(0,mylength)), axis=1)
    output_path = input_csvs.pop(0).replace("ENCODED", "GENERATED")
    output_path = output_path.split(".")[0] + "_wgan_gen.csv"
    np.savetxt(output_path, to_output, delimiter="\t")
    