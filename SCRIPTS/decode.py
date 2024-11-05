
import sys
import math
import pandas as pd
import numpy as np
import tensorflow as tf


decoder = tf.keras.models.load_model(sys.argv[1])
decoded = np.transpose(decoder.predict(np.loadtxt(sys.argv[2])))

ref = pd.read_csv(sys.argv[3], sep=";", index_col=0)
if not len(ref.columns) == decoded.shape[1]:
    mystrlen = math.ceil(math.log10(decoded.shape[1]))+1
    names = ["H2G2_"+str(x+1).zfill(mystrlen) for x in range(decoded.shape[1])]
else:
    names = ref.columns

pd.DataFrame(decoded, index=ref.index, columns=names).to_csv(sys.argv[4], sep=";")