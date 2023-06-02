### TENSORFLOW MODELS

"""
  This is where the most important figures of the project are created.
  This does not contain *all* the figure plotting scripts, as a lot of them are just boxplots or lineplot of data and their associated statistics.
"""


## imports
import numpy as np
import pandas as pd
import re
import csv
import math
import matplotlib.pyplot as plt

def scatter_diagdistance(x, y) :
    """
    Create a scatter plot of two vectors of data, return the figure and the distance values
    """
    z = abs(y-x)
    fig, ax = plt.subplots(dpi=200)
    ax2 = ax.twinx() ##Create secondary axis
    ax2.set_yticks([]) ##No ticks for the secondary axis
    sc = ax.scatter(x, y, c=z, s=50, edgecolor='none')
    ax2.set_ylabel('Distance from diagonal') ##Label for secondary axis
    ax.plot([0, 1], [0, 1], '-', c="red", transform=ax.transAxes) #Line from 0 to 1
    fig.colorbar(sc)
    ax.set_xlabel('Ref mutation frequencies')
    ax.set_ylabel('Decoded mutation frequencies')
    return fig, z

def read_mutcount(in_f:str, sep=';', index=True, header=True):
    """
    Parse a haplotypic vcf file and count the number of occurrences of each mutation
    Returns a 1D array of the number of occurrences for each mutation, and the number of samples in the file.
    """
    if index and header:
        data = pd.read_csv(in_f, sep=sep, index_col=0, dtype=np.float64)
    elif header:
        data = pd.read_csv(in_f, sep=sep, dtype=np.float64)
    elif index:
        data = pd.read_csv(in_f, sep=sep, header=None, index_col=0, dtype=np.float64)
    else:
        data = pd.read_csv(in_f, sep=sep, header=None, dtype=np.float64)

    return data.sum(axis=0), data.shape[1]