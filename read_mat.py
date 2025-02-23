# read the matlab files
import numpy as np
import scipy.io
import pandas as pd

def read_probe_mat(probe_locs_path):
    mat = scipy.io.loadmat(probe_locs_path)
    probe_locs = np.array(mat['probe_locs'])
    print(probe_locs)
    return probe_locs
 
def read_borders_table(border_tables_path):
    mat = scipy.io.loadmat(border_tables_path)
    borders_table = pd.DataFrame(mat['borders_table'])
    print(borders_table)
    return borders_table