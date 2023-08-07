import itertools
import h5py
import matplotlib.pyplot as plt
from pennylane import numpy as np
import pandas as pd
import seaborn as sns

"""
FILE READING
"""

jet_filename = '../jetImage_0_30p_0_10000.h5'
f1 = h5py.File(jet_filename,'r')
jets = np.array(f1["jets"])

j_pt = jets[:, 1]
j_mass = jets[:, 3]
j_tau1_b1 = jets[:, 4]
j_tau2_b1 = jets[:, 5]
tau_ratio = j_tau2_b1/j_tau1_b1

num_bins = 16

min_pt = 900
max_pt = 1100
distribution_pt, bins_pt_edge, __ = plt.hist(j_pt, range=[min_pt, max_pt], bins=num_bins)
plt.close("all")
probability_pt = distribution_pt / np.sum(distribution_pt)
bins_pt = np.asarray([(bins_pt_edge[i] + bins_pt_edge[i+1])/2. for i in range(len(bins_pt_edge)-1)])

min_mass = 25
max_mass = 175
distribution_mass, bins_mass_edge, __ =plt.hist(j_mass, range=[min_mass, max_mass],bins=num_bins)
plt.close("all")
probability_mass = distribution_mass / np.sum(distribution_mass)
bins_mass = np.asarray([(bins_mass_edge[i] + bins_mass_edge[i+1])/2. for i in range(len(bins_mass_edge)-1)])

min_tau = 0.0
max_tau = 1.0
distribution_tau, bins_tau_edge, __ = plt.hist(tau_ratio, range=[min_tau, max_tau], bins=num_bins)
plt.close("all")
probability_tau = distribution_tau / np.sum(distribution_tau)
bins_tau = np.asarray([(bins_tau_edge[i] + bins_tau_edge[i+1])/2. for i in range(len(bins_tau_edge)-1)])


###### 2D ########
target_2d, bins_pt_edge, bins_mass_edge, _ = plt.hist2d(j_pt, j_mass, bins=[16,16], range=[[min_pt, max_pt],[min_mass, max_mass]])
bins_pt = np.asarray([(bins_pt_edge[i] + bins_pt_edge[i+1])/2. for i in range(len(bins_pt_edge)-1)])
bins_mass = np.asarray([(bins_mass_edge[i] + bins_mass_edge[i+1])/2. for i in range(len(bins_mass_edge)-1)])
bins_pt_mass=list(itertools.product(bins_pt,bins_mass))
target_2d, bins_pt_edge, bins_mass_edge, _ = plt.hist2d(j_pt, j_mass, bins=[16,16], range=[[min_pt, max_pt],[min_mass, max_mass]])
plt.close('all')
target_2d = target_2d / np.sum(target_2d)
target_1d_2 = np.ravel(target_2d)



##### 3D ########
hist_3_var = [j_pt, j_mass, tau_ratio]
target_3d , [bins_pt_edge,bins_mass_edge,bins_tau_edge] = np.histogramdd(hist_3_var, 
                                   bins=(16,16,16),
                                   range=( [min_pt, max_pt], [min_mass, max_mass], [min_tau, max_tau])
                                  )

bins_pt_mass_nsubj=list(itertools.product(bins_pt,bins_mass,bins_tau))

target_3d = target_3d / np.sum(target_3d)
target_1d_3 = np.ravel(target_3d)
f1.close()




