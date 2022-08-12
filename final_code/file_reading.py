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


num_bins = 16

min_pt = 900
max_pt = 1100
distribution_pt, bins_pt_edge, __ = plt.hist(j_pt, range=[min_pt, max_pt], bins=num_bins)
probability_pt = distribution_pt / np.sum(distribution_pt)
bins_pt = np.asarray([(bins_pt_edge[i] + bins_pt_edge[i+1])/2. for i in range(len(bins_pt_edge)-1)])

min_mass = 25
max_mass = 175
distribution_mass, bins_mass_edge, __ =plt.hist(j_mass, range=[min_mass, max_mass],bins=num_bins)
probability_mass = distribution_mass / np.sum(distribution_mass)
bins_mass = np.asarray([(bins_mass_edge[i] + bins_mass_edge[i+1])/2. for i in range(len(bins_mass_edge)-1)])

target_2d, bins_pt_edge, bins_mass_edge, _ = plt.hist2d(j_pt, j_mass, bins=[16,16], range=[[min_pt, max_pt],[min_mass, max_mass]])
plt.xlabel('Jet pT [GeV]')
plt.ylabel('Jet mass [GeV]')
bins_pt = np.asarray([(bins_pt_edge[i] + bins_pt_edge[i+1])/2. for i in range(len(bins_pt_edge)-1)])
bins_mass = np.asarray([(bins_mass_edge[i] + bins_mass_edge[i+1])/2. for i in range(len(bins_mass_edge)-1)])
bins_pt_mass=list(itertools.product(bins_pt,bins_mass))
plt.title("2D Data between Jet Mass and Jet pT")
plt.savefig("../target_data/mass_pt.png")
plt.close('all')



shortened_j_pt = j_pt[(min_pt < j_pt) & (max_pt > j_pt)]
shortened_j_mass = j_mass[(min_mass < j_mass) & (max_mass > j_mass)][:len(shortened_j_pt)]

generated_df_init = pd.DataFrame(shortened_j_pt, columns=['jet_pt'])
generated_df_init['jet_mass'] = shortened_j_mass


g=sns.PairGrid(generated_df_init)
g.map_diag(sns.histplot,color='orange')
g.map_upper(sns.scatterplot,color='orange')
g.map_lower(sns.kdeplot,color='orange')
g.fig.suptitle("Target Correlation")
g.fig.savefig("../target_data/target_mass_pt_correlation.png") 
plt.close('all')

target_2d = target_2d / np.sum(target_2d)
target_1d = np.ravel(target_2d)