from ansatz_optimizer import *
import os

import pandas as pd
import matplotlib.pyplot as plt
from pennylane import numpy as np
import h5py
import pennylane as qml
import random
import time
import csv

jet_filename = '../jetImage_0_30p_0_10000.h5'
f1 = h5py.File(jet_filename,'r')
jets = np.array(f1["jets"])

j_pt = jets[:, 1]
j_eta = jets[:, 2]
j_mass = jets[:, 3]

num_bins = 16

min_pt = 900
max_pt = 1100
distribution_pt, bins_pt_edge, __ = plt.hist(j_pt, range=[min_pt, max_pt], bins=num_bins)
probability_pt = distribution_pt / np.sum(distribution_pt)
bins_pt = np.asarray([(bins_pt_edge[i] + bins_pt_edge[i+1])/2. for i in range(len(bins_pt_edge)-1)])

min_eta = -2
max_eta = 2
distribution_eta, bins_eta_edge, __ =plt.hist(j_eta, range=[min_eta, max_eta], bins=num_bins)
probability_eta = distribution_eta / np.sum(distribution_eta)
bins_eta = np.asarray([(bins_eta_edge[i] + bins_eta_edge[i+1])/2. for i in range(len(bins_eta_edge)-1)])

min_mass = 25
max_mass = 175
distribution_mass, bins_mass_edge, __ =plt.hist(j_mass, range=[min_mass, max_mass],bins=num_bins)
probability_mass = distribution_mass / np.sum(distribution_mass)
bins_mass = np.asarray([(bins_mass_edge[i] + bins_mass_edge[i+1])/2. for i in range(len(bins_mass_edge)-1)])


min_mass = 25
max_mass = 175
distribution_mass, bins_mass_edge, __ =plt.hist(j_mass, range=[min_mass, max_mass],bins=num_bins)
probability_mass = distribution_mass / np.sum(distribution_mass)
bins_mass = np.asarray([(bins_mass_edge[i] + bins_mass_edge[i+1])/2. for i in range(len(bins_mass_edge)-1)])

opt = CircuitOptimizer(pop_size, probability_pt, num_one_gates, num_two_gates, max_moments, n_qubits)

header = []
for i in range(pop_size):
    header.append("circuit " + str(i+1))
print(header)

with open('../optimizer_data/fitnesses_1d.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    
    fitnesses = []
    
    for circuit in opt.population.individuals:
        fitnesses.append(circuit.get_fitness())
    writer.writerow(fitnesses)

distributions_folder_name = '../optimizer_data/distributions_1D_fixedLR/generation_'

def plot_distribution():
    with open('../optimizer_data/fitnesses_1d.csv', 'a') as f:
        writer = csv.writer(f)

        fitnesses = []

        for circuit in opt.population.individuals:
            fitnesses.append(circuit.get_fitness())
        writer.writerow(fitnesses)
        
    plt.title("generation " + str(opt.generation))
    plt.xlabel('fitness values')
    plt.ylabel('quantity')
    plt.hist(fitnesses, range=[0.0, 1.0], bins=10, rwidth=0.95)
    plt.savefig(distributions_folder_name + str(opt.generation) + '.png', bbox_inches='tight')
    # plt.show()

fitnesses_folder_name = '../optimizer_data/fitnesses_1D_fixedLR/generation_'
circuits_folder_name = '../optimizer_data/circuits_1D_fixedLR/generation_'


def plot_fitnesses():
    for i in range(len(opt.population.individuals)):
        plt.title("circuit " + str(i+1))
        plt.plot(opt.population.individuals[i].fitnesses)
        plt.xlabel('iterations')
        plt.ylabel('fitnesses')
        directory1 = fitnesses_folder_name + str(opt.generation)
        if not os.path.exists(directory1):
            os.makedirs(directory1)
        plt.scatter(opt.population.individuals[i].best_loss_idx, opt.population.individuals[i].fitnesses[opt.population.individuals[i].best_loss_idx],c='r')
        plt.savefig(fitnesses_folder_name + str(opt.generation) + '/circuit_' + str(i+1) +'.png', bbox_inches='tight')
        # plt.show()
        
        
def draw_circuits(): 
    for i in range(len(opt.population.individuals)):
        circuit = opt.population.individuals[i]
        fig, ax = qml.draw_mpl(convert_tensor_to_circuit)(circuit.tensor, circuit.params, circuit.n_qubits, circuit.num_one_gates, circuit.num_two_gates)
        title = "generation " + str(opt.generation) +  " circuit " + str(i+1)
        ax.set_title(title)
        directory2 = circuits_folder_name + str(opt.generation)
        if not os.path.exists(directory2):
            os.makedirs(directory2)
        fig.savefig(directory2 + '/circuit_' + str(i+1) +'.png', bbox_inches='tight')
        # fig.show()

        
offsprings_folder_name = '../optimizer_data/offsprings_1D_fixedLR/generation_'


def draw_and_save_offsprings():
    for i in range(len(opt.offsprings.individuals)):
        title = "offspring " + str(i+1) + " generation " + str(opt.generation) + " fitness: " + str(opt.offsprings.individuals[i].get_fitness())
        circuit = opt.offsprings.individuals[i]
        fig, ax = qml.draw_mpl(convert_tensor_to_circuit)(circuit.tensor, circuit.params, circuit.n_qubits, circuit.num_one_gates, circuit.num_two_gates)
        ax.set_title(title)
        directory = offsprings_folder_name + str(opt.generation)
        if not os.path.exists(directory):
            os.makedirs(directory)
        fig.savefig(directory + '/offspring_' + str(i+1) +'.png', bbox_inches='tight')
    
plot_distribution()
draw_circuits()
plot_distribution()