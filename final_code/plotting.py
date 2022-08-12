from itertools import cycle
import h5py
import matplotlib.pyplot as plt
from pennylane import numpy as np
import pennylane as qml
import pandas as pd
import seaborn as sns
import csv
import multiprocessing as mp
import os

from optimizer import *
from file_reading import *
from correlation import *
from multiprocessing_methods import *


folder = '../optimizer_data/2D_run3/'
distributions_folder_name = folder + 'distributions_2D/generation_'
fitnesses_folder_name = folder + 'plots_and_circuits/generation_'
offsprings_folder_name = folder +'offsprings_2D/generation_'
correlations_folder_name = folder + 'correlations/generation_'


def write_header(pop_size):
    header = []
    for i in range(pop_size):
        header.append("circuit " + str(i+1))


    with open(folder + 'fitnesses_2d.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)



def plot_correlations(opt):    
    upper_limit = 500
    iterations = int(opt.pop_size/upper_limit)
    
    for j in range(iterations):
        args = []
        gpu_device = cycle(range(8))
        for i in range(upper_limit):
            circuit = opt.population.individuals[i]
            args.append((opt, circuit, i, str(next(gpu_device))))

        with mp.get_context("spawn").Pool(processes=upper_limit) as pool:
            pool.starmap(plot_correlation, args)

    remainder = opt.pop_size % upper_limit
    args = []
    gpu_device = cycle(range(8))
    for index in range(remainder):
        circuit = opt.population.individuals[index]
        args.append((opt, circuit, index, str(next(gpu_device))))
        
    if remainder > 0:
        with mp.get_context("spawn").Pool(processes=remainder) as pool:
            pool.starmap(plot_correlation, args)

            
def plot_and_draw_everything(opt):
    upper_limit = 500
    iterations = int(opt.pop_size/upper_limit)
    for j in range(iterations):
        args = []
        gpu_device = cycle(range(8))
        for i in range(upper_limit):
            circuit = opt.population.individuals[i]
            args.append((opt, circuit, i, str(next(gpu_device))))

        with mp.get_context("spawn").Pool(processes=upper_limit) as pool:
            pool.starmap(plot_everything, args)
    
    remainder = opt.pop_size % upper_limit
    args = []
    gpu_device = cycle(range(8))
    for i in range(remainder):
        circuit = opt.population.individuals[i]
        args.append((opt, circuit, i, str(next(gpu_device))))
        
    if remainder > 0:
        with mp.get_context("spawn").Pool(processes=remainder) as pool:
            pool.starmap(plot_everything, args)
            
            
            
            
def plot_distribution(opt):
    plt.rcParams['font.size'] = '15'
    with open(folder + 'fitnesses_2d.csv', 'a') as f:
        writer = csv.writer(f)

        fitnesses = []

        for circuit in opt.population.individuals:
            fitnesses.append(circuit.get_fitness())
        writer.writerow(fitnesses)
        
    plt.title("generation " + str(opt.generation))
    plt.xlabel('fitness values')
    plt.ylabel('quantity')
    plt.hist(fitnesses, range=[0.0, 1.0], bins=10, rwidth=0.95)
    if not os.path.exists(folder + 'distributions_2D/'):
        os.makedirs(folder + 'distributions_2D/')
    plt.savefig(distributions_folder_name + str(opt.generation) + '.png', bbox_inches='tight')
    plt.close()
    
    
    
    
def draw_and_save_offsprings(opt):
    upper_limit = 500
    iterations = int(opt.pop_size/upper_limit)
    for j in range(iterations):
        args = []
        gpu_device = cycle(range(8))
        for i in range(upper_limit):
            circuit = opt.offsprings.individuals[i]
            args.append((opt, circuit, i, str(next(gpu_device))))

        with mp.get_context("spawn").Pool(processes=upper_limit) as pool:
            pool.starmap(draw_and_save_offspring, args)
    
    remainder = opt.offsprings.pop_size % upper_limit
    args = []
    gpu_device = cycle(range(8))
    for i in range(remainder):
        circuit = opt.offsprings.individuals[i]
        args.append((opt, circuit, i, str(next(gpu_device))))
        
    if remainder > 0:
        with mp.get_context("spawn").Pool(processes=remainder) as pool:
            pool.starmap(draw_and_save_offspring, args)
    
    
def write_params_and_tensors(opt):
    directory = fitnesses_folder_name  + str(opt.generation)
    if not os.path.exists(directory):
        os.makedirs(directory)
    best_params_file = directory + "/optimal_params.csv"

    with open(best_params_file, 'w') as f:
        writer = csv.writer(f)

        for circuit in opt.population.individuals:
            writer.writerow(circuit.params)
        
    tensors_file = directory + "/tensors.csv"
    with open(tensors_file, 'w') as f:
        writer = csv.writer(f)

        for circuit in opt.population.individuals:
            writer.writerow(circuit.tensor)

    
    
####################################################################################################################
    
    
    
    

def plot_fitnesses(opt):
    
    for i in range(len(opt.population.individuals)):
        plt.figure( figsize=(20, 14))
        plt.title("generation " + str(opt.generation) + " circuit " + str(i+1) + " fitness: " + str(opt.population.individuals[i].get_fitness()))
        plt.plot(opt.population.individuals[i].fitnesses)
        
        plt.xlabel('iterations')
        plt.ylabel('fitnesses')
        directory1 = fitnesses_folder_name + str(opt.generation)
        if not os.path.exists(directory1):
            os.makedirs(directory1)
        plt.scatter(opt.population.individuals[i].best_loss_idx, opt.population.individuals[i].fitnesses[opt.population.individuals[i].best_loss_idx],c='r')
        plt.savefig(directory1 + '/circuit_' + str(i+1) +'_fitness.png', bbox_inches='tight')
        plt.cla()
        plt.close("all")
        

def plot_losses(opt):
    for i in range(len(opt.population.individuals)):
        plt.figure( figsize=(20, 14))
        plt.title("generation " + str(opt.generation) + " circuit " + str(i+1) + " losses: " + str(opt.population.individuals[i].losses[opt.population.individuals[i].best_loss_idx]))
        plt.plot(opt.population.individuals[i].losses)
        plt.xlabel('iterations')
        plt.ylabel('losses')
        directory1 = fitnesses_folder_name + str(opt.generation)
        if not os.path.exists(directory1):
            os.makedirs(directory1)
        plt.scatter(opt.population.individuals[i].best_loss_idx, opt.population.individuals[i].losses[opt.population.individuals[i].best_loss_idx],c='r')
        plt.savefig(directory1 + '/circuit_' + str(i+1) +'_losses.png', bbox_inches='tight')
        plt.cla()
        plt.close("all")
        
        
def draw_circuits(opt): 
    for i in range(len(opt.population.individuals)):
        circuit = opt.population.individuals[i]
        dev = qml.device('qulacs.simulator', wires=circuit.dev_wires, gpu=True, shots=10000)
        qnode = qml.QNode(convert_tensor_to_circuit, dev)

        fig, ax = qml.draw_mpl(qnode)(circuit.tensor, circuit.params, circuit.n_qubits, circuit.num_one_gates, circuit.num_two_gates, circuit.dev_wires)
        title = "generation " + str(opt.generation) +  " circuit " + str(i+1) + " fitness: " + str(circuit.get_fitness())
        ax.set_title(title)
        directory = fitnesses_folder_name + str(opt.generation)
        if not os.path.exists(directory):
            os.makedirs(directory)
        fig.savefig(directory + '/circuit_' + str(i+1) +'_circuit.png', bbox_inches='tight')
        plt.close("all")
        
        
        
        
def plot_comparisons(opt, title = ''):
    for i in range(len(opt.population.individuals)):
        circuit = opt.population.individuals[i]
        
        dev = qml.device('qulacs.simulator', wires=circuit.dev_wires, gpu=True, shots=10000)
        qnode = qml.QNode(convert_tensor_to_circuit, dev)

        h1dgen = qnode(circuit.tensor, circuit.params, circuit.n_qubits, circuit.num_one_gates, circuit.num_two_gates, circuit.dev_wires)
        h2dgen = h1dgen.reshape((16,16))
        plt.figure( figsize=(20,20))
        h1gen = np.sum(h2dgen, axis=1)
        h1gen = h1gen / np.sum(h1gen)
        h2gen = np.sum(h2dgen, axis=0)
        h2gen = h2gen / np.sum(h2gen)
        plt.suptitle(title)
        plt.subplot(221)
        plt.bar(bins_pt,probability_pt, label='target', width=5, align='edge')
        plt.bar(bins_pt, h1gen, label='generated',width=-5, align='edge')
        plt.xlabel('Jet pT [GeV]')
        plt.legend()
        plt.subplot(222)
        plt.bar(bins_mass,probability_mass , label='target', width=4.5, align='edge')
        plt.bar(bins_mass, h2gen, label='generated',width=-4.5, align='edge')
        plt.xlabel('Jet mass [GeV]')
        plt.legend()
        plt.subplot(223)
        plt.imshow(np.flip(target_2d.transpose(), axis=0),
                   extent=[ bins_pt_edge[0], bins_pt_edge[-1],bins_mass_edge[0],bins_mass_edge[-1]],
                   aspect='auto')
        plt.xlabel('Jet pT [GeV]')
        plt.ylabel('Jet mass [GeV]')
        plt.title('target')
        plt.subplot(224)
        plt.imshow(np.flip(h2dgen.transpose(),axis=0),
                   extent=[ bins_pt_edge[0], bins_pt_edge[-1],bins_mass_edge[0],bins_mass_edge[-1]],
                   aspect='auto')
        plt.xlabel('Jet pT [GeV]')
        plt.ylabel('Jet mass [GeV]')
        plt.title('generated')
        directory = fitnesses_folder_name  + str(opt.generation)
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(directory + '/circuit_' + str(i+1) +'_comparisons.png', bbox_inches='tight')
        plt.close("all")
    
        
        

def write_params_and_tensors(opt):
    directory = fitnesses_folder_name  + str(opt.generation)
    if not os.path.exists(directory):
        os.makedirs(directory)
    best_params_file = directory + "/optimal_params.csv"

    with open(best_params_file, 'w') as f:
        writer = csv.writer(f)

        for circuit in opt.population.individuals:
            writer.writerow(circuit.params)
        
    tensors_file = directory + "/tensors.csv"
    with open(tensors_file, 'w') as f:
        writer = csv.writer(f)

        for circuit in opt.population.individuals:
            writer.writerow(circuit.tensor)