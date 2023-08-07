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
from file_reading_general import *
from plotting_multiprocessing import *
from correlation_plotting import *
from dimension_setting import *


folder = '../more_gates/' + str(dim)+'d/'
distributions_folder_name = folder + 'distributions/generation_'
fitnesses_folder_name = folder + 'plots_and_circuits/generation_'
offsprings_folder_name = folder +'offsprings/generation_'
correlations_folder_name = folder + 'correlations/generation_'


def write_header(pop_size):
    header = []
    for i in range(pop_size):
        header.append("circuit " + str(i+1))
    
    with open(folder + 'fitnesses.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        

def plot_correlations(opt, all=False):  
    if not all:
        remainder = int(opt.pop_size * 0.1)
    else:
        remainder = opt.pop_size
#         upper_limit = 500
#         iterations = int(opt.pop_size/upper_limit)

#         for j in range(iterations):
#             args = []
#             gpu_device = cycle(range(8))
#             for i in range(upper_limit):
#                 circuit = opt.population.individuals[i]
#                 args.append((opt, circuit, i, str(next(gpu_device))))

#             with mp.get_context("spawn").Pool(processes=upper_limit) as pool:
#                 pool.starmap(plot_correlation, args)

#         remainder = opt.pop_size % upper_limit
        
    args = []
    gpu_device = cycle(range(8))
    for index in range(remainder):
        circuit = opt.population.individuals[index]
        args.append((opt, circuit, index, str(next(gpu_device))))
        
    if remainder > 0:
        with mp.get_context("spawn").Pool(processes=remainder) as pool:
            pool.starmap(plot_correlation, args)
            
    
        
        
def plot_and_draw_everything(opt, all=False):
    if not all:
        remainder = int(opt.pop_size * 0.1)
    else:
        remainder = opt.pop_size
#         upper_limit = 500
#         iterations = int(opt.pop_size/upper_limit)
#         for j in range(iterations):
#             args = []
#             gpu_device = cycle(rasnge(8))
#             for i in range(upper_limit):
#                 circuit = opt.population.individuals[i]
#                 args.append((opt, circuit, i, str(next(gpu_device))))

#             with mp.get_context("spawn").Pool(processes=upper_limit) as pool:
#                 pool.starmap(plot_everything, args)
            
#         remainder = opt.pop_size % upper_limit
        
    args = []
    gpu_device = cycle(range(8))
    for i in range(remainder):
        circuit = opt.population.individuals[i]
        args.append((opt, circuit, i, str(next(gpu_device))))
        
    if remainder > 0:
        with mp.get_context("spawn").Pool(processes=remainder) as pool:
            pool.starmap(plot_everything, args)
            

def plot_distribution(opt):
    with open(folder + 'fitnesses.csv', 'a') as f:
        writer = csv.writer(f)

        fitnesses = []

        for circuit in opt.population.individuals:
            fitnesses.append(circuit.get_fitness())
        writer.writerow(fitnesses)
        
    fig = plt.figure( figsize=(20, 14))
    plt.xticks(fontsize= 15)
    plt.yticks(fontsize= 15)
    plt.title("generation " + str(opt.generation), fontsize=20)
    plt.xlabel('fitness values', fontsize=15)
    plt.ylabel('quantity', fontsize=15)
    plt.hist(fitnesses, range=[0.0, np.ceil(opt.population.fittest.fitness)], bins=10, rwidth=0.95)
    if not os.path.exists(folder + 'distributions/'):
        os.makedirs(folder + 'distributions/',exist_ok=True)
    plt.savefig(distributions_folder_name + str(opt.generation) + '.pdf', bbox_inches='tight')
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

        

