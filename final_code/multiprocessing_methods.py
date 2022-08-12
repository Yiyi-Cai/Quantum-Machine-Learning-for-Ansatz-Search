import csv
from pennylane import numpy as np
import os

from file_reading import *
from gates_setup import *
plt.rcParams['font.size'] = '25'

folder = '../optimizer_data/2D_run3/'
fitnesses_folder_name = folder + 'plots_and_circuits/generation_'
offsprings_folder_name = folder +'offsprings_2D/generation_'

def plot_everything(opt, circuit, i, device_number):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device_number)
    
    plot_fitness(opt, circuit, i)
    plot_loss(opt, circuit, i)
    draw_circuit(opt, circuit, i)
    plot_comparison(opt, circuit, i)
    
    

def plot_fitness(opt, circuit, i):
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
    
    
def plot_loss(opt, circuit, i):
    plt.figure( figsize=(20, 14))
    plt.title("generation " + str(opt.generation) + " circuit " + str(i+1) + " loss: " + str(opt.population.individuals[i].losses[opt.population.individuals[i].best_loss_idx]))
    plt.plot(opt.population.individuals[i].losses)
    plt.xlabel('iterations')
    plt.ylabel('losses')
    directory1 = fitnesses_folder_name + str(opt.generation)
    # if not os.path.exists(directory1):
    #     os.makedirs(directory1)
    plt.scatter(opt.population.individuals[i].best_loss_idx, opt.population.individuals[i].losses[opt.population.individuals[i].best_loss_idx],c='r')
    plt.savefig(directory1 + '/circuit_' + str(i+1) +'_losses.png', bbox_inches='tight')
    plt.cla()
    plt.close("all")
    
    
def draw_circuit(opt, circuit, i):
    dev = qml.device('qulacs.simulator', wires=circuit.dev_wires, gpu=True, shots=10000)
    qnode = qml.QNode(convert_tensor_to_circuit, dev)

    fig, ax = qml.draw_mpl(qnode)(circuit.tensor, circuit.params, circuit.n_qubits, circuit.num_one_gates, circuit.num_two_gates, circuit.dev_wires)
    title = "generation " + str(opt.generation) +  " circuit " + str(i+1) + " fitness: " + str(circuit.get_fitness())
    ax.set_title(title)
    directory = fitnesses_folder_name + str(opt.generation)
    # if not os.path.exists(directory):
    #     os.makedirs(directory)
    fig.savefig(directory + '/circuit_' + str(i+1) +'_circuit.png', bbox_inches='tight')
    plt.close("all")
    

def plot_comparison(opt, circuit, i, title = ''):        
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
    # if not os.path.exists(directory):
    #     os.makedirs(directory)
    plt.savefig(directory + '/circuit_' + str(i+1) +'_comparisons.png', bbox_inches='tight')
    plt.close("all")
    
    
    
def draw_and_save_offspring(opt, circuit, i, device_number):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device_number)
    title = "offspring " + str(i+1) + " generation " + str(opt.generation) + " fitness: " + str(circuit.get_fitness())
    dev = qml.device('qulacs.simulator', wires=circuit.dev_wires, gpu=True, shots=10000)
    qnode = qml.QNode(convert_tensor_to_circuit, dev)

    fig, ax = qml.draw_mpl(qnode)(circuit.tensor, circuit.params, circuit.n_qubits, circuit.num_one_gates, circuit.num_two_gates, circuit.dev_wires)
    ax.set_title(title)
    directory = offsprings_folder_name + str(opt.generation)
    if not os.path.exists(directory):
        os.makedirs(directory)
    fig.savefig(directory + '/offspring_' + str(i+1) +'.png', bbox_inches='tight')
    plt.close("all")
    

    
    
    
    