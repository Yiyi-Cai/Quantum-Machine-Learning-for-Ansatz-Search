import csv
from pennylane import numpy as np
from matplotlib.ticker import MaxNLocator
import os

from file_reading_general import *
from gates_setup_general import *
from dimension_setting import *


folder = '../more_gates/' + str(dim)+'d/'
fitnesses_folder_name = folder + 'plots_and_circuits/generation_'
offsprings_folder_name = folder +'offsprings/generation_'


def plot_everything(opt, circuit, i, device_number):
    directory = fitnesses_folder_name + str(opt.generation)
    if not os.path.exists(directory):
        os.makedirs(directory,exist_ok=True)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device_number)
    # print('plotting everything.... ')
    if dim == 2:
        plot_comparison_2d(opt, circuit, i)
    elif dim == 3:
        # plot_fitness_3d(opt, circuit, i)
        plot_comparison_3d(opt, circuit, i)
    plot_log_loss(opt, circuit, i)
    draw_circuit(opt, circuit, i)


    
def plot_fitness(opt, circuit, i):    
    plt.figure( figsize=(20, 14))
    plt.plot(opt.population.individuals[i].fitnesses)
    
    plt.xticks(fontsize= 15)
    plt.yticks(fontsize= 15)

    plt.xlabel('iterations', fontsize=15)
    plt.ylabel('fitness: -log JS Divergence', fontsize=15)

    directory1 = fitnesses_folder_name + str(opt.generation)
    if not os.path.exists(directory1):
        os.makedirs(directory1,exist_ok=True)
    plt.scatter(opt.population.individuals[i].best_loss_idx, opt.population.individuals[i].fitnesses[opt.population.individuals[i].best_loss_idx],c='r')
    plt.savefig(directory1 + '/circuit_' + str(i+1) +'_fitness_{}.pdf'.format(str(opt.population.individuals[i].get_fitness())), bbox_inches='tight')
    plt.cla()
    plt.close("all")
    
    
def plot_log_loss(opt, circuit, i):
    plt.figure( figsize=(20, 14))
    ax = plt.plot(opt.population.individuals[i].losses, linewidth=5)
    
    plt.xticks(fontsize= 30)
    plt.yticks(fontsize= 30)

    plt.xlabel('iterations', fontsize=50)
    plt.ylabel('losses: log JS Divergence', fontsize=50)
    
    
    directory1 = fitnesses_folder_name + str(opt.generation)
    plt.scatter(opt.population.individuals[i].best_loss_idx, opt.population.individuals[i].losses[opt.population.individuals[i].best_loss_idx],c='r', linewidths=20)
    
    if not os.path.exists(directory1):
        os.makedirs(directory1,exist_ok=True)
    plt.savefig(directory1 + '/circuit_' + str(i+1) +'_losses_{}.pdf'.format(opt.population.individuals[i].losses[opt.population.individuals[i].best_loss_idx]), bbox_inches='tight')
    plt.cla()
    plt.close("all")
    
    
def draw_circuit(opt, circuit, i):
    dev = qml.device('qulacs.simulator', wires=circuit.dev_wires, gpu=True, shots=10000)
    qnode = qml.QNode(convert_tensor_to_circuit, dev)

    fig, ax = qml.draw_mpl(qnode)(circuit.tensor, circuit.params, circuit.n_qubits, circuit.one_q_gates, circuit.two_q_gates, circuit.dev_wires)
    title = "generation " + str(opt.generation) +  " circuit " + str(i+1)
    ax.set_title(title)
    directory = fitnesses_folder_name + str(opt.generation)
    fig.savefig(directory + '/circuit_' + str(i+1) +'_circuit.pdf', bbox_inches='tight')
    plt.close("all")
    
    
    
def plot_comparison_2d(opt, circuit, i, title = ''):       
    dev = qml.device('qulacs.simulator', wires=circuit.dev_wires, gpu=True, shots=10000)
    qnode = qml.QNode(convert_tensor_to_circuit, dev)

    h1dgen = qnode(circuit.tensor, circuit.params, circuit.n_qubits, circuit.one_q_gates, circuit.two_q_gates, circuit.dev_wires)
    h2dgen = h1dgen.reshape((16,16))
    
    fig = plt.figure( figsize=(250, 250))
    
    h1gen = np.sum(h2dgen, axis=1)
    h1gen = h1gen / np.sum(h1gen)
    h2gen = np.sum(h2dgen, axis=0)
    h2gen = h2gen / np.sum(h2gen)
    plt.suptitle(title)
    
    fig.autofmt_xdate()
    ax1 = plt.subplot2grid((2,2), (0,0))
    ax1.tick_params(axis='both', which='major', labelsize=170)
    ax1.tick_params(axis='both', which='minor', labelsize=170)
    plt.bar(bins_pt,probability_pt, label='target', width=5, align='edge')
    plt.bar(bins_pt, h1gen, label='generated',width=-5, align='edge')
    plt.xlabel('Jet pT [GeV]', fontsize=250)
    # plt.legend()
    
    
    ax2 = plt.subplot2grid((2,2), (0,1))
    ax2.tick_params(axis='both', which='major', labelsize=170)
    ax2.tick_params(axis='both', which='minor', labelsize=170)
    plt.bar(bins_mass,probability_mass , label='target', width=4.5, align='edge')
    plt.bar(bins_mass, h2gen, label='generated',width=-4.5, align='edge')
    plt.xlabel('Jet mass [GeV]', fontsize=250)
    plt.legend(fontsize=230)
    
    ax3 = plt.subplot2grid((2,2), (1,0))
    ax3.tick_params(axis='both', which='major', labelsize=170)
    ax3.tick_params(axis='both', which='minor', labelsize=170)
    plt.imshow(np.flip(target_2d.transpose(), axis=0),
               extent=[ bins_pt_edge[0], bins_pt_edge[-1],bins_mass_edge[0],bins_mass_edge[-1]],
               aspect='auto')
    plt.xlabel('Jet pT [GeV]', fontsize=250)
    plt.ylabel('Jet mass [GeV]', fontsize=250)
    plt.title('Target',fontsize=300)
    
    ax4 = plt.subplot2grid((2,2), (1,1))
    ax4.tick_params(axis='both', which='major', labelsize=170)
    ax4.tick_params(axis='both', which='minor', labelsize=170)
    plt.subplot(224)
    plt.imshow(np.flip(h2dgen.transpose(),axis=0),
               extent=[ bins_pt_edge[0], bins_pt_edge[-1],bins_mass_edge[0],bins_mass_edge[-1]],
               aspect='auto')
    plt.xlabel('Jet pT [GeV]', fontsize=250)
    plt.ylabel('Jet mass [GeV]', fontsize=250)
    plt.title('Generated',fontsize=300)
    
    
    directory = fitnesses_folder_name + str(opt.generation)
    plt.savefig(directory + '/circuit_' + str(i+1) +'_comparisons.pdf', bbox_inches='tight')
    plt.close("all")
    
    

def plot_comparison_3d(opt, circuit, i, title = ''):        
    dev = qml.device('qulacs.simulator', wires=circuit.dev_wires, gpu=True, shots=10000)
    qnode = qml.QNode(convert_tensor_to_circuit, dev)
    
    h1dgen = qnode(circuit.tensor, circuit.params, circuit.n_qubits, circuit.one_q_gates, circuit.two_q_gates, circuit.dev_wires)
    h3dgen = h1dgen.reshape((16,16,16))
    h1gen = np.sum(h3dgen, axis=2).sum(axis=1)
    h1gen = h1gen / np.sum(h1gen)
    h2gen = np.sum(h3dgen, axis=0).sum(axis=1)
    h2gen = h2gen / np.sum(h2gen)
    h3gen = np.sum(h3dgen, axis=1).sum(axis=0)
    h3gen = h3gen / np.sum(h3gen)
    
    h12gen = np.sum(h3dgen, axis=2)
    h12gen = h12gen / np.sum(h12gen)
    h13gen = np.sum(h3dgen, axis=1)
    h13gen = h13gen / np.sum(h13gen)
    h14gen = np.sum(h3dgen, axis=0)
    h14gen = h14gen / np.sum(h14gen)
    
    h3dd = target_3d.reshape((16,16,16))
    P_target_pt = np.sum(h3dd, axis=2).sum(axis=1)
    P_target_pt = P_target_pt / np.sum(P_target_pt)
    P_target_mass = np.sum(h3dd, axis=0).sum(axis=1)
    P_target_mass = P_target_mass  / np.sum(P_target_mass)
    P_target_nsubj = np.sum(h3dd, axis=1).sum(axis=0)
    P_target_nsubj = P_target_nsubj / np.sum(P_target_nsubj)
    
    h12d = np.sum(h3dd, axis=2)
    h12d = h12d / np.sum(h12d)
    h13d = np.sum(h3dd, axis=1)
    h13d = h13d / np.sum(h13d)
    h14d = np.sum(h3dd, axis=0)
    h14d = h14d / np.sum(h14d)
    
    
    fig = plt.figure( figsize=(270, 350))
    fig.autofmt_xdate()
    ax1 = plt.subplot2grid((4,3), (0,0))
    ax1.tick_params(axis='both', which='major', labelsize=120)
    ax1.tick_params(axis='both', which='minor', labelsize=120)
    plt.bar(bins_pt,P_target_pt , label='target', width=7, align='edge')
    plt.bar(bins_pt, h1gen, label='generated',  width=-7, align='edge')
    plt.xlabel('Jet pT [GeV]', fontsize=250)   
    # plt.legend(loc='upper center')
    # plt.legend(fontsize=60)
    

    ax2 = plt.subplot2grid((4,3), (0,1))
    ax2.tick_params(axis='both', which='major', labelsize=120)
    ax2.tick_params(axis='both', which='minor', labelsize=120)
    plt.bar(bins_mass,P_target_mass , label='target', width=5, align='edge')
    plt.bar(bins_mass, h2gen, label='generated',width=-5, align='edge')
    plt.xlabel('Jet mass [GeV]', fontsize=250)
    plt.legend(fontsize=230)
    # plt.legend()
    
    ax3 = plt.subplot2grid((4,3), (0,2))
    ax3.tick_params(axis='both', which='major', labelsize=120)
    ax3.tick_params(axis='both', which='minor', labelsize=120)
    plt.bar(bins_tau,P_target_nsubj , label='target', width=0.03, align='edge')
    plt.bar(bins_tau, h3gen, label='generated',width=-0.03, align='edge')
    plt.xlabel('N-subjettiness Discriminant',  fontsize=250)
    
    
    ax4 = plt.subplot2grid((4,3), (1,0))
    ax4.tick_params(axis='both', which='major', labelsize=120)
    ax4.tick_params(axis='both', which='minor', labelsize=120)
    plt.imshow(np.flip(h13d.transpose(), axis=0),
               extent=[ bins_pt_edge[0], bins_pt_edge[-1],bins_tau_edge[0],bins_tau_edge[-1]],
               aspect='auto')
    plt.xlabel('Jet pT [GeV]',  fontsize=180)
    plt.ylabel('N-subjettiness Discriminant',  fontsize=180)
    plt.gca().yaxis.set_major_locator(MaxNLocator(prune='lower'))
    plt.title('Target',fontsize=300)
    
    
    ax5 = plt.subplot2grid((4,3), (1,1))
    ax5.tick_params(axis='both', which='major', labelsize=120)
    ax5.tick_params(axis='both', which='minor', labelsize=120)
    plt.imshow(np.flip(h13gen.transpose(),axis=0),
               extent=[ bins_pt_edge[0], bins_pt_edge[-1],bins_tau_edge[0],bins_tau_edge[-1]],
               aspect='auto')
    plt.xlabel('Jet pT [GeV]',  fontsize=180)
    plt.ylabel('N-subjettiness Discriminant',  fontsize=180)
    plt.gca().yaxis.set_major_locator(MaxNLocator(prune='lower'))
    plt.title('Generated', fontsize=300)
    
    ax6 = plt.subplot2grid((4,3), (2,0))
    ax6.tick_params(axis='both', which='major', labelsize=120)
    ax6.tick_params(axis='both', which='minor', labelsize=120)
    plt.imshow(np.flip(h12d.transpose(), axis=0),
               extent=[ bins_pt_edge[0], bins_pt_edge[-1],bins_mass_edge[0],bins_mass_edge[-1]],
               aspect='auto')
    plt.xlabel('Jet pT [GeV]',fontsize=180)
    plt.ylabel('Jet mass [Gev]',fontsize=180)
    plt.title('Target', fontsize=300)
    
    ax7 = plt.subplot2grid((4,3), (2,1))
    ax7.tick_params(axis='both', which='major', labelsize=120)
    ax7.tick_params(axis='both', which='minor', labelsize=120)
    plt.imshow(np.flip(h12gen.transpose(),axis=0),
               extent=[ bins_pt_edge[0], bins_pt_edge[-1],bins_mass_edge[0],bins_mass_edge[-1]],
               aspect='auto')
    plt.xlabel('Jet pT [GeV]',fontsize=180)
    plt.ylabel('Jet mass [GeV]',fontsize=180)
    plt.title('Generated',fontsize=300)
    
    ax8 = plt.subplot2grid((4,3), (3,0))
    ax8.tick_params(axis='both', which='major', labelsize=120)
    ax8.tick_params(axis='both', which='minor', labelsize=120)
    plt.imshow(np.flip(h14d.transpose(), axis=0),
               extent=[bins_pt_edge[0],bins_pt_edge[-1],  bins_tau_edge[0], bins_tau_edge[-1]],
               aspect='auto')
    plt.xlabel('Jet mass [Gev]',fontsize=180)
    plt.ylabel('N-subjettiness Discriminant',  fontsize=180)
    plt.title('Target', fontsize=300)
    
    ax9 = plt.subplot2grid((4,3), (3,1))
    ax9.tick_params(axis='both', which='major', labelsize=120)
    ax9.tick_params(axis='both', which='minor', labelsize=120)
    plt.imshow(np.flip(h14gen.transpose(),axis=0),
               extent=[bins_pt_edge[0],bins_pt_edge[-1],  bins_tau_edge[0], bins_tau_edge[-1]],
               aspect='auto')
    plt.xlabel('Jet mass [Gev]',fontsize=180)
    plt.ylabel('N-subjettiness Discriminant',  fontsize=180)
    plt.title('Generated',fontsize=300)
    
    directory = fitnesses_folder_name + str(opt.generation)
    if not os.path.exists(directory):
        os.makedirs(directory,exist_ok=True)
    plt.savefig(directory + '/circuit_' + str(i+1) +'_comparisons.pdf', bbox_inches='tight')
    plt.close("all")
    
    
    
    
def draw_and_save_offspring(opt, circuit, i, device_number):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device_number)
    title = "offspring " + str(i+1) + " generation " + str(opt.generation) + " fitness: " + str(circuit.get_fitness())
    dev = qml.device('qulacs.simulator', wires=circuit.dev_wires, gpu=True, shots=10000)
    qnode = qml.QNode(convert_tensor_to_circuit, dev)

    fig, ax = qml.draw_mpl(qnode)(circuit.tensor, circuit.params, circuit.n_qubits, circuit.one_q_gates, circuit.two_q_gates, circuit.dev_wires)
    ax.set_title(title)
    directory = offsprings_folder_name + str(opt.generation)
    if not os.path.exists(directory):
        os.makedirs(directory,exist_ok=True)
    fig.savefig(directory + '/offspring_' + str(i+1) +'.pdf', bbox_inches='tight')
    plt.close("all")