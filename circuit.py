import copy
import os
import random
import time


import pandas as pd
import matplotlib.pyplot as plt
from pennylane import numpy as np
from pennylane import broadcast
import pennylane as qml
import multiprocessing as mp

from gates_setup_general import *

qml.drawer.use_style('black_white')


class IndividualCircuit:
    
    def __init__(self, tensor, target, one_q_gates, two_q_gates, n_qubits, dimension):
        self.tensor = tensor
        self.params = (2.*np.pi)*np.random.random(get_num_params(tensor, n_qubits, one_q_gates, two_q_gates))
        self.one_q_gates = one_q_gates
        self.two_q_gates = two_q_gates
        self.fitness = 0.0
        self.loss = 0.0
        self.losses = []
        self.fitnesses = []
        self.n_qubits = n_qubits
        self.best_loss_idx = 0
        self.target = target
        if dimension == 1:
            self.loss_method = self.loss_1d
        elif dimension == 2:
            self.loss_method = self.loss_2d
        elif dimension == 3:
            self.loss_method = self.loss_3d
        
    
        
    def train(self, circuit):
        trained_params = self.params.copy()
        start = time.mktime(time.gmtime())
        stop = start
        i_steps = 0
        init_stepsize = 0.01
        opt = qml.AdamOptimizer(stepsize=init_stepsize)
        best_params = trained_params
        while True:
            iteration = len(self.losses)
            trained_params, L = opt.step_and_cost(lambda v: self.loss_method(circuit,v), trained_params)
            self.losses.append( L )
                
            # update the stop conditions
            early_stop  = self.early_stopping(patience=20)
            stop = early_stop
            
            # keep track of the min loss and corresponding params
            curr_min_idx = np.where(self.losses == np.min(self.losses))[0][-1]
            self.best_loss_idx = curr_min_idx
            if curr_min_idx == iteration - 1: 
                self.best_loss_idx = curr_min_idx
                self.params = trained_params
                
            if stop: break
        
        self.losses = np.log(self.losses)
        self.fitnesses = -self.losses
        
    
    def early_stopping(self, patience=1, draw=False):
        #check whether the loss is plateauing
        if len(self.losses) < 2*patience:
            #not enough values to think about
            return False

        ref_value = np.min(self.losses[-2*patience:-patience])
        last_value = np.min( self.losses[-patience:])
        return (last_value >= ref_value)
    

    def train_fitness(self):
        dev_wires = [np.array(idx, requires_grad=True) for idx in range(self.n_qubits)]
        dev = qml.device('qulacs.simulator', wires=dev_wires, gpu=True, shots=10000)
        self.dev_wires = dev_wires
    
        qnode = qml.QNode(convert_tensor_to_circuit, dev)
        self.train(qnode)
        self.loss = self.losses[self.best_loss_idx]
        self.fitness = self.fitnesses[self.best_loss_idx]
        
    
    def get_fitness(self):
        return self.fitness
    
    
    def draw(self, title=""):
        dev = qml.device('qulacs.simulator', wires=self.dev_wires, gpu=True, shots=10000)
        qnode = qml.QNode(convert_tensor_to_circuit, dev)
        fig, ax = qml.draw_mpl(qnode)(self.tensor, self.params, self.n_qubits, self.one_q_gates, self.two_q_gates, self.dev_wires)
        ax.set_title(title)
        fig.show()
    
    
    def plot_fitness(self, title="fitnesses"):
        plt.title(title)
        plt.plot(self.fitnesses)
        plt.xlabel('iterations')
        plt.ylabel('fitness: -log JS Divergence')
        plt.scatter(self.best_loss_idx, self.fitnesses[self.best_loss_idx],c='r')
        plt.show()
    
    
    def plot_loss(self, title="losses"):
        plt.title(title)
        plt.plot(self.losses)
        plt.xlabel('iterations')
        plt.ylabel('loss: log JS Divergence')
        plt.scatter(self.best_loss_idx, self.losses[self.best_loss_idx],c='r')
        plt.show()
        
        
    def update_params(self):
        self.params = (2.*np.pi)*np.random.random(get_num_params(self.tensor, self.n_qubits, self.one_q_gates, self.two_q_gates))
        
        
    def get_tensor(self):
        return self.tensor 
    
    def qml_entropy(self,pk,qk):
        '''
        do not use scipy implementatin of entropy, will use different numpy versions
        this uses the pennylane wrapped version of numpy
        '''
        qk = np.asarray(qk)
        ck = np.broadcast(pk, qk)

        vec = [u*np.log(u/v) if (u>0 and v>0) else 0 if (u == 0 and v>=0) else np.inf for (u,v) in ck]
        S = np.sum(vec)
        return S

    def jensen_shannon_loss(self, p,q):
        M = np.multiply(q+p,0.5)
        return 0.5*self.qml_entropy(p,M)+0.5*self.qml_entropy(q,M)
        
        
    def loss_1d(self, circuit, params):
        return self.jensen_shannon_loss(circuit(self.tensor, params, self.n_qubits, self.one_q_gates, self.two_q_gates, self.dev_wires), self.target)
    
    def loss_2d(self, circuit, params):
        return self.jensen_shannon_loss(circuit(self.tensor, params, self.n_qubits, self.one_q_gates, self.two_q_gates, self.dev_wires).reshape(16,16),self.target.reshape(16,16))
    
    def loss_3d(self, circuit, params):
        return self.jensen_shannon_loss(circuit(self.tensor, params, self.n_qubits, self.one_q_gates, self.two_q_gates, self.dev_wires).reshape(16,16,16),self.target.reshape(16,16,16))
    
    
def circuit_fitness(circuit, device_number):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device_number)
    # print('cuda device: ', os.environ['CUDA_VISIBLE_DEVICES'])
    circuit.train_fitness()
    return circuit