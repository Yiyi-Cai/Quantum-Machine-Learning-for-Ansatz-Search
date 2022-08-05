import pandas as pd
import matplotlib.pyplot as plt
from pennylane import numpy as np
from pennylane.templates import BasicEntanglerLayers,StronglyEntanglingLayers,RandomLayers
from pennylane import broadcast
import h5py
import pennylane as qml
import random
import time
import multiprocessing as mp
from itertools import combinations
qml.__version__


n_qubits = 8
dev_wires = [np.array(idx, requires_grad=True) for idx in range(n_qubits)]
dev = qml.device('qulacs.simulator', wires=dev_wires, gpu=True, shots=10000)
num_bins = 16

### keys represent gates, values represent number of parameters required
two_qubits_gates_with_params = {qml.CNOT:0}
one_qubit_gates_with_params = {qml.RX:1, qml.RY:1, qml.RZ:1}
two_qubits_gates = list(two_qubits_gates_with_params.keys())
one_qubit_gates = list(one_qubit_gates_with_params.keys())
num_one_gates = 3
num_two_gates = 1


def get_num_params(tensor, n_qubits, num_one_gates, num_two_gates):
    num_params = 0
    total_gates = num_one_gates + num_two_gates
    for moment in tensor:
        for q_idx in range(n_qubits):
            curr_qubit = moment[q_idx]
            for gate_idx in range(0, total_gates):
                if not curr_qubit[gate_idx]:
                    if gate_idx <= num_one_gates:
                        gate = one_qubit_gates[gate_idx]
                        curr_num_params = one_qubit_gates_with_params[gate]
                        num_params += curr_num_params
                    else:
                        gate = two_qubits_gates[gate_idx - num_one_gates]
                        target_idx = curr_qubit[gate_idx] - 1
                        curr_num_params = two_qubits_gates_with_params[gate]
                        num_params += curr_num_params
                    break
    return num_params


def generate_random_tensor(n_qubits, num_one_gates, num_two_gates, max_moments):
    tensor = []
    total_gates = num_one_gates + num_two_gates * 2
    num_moments = random.randint(1,max_moments)
    for m in range(num_moments):
        moment = []
        for q in range(n_qubits):
            curr_qubit = [0] * total_gates
            moment.append(curr_qubit)
        qubit_options = list(range(n_qubits))
        gates_options = list(range(total_gates))
        while not len(qubit_options) == 0:
            rand_gates_idx = random.choice(gates_options)
            if rand_gates_idx < num_one_gates:
                rand_qubit_idx = random.choice(qubit_options)
                moment[rand_qubit_idx][rand_gates_idx] = 1
                qubit_options.remove(rand_qubit_idx)
            else:
                if len(qubit_options) >= 2:
                    rand_control, rand_target = random.sample(qubit_options, k = 2)
                    qubits_used = [rand_control, rand_target]
                    # qubits_used = list(range(min(rand_target, rand_control), max(rand_target, rand_control)))
                    if all(x in qubit_options for x in qubits_used):
                        if (rand_gates_idx - num_one_gates) % 2 == 0:
                            moment[rand_control][rand_gates_idx] = rand_target + 1
                            moment[rand_target][rand_gates_idx + 1] = rand_control + 1
                        else:
                            moment[rand_control][rand_gates_idx -1] = rand_target + 1
                            moment[rand_target][rand_gates_idx] = rand_control + 1
                        qubit_options = [ q for q in qubit_options if not q in qubits_used]
                    else:
                        qubit_options.append(rand_control)
        tensor.append(moment)
    return tensor


@qml.qnode(dev)
def convert_tensor_to_circuit(tensor, params, n_qubits, num_one_gates, num_two_gates, prob=True):
    total_gates = num_one_gates + num_two_gates*2
    params_idx = 0
    for moment in tensor:
        for q_idx in range(n_qubits):
            curr_qubit = moment[q_idx]
            for gate_idx in range(total_gates):
                if not curr_qubit[gate_idx] == 0:
                    if gate_idx < num_one_gates:
                        gate = one_qubit_gates[gate_idx]
                        curr_num_params = one_qubit_gates_with_params[gate]
                        if curr_num_params == 0:
                            gate(wires=dev_wires[q_idx])
                        else:
                            broadcast(unitary=gate, pattern="single", wires=dev_wires[q_idx], parameters=params[params_idx:params_idx+curr_num_params])
                            params_idx += curr_num_params
                    else:
                        if (gate_idx - num_one_gates) % 2 == 0:
                            gate = two_qubits_gates[gate_idx - num_one_gates]
                            target_idx = curr_qubit[gate_idx] -1
                            curr_num_params = two_qubits_gates_with_params[gate]
                            if curr_num_params == 0:
                                gate(wires=[dev_wires[q_idx], dev_wires[target_idx]])
                            else:
                                broadcast(unitary=gate, pattern="double", wires=[dev_wires[q_idx], dev_wires[target_idx]], parameters=params[params_idx:params_idx+curr_num_params])
                                params_idx += curr_num_params
                    break
    
    if prob:
        return qml.probs(wires=dev_wires[0:n_qubits])
    else:
        return qml.sample(wires=dev_wires[0:n_qubits])
    

    
### Loss Functions

def qml_entropy(pk,qk):
    '''
    do not use scipy implementatin of entropy, will use different numpy versions
    this uses the pennylane wrapped version of numpy
    '''
    qk = np.asarray(qk)
    ck = np.broadcast(pk, qk)

    vec = [u*np.log(u/v) if (u>0 and v>0) else 0 if (u == 0 and v>=0) else np.inf for (u,v) in ck]
    S = np.sum(vec)
    return S

def hybrid_jensen_shannon_loss(pk,qk):
    ''' this is a temporary function while we debug the layerwise training'''
    p = target
    q = hybrid_pdf(params,fixed_params=fixed_params)
    M = np.multiply(q+p,0.5)

    return 0.5*qml_entropy(p,M)+0.5*qml_entropy(q,M)

def jensen_shannon_loss(p,q):
    M = np.multiply(q+p,0.5)
    return 0.5*qml_entropy(p,M)+0.5*qml_entropy(q,M)



### Stop conditions

def scheduled_stepsize(iteration):
    min_step= 0.000001
    max_step = 0.01
    ss = min_step+(max_step - min_step)*np.abs(np.cos(np.pi/10.*iteration))
    print('step size set to {}'.format(ss))
    return ss

def scheduled_opt(self,iteration, opt_method):
    return opt_method(stepsize=scheduled_stepsize(iteration))


def early_stopping(loss_values ,  patience=1, draw=False):
    #check whether the loss is plateauing
    if len(loss_values) < 2*patience:
        #not enough values to think about
        return False

    ref_value = np.min(loss_values[-2*patience:-patience])
    last_value = np.min( loss_values[-patience:])
    return (last_value >= ref_value)




class IndividualCircuit:
    
    def __init__(self, tensor, target, num_one_gates, num_two_gates, n_qubits, n_steps=None, scheduled=False):
        self.tensor = tensor
        self.params = (2.*np.pi)*np.random.random(get_num_params(tensor, n_qubits, num_one_gates, num_two_gates))
        self.num_one_gates = num_one_gates
        self.num_two_gates = num_two_gates
        self.fitness = 0.0
        self.losses = []
        self.fitnesses = []
        self.n_qubits = n_qubits
        self.best_loss_idx = 0
        self.target = target
        self.LR = []
        
    def train(self, circuit, n_steps=None, scheduled=False):
        trained_params = self.params.copy()
        start = time.mktime(time.gmtime())
        stop = start
        i_steps = 0
        init_stepsize = 0.01
        opt = qml.AdamOptimizer(stepsize=init_stepsize)
        # self.LR.append(init_stepsize)
        best_params = trained_params
        while True:
            # update the circuit parameters in each iteration
            prelap = time.mktime(time.gmtime())
            iteration = len(self.losses)
            # print('starting iteration {}'.format(len(self.losses)+1))
            if scheduled: 
                opt=scheduled_opt(len(self.losses))
            trained_params, L = opt.step_and_cost(lambda v: self.loss_2d(circuit,v), trained_params)
            lap = start = time.mktime(time.gmtime())
            step_time = lap - prelap
            stop+=step_time
            # L = loss(trained_params) 
            self.losses.append( L )
            
#             if adaptLR(self.losses, opt):
#                 self.LR.append(opt.stepsize)
                
            
            # update the stop conditions
            early_stop  = early_stopping( self.losses, patience=20)
            i_steps+=1
            i_stop = (n_steps!=None) and (i_steps>=n_steps)
            stop = early_stop or i_stop
            
            # keep track of the min loss and corresponding params
            curr_min_idx = np.where(self.losses == np.min(self.losses))[0][-1]
            self.best_loss_idx = curr_min_idx
            if curr_min_idx == iteration - 1: 
                self.best_loss_idx = curr_min_idx
                self.params = trained_params
                
            if stop: break
        
        self.fitnesses = [1 - x for x in self.losses]
        
        
    def adaptLR(self, opt):
        last_N = 20
        factor = 1.5
        if len(self.losses)<last_N: return False
        if len(self.losses)-self.last_LR_change< last_N: return False
        max_consecutive_downards = np.max([ sum( 1 for _ in group ) for key, group in itertools.groupby( np.diff( self.losses[-last_N:]) <0  ) if key ])
        roughness = 1.-max_consecutive_downards / float(last_N-1)
        print(f"Loss roughness: {roughness}")
        if roughness > 0.5:
            self.last_LR_change = len(losses)
            old_LR = np.asarray(opt.stepsize)
            opt.stepsize = old_LR/ factor
            print(f"Reducing LR :{old_LR} \u2192 {np.asarray(opt.stepsize)}")
            plt.figure(figsize=(10,3))
            plt.plot(losses[-last_N:])
            ymin,ymax = plt.ylim()
            offset = 0.1* (ymax-ymin)
            plt.xlabel('last iterations')
            plt.ylabel('training loss')
            plt.show()
            return True


    def train_fitness(self, n_steps=None, scheduled=False):
        self.train(convert_tensor_to_circuit, n_steps, scheduled)
        self.fitness = self.fitnesses[self.best_loss_idx]
    
    def get_fitness(self):
        return self.fitness
    
    
    def draw(self, title=""):
        fig, ax = qml.draw_mpl(convert_tensor_to_circuit)(self.tensor, self.params, self.n_qubits, self.num_one_gates, self.num_two_gates)
        ax.set_title(title)
        fig.show()
    
    def plot_fitness(self, title="fitnesses"):
        plt.title(title)
        plt.plot(self.fitnesses)
        plt.xlabel('iterations')
        plt.ylabel('fitness')
        plt.scatter(self.best_loss_idx, self.fitnesses[self.best_loss_idx],c='r')
        plt.show()
    
    def plot_loss(self, title="losses"):
        plt.title(title)
        plt.plot(self.losses)
        plt.xlabel('iterations')
        plt.ylabel('training loss')
        plt.scatter(self.best_loss_idx, self.losses[self.best_loss_idx],c='r')
        plt.show()
        
        
    def get_tensor(self):
        return self.tensor        
        
    def loss_2d(self, circuit, params):
        return jensen_shannon_loss(circuit(self.tensor, params, self.n_qubits, self.num_one_gates, self.num_two_gates).reshape(16,16),self.target.reshape(16,16))
    
    def update_params(self):
        self.params = (2.*np.pi)*np.random.random(get_num_params(self.tensor, self.n_qubits, self.num_one_gates, self.num_two_gates))
    
    

class Population:
    def __init__(self):
        self.pop_size = 0
        self.individuals = []
        self.fittest = None
        self.least_fittest = None
    
    def fill_population(self, pop_size, target, num_one_gates, num_two_gates, max_moments, n_qubits):
        for i in range(pop_size):
            unique = True
            random_tensor = generate_random_tensor(n_qubits, num_one_gates, num_two_gates, max_moments)
            for j in range(self.pop_size):
                if random_tensor == self.individuals[j].tensor:
                    unique = False
                    break
            if unique:
                curr_circuit = IndividualCircuit(random_tensor, target, num_one_gates, num_two_gates, n_qubits)
                self.pop_size += 1
                self.individuals.append(curr_circuit)
            else:
                i -= 1
        self.update()
    
    
    def add_circuit(self, circuit, increase_size=True):
        self.individuals.append(circuit)
        if increase_size:
            self.pop_size += 1
        self.update()
    
    def add_circuits(self, circuits, increase_size=True):
        self.individuals.extend(circuits.individuals)
        if increase_size:
            self.pop_size += len(circuits)
        self.update()
    
    def get_fittest_fitness(self):
        return self.fittest.get_fitness()
    
    def remove_circuit(self, circuit):
        self.individuals.remove(circuit)
        self.pop_size -= 1
        self.update()
        
        
    def update(self):
        self.individuals.sort(key=lambda circuit:circuit.fitness, reverse=True)
        self.fittest = self.individuals[0]
        self.least_fittest = self.individuals[-1]
        self.pop_size = len(self.individuals)
        
    
    def train_all(self):
        start_time = time.time()
        procs = []
        for circuit in self.individuals:
            proc = mp.Process(target=circuit.train_fitness())
            procs.append(proc)
            proc.start()
        
        for proc in procs:
            proc.join()
        
        end_time = time.time()
        exec_time = end_time - start_time
        print("Execution time multiprocessing {}".format(exec_time))
        self.update()
        
    def train_linear(self):
        start_time = time.time()
        for circuit in self.individuals:
            circuit.train_fitness()
        end_time = time.time()
        exec_time = end_time - start_time
        print("Execution time linear {}".format(exec_time))
        
    
    
    def get_fittests(self, num_fittests):
        self.fittest = self.individuals[0]
        return self.individuals[:num_fittests]     
             
  
class CircuitOptimizer:
    
    def __init__(self, pop_size, target, num_one_gates, num_two_gates, max_moments, n_qubits):
        self.population = Population()
        self.population.fill_population(pop_size, target, num_one_gates, num_two_gates, max_moments, n_qubits)
        self.generation = 0
        self.pop_size = pop_size
        self.num_one_gates = num_one_gates
        self.num_two_gates = num_two_gates
        self.parents = []
        parents_size = int(pop_size * 0.25)
        if parents_size < 2:
            self.parents_size = 2
        else:
            self.parents_size = parents_size
        self.offsprings = Population()
        self.target = target
        self.n_qubits = n_qubits
        self.population.train_all()
        self.print_info()
    
    def draw_all_circuits(self):
        self.print_info()
        for i in range(len(self.population.individuals)):
            title = "circuit " + str(i+1) + " generation " + str(self.generation) + " fitness: " + str(self.population.individuals[i].get_fitness())
            self.population.individuals[i].draw(title)
    
    
    def draw_offsprings(self):
        for i in range(len(self.offsprings.individuals)):
            title = "offspring " + str(i+1) + " generation " + str(self.generation) + " fitness: " + str(self.offsprings.individuals[i].get_fitness())
            self.offsprings.individuals[i].draw(title)
    
            
    def train_offsprings(self):
        self.offsprings.train_all()
            
    def add_offsprings(self):
        self.population.add_circuits(self.offsprings, increase_size=False)
        self.population.individuals = self.population.individuals[:-len(self.offsprings.individuals)]
        self.offsprings = Population()
        
    
    def print_info(self):
        self.population.update()
        print("Generation: {} Fittest: {}".format(self.generation, self.population.fittest.get_fitness()))
        
    
    # choose the two most fit
    def selection(self):
        self.parents = self.population.get_fittests(self.parents_size)
            
        
    
    # choose a random moment as the crossover point, then swap the rest of the moments
    def crossover(self):
        for i in range(0, len(self.parents)-1, 2):
            parent1 = self.parents[i]
            parent2 = self.parents[i+1]
            self.crossover_pair(parent1, parent2)
        if len(self.parents) % 2 == 1:
            parent1 = self.parents[-1]
            parent2 = random.choice(self.parents[:len(self.parents)-1])
            self.crossover_pair(parent1, parent2)
        
            
    
    def crossover_pair(self, parent1, parent2):
        if len(parent1.tensor) == 1:
            cross_pt1 = random.randint(1, len(parent1.tensor))
            tensor1 = parent1.get_tensor()
            if len(parent2.tensor) == 1:
                cross_pt2 = random.randint(1, len(parent2.tensor))
                tensor2 = parent2.get_tensor()
            else:
                cross_pt2 = random.randint(1, len(parent2.tensor)-1)
                tensor2 = parent2.get_tensor()[:cross_pt2]
                tensor1 += parent2.get_tensor()[:cross_pt2]
        elif len(parent2.tensor) == 1:
            cross_pt2 = random.randint(1, len(parent2.tensor))
            tensor2 = parent2.get_tensor()
            cross_pt1 = random.randint(1, len(parent1.tensor)-1)
            tensor1 = parent1.get_tensor()[:cross_pt1]
            tensor2 += parent1.get_tensor()[:cross_pt1]
        else:
            cross_pt1 = random.randint(1, len(parent1.tensor)-1)
            cross_pt2 = random.randint(1, len(parent2.tensor)-1)
            tensor1 = parent1.get_tensor()[:cross_pt1] + parent2.get_tensor()[cross_pt2:]
            tensor2 = parent2.get_tensor()[:cross_pt2] + parent1.get_tensor()[cross_pt1:]
            
        
        
        
        params1 = (2.*np.pi)*np.random.random(get_num_params(tensor1, self.n_qubits, self.num_one_gates, self.num_two_gates))
        params2 = (2.*np.pi)*np.random.random(get_num_params(tensor2, self.n_qubits, self.num_one_gates, self.num_two_gates))
        
        self.offsprings.add_circuit(IndividualCircuit(tensor1, self.target, self.num_one_gates, self.num_two_gates, self.n_qubits))
        self.offsprings.add_circuit(IndividualCircuit(tensor2, self.target, self.num_one_gates, self.num_two_gates, self.n_qubits))
    
        
    
    
    
    
    # choose a random point to switch on/off the gate; for two qubit gates, the target gate is changed to a random value
    def mutate(self):
        # identical_circuits = 0
        for i in range(len(self.offsprings.individuals)):
            
            curr_offspring = self.offsprings.individuals[i]
            mutate_gate = random.randint(0, self.num_one_gates-1)
            mutate_moment = random.randint(0, len(curr_offspring.tensor)-1)
            qubits_options = []
            curr_moment = curr_offspring.tensor[mutate_moment]

            ## mutate one qubit gate
            if random.randint(0, 1):
                
                for q in range(len(curr_moment)):
                    curr_qubit = curr_moment[q]
                    if not any(curr_qubit[self.num_one_gates:]):
                        qubits_options.append(q)
                
                
                if len(qubits_options) == 0:
                    i -= 1
                    continue
                
                
                mutate_qubit = random.choice(qubits_options)
                qubits_options.remove(mutate_qubit)  
                
                
                q = curr_offspring.tensor[mutate_moment][mutate_qubit]
                q[mutate_gate] = curr_offspring.tensor[mutate_moment][mutate_qubit][mutate_gate] ^ 1
                for i in range(self.num_one_gates + self.num_two_gates):
                    if q[i] and not i == mutate_gate:
                        q[i] = 0
                                
                
                
                
            # mutate two qubit gate
            else:
                two_gates_idx = self.num_one_gates + random.randrange(0, self.num_two_gates*2, 2)
                
                for q in range(len(curr_moment)):
                    curr_qubit = curr_moment[q]                        
                    if not any(curr_qubit[:two_gates_idx]):
                        if two_gates_idx + 2 < len(curr_qubit):
                            if not any(curr_qubit[two_gates_idx+2:]):
                                qubits_options.append(q)
                        else:
                            qubits_options.append(q)
                
                if len(qubits_options) <= 1:
                    i -= 1
                    continue
                
                mutate_qubit = random.choice(qubits_options)
                qubits_options.remove(mutate_qubit)  
                
                no_gates_qubits = []
                for qubit in qubits_options:
                    if not any(curr_moment[qubit]):
                        no_gates_qubits.append(qubit)
                
                if not curr_moment[mutate_qubit][two_gates_idx] == 0:
                    choice = random.randint(0,2)
                    if choice == 0:
                        self.swap(qubits_options, curr_moment, mutate_qubit, two_gates_idx)
                    elif choice == 1:
                        self.turn_off(qubits_options, curr_moment, mutate_qubit, two_gates_idx) 
                    else:
                        if len(no_gates_qubits) == 0:
                            i -= 1
                            continue
                        self.mutate_one(no_gates_qubits, curr_moment, mutate_qubit, two_gates_idx)
                        
                elif not curr_moment[mutate_qubit][two_gates_idx+1] == 0:
                    choice = random.randint(0,2)
                    if choice == 0:
                        self.swap(qubits_options, curr_moment, mutate_qubit, two_gates_idx+1, target=True)
                    elif choice == 1:
                        self.turn_off(qubits_options, curr_moment, mutate_qubit, two_gates_idx+1, target=True)
                    else:
                        if len(no_gates_qubits) == 0:
                            i -= 1
                            continue
                        self.mutate_one(no_gates_qubits, curr_moment, mutate_qubit, two_gates_idx+1, target=True)
                    
                else:
                    if len(no_gates_qubits) == 0:
                        i -= 1
                        continue
                    else:
                        new_qubit = random.choice(no_gates_qubits)
                        curr_moment[mutate_qubit][two_gates_idx] = new_qubit + 1
                        curr_moment[new_qubit][two_gates_idx+1] = mutate_qubit + 1
            for j in range(len(self.population.individuals)):
                if self.population.individuals[j].tensor == curr_offspring.tensor:
                    # self.offsprings.remove_circuit(curr_offspring)
                    i -= 1
                    # identical_circuits += 1
            curr_offspring.update_params()
        
#         if identical_circuits > 0:
#              generate_distinct_circuits(identical_circuits)
        
    
#     def generate_distinct_circuits(self, identical_circuits):
#         quantity = self.parent_size - identical_circuits
#         if identical_circuits < self.parent_size:
            
                    
                    
                    
    def plot_all(self):
        for circuit in self.population.individuals:
            circuit.plot_fitness()
            
    
    def plot_offsprings(self):
        for circuit in self.offsprings.individuals:
            circuit.plot_fitness()
            
                    
    def swap(self, qubits_options, curr_moment, mutate_qubit, two_gates_idx, target=False):
        if target:
            new_target = curr_moment[mutate_qubit][two_gates_idx] -1
            new_control = mutate_qubit
            curr_moment[mutate_qubit][two_gates_idx-1] = new_target + 1
            curr_moment[new_target][two_gates_idx] = new_control + 1
            curr_moment[mutate_qubit][two_gates_idx] = 0
            curr_moment[new_target][two_gates_idx - 1] = 0
        else:
            new_control = curr_moment[mutate_qubit][two_gates_idx] - 1
            new_target = mutate_qubit
            curr_moment[mutate_qubit][two_gates_idx+1] = new_control + 1
            curr_moment[new_control][two_gates_idx] = new_target + 1
            curr_moment[mutate_qubit][two_gates_idx] = 0
            curr_moment[new_control][two_gates_idx+1] = 0
    
    def turn_off(self, qubits_options, curr_moment, mutate_qubit, two_gates_idx, target=False):
        if target:
            curr_control = curr_moment[mutate_qubit][two_gates_idx] -1
            curr_moment[curr_control][two_gates_idx-1] = 0
            curr_moment[mutate_qubit][two_gates_idx] = 0
        else:
            curr_target = curr_moment[mutate_qubit][two_gates_idx] - 1
            curr_moment[curr_target][two_gates_idx+1] = 0
            curr_moment[mutate_qubit][two_gates_idx] = 0
    
    def mutate_one(self, qubits_options, curr_moment, mutate_qubit, two_gates_idx, target=False):
        new_qubit = random.choice(qubits_options)
        target_q = curr_moment[mutate_qubit][two_gates_idx]
        
        curr_moment[new_qubit][two_gates_idx] = target_q
        curr_moment[mutate_qubit][two_gates_idx] = 0
        if target:
            curr_moment[target_q-1][two_gates_idx-1] = new_qubit+1
        else:
            curr_moment[target_q-1][two_gates_idx+1] = new_qubit+1
        qubits_options.remove(new_qubit)
        
        
        
        
    
    