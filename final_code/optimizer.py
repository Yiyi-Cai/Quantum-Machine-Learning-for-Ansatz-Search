from population import *

class CircuitOptimizer:
    
    def __init__(self, pop_size, target, num_one_gates, num_two_gates, max_moments, n_qubits):
        self.population = Population()
        self.population.fill_population(pop_size, target, num_one_gates, num_two_gates, max_moments, n_qubits)
        self.generation = 0
        self.pop_size = pop_size
        self.num_one_gates = num_one_gates
        self.num_two_gates = num_two_gates
        self.parents = []
        parents_size = int(pop_size * 0.3)
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
        