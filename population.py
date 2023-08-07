from itertools import cycle
import textdistance

from circuit import *

# qnode_2d = qml.QNode(convert_tensor_to_circuit, dev_2d)
# qnode_3d = qml.QNode(convert_tensor_to_circuit, dev_3d)
different_ratio = 50


import numpy

def levenshteinDistanceDP(token1, token2):
    distances = numpy.zeros((len(token1) + 1, len(token2) + 1))

    for t1 in range(len(token1) + 1):
        distances[t1][0] = t1

    for t2 in range(len(token2) + 1):
        distances[0][t2] = t2
        
    a = 0
    b = 0
    c = 0
    
    for t1 in range(1, len(token1) + 1):
        for t2 in range(1, len(token2) + 1):
            if (token1[t1-1] == token2[t2-1]):
                distances[t1][t2] = distances[t1 - 1][t2 - 1]
            else:
                a = distances[t1][t2 - 1]
                b = distances[t1 - 1][t2]
                c = distances[t1 - 1][t2 - 1]
                
                if (a <= b and a <= c):
                    distances[t1][t2] = a + 1
                elif (b <= a and b <= c):
                    distances[t1][t2] = b + 1
                else:
                    distances[t1][t2] = c + 1
    return distances[len(token1)][len(token2)]
        

class Population:
    
    def __init__(self):
        self.pop_size = 0
        self.individuals = []
        self.fittest = None
        self.least_fittest = None
        
    def fill_population(self, pop_size, target, num_one_gates, num_two_gates, max_moments, n_qubits, dimension):
        print('updated filling population')
        while len(self.individuals) < pop_size:
        # for i in range(pop_size):
            random_tensor = generate_random_tensor(n_qubits, num_one_gates, num_two_gates, max_moments)
            num_params_rand = get_num_params(random_tensor, n_qubits, one_q_gates, two_q_gates)
            init_params_rand = (2.*np.pi)*np.random.random(num_params_rand)
            num_moment = len(random_tensor)
            
            
            if dimension == 2:
                dev_wires_2d = [np.array(idx, requires_grad=True) for idx in range(qubits_2d)]
                dev_2d = qml.device('qulacs.simulator', wires=dev_wires_2d, gpu=True, shots=10000)
                qnode_2d = qml.QNode(convert_tensor_to_circuit, dev_2d)
                drawer = qml.draw(qnode_2d)
                dev_wires = dev_wires_2d
            elif dimension == 3:
                dev_wires_3d = [np.array(idx, requires_grad=True) for idx in range(qubits_3d)]
                dev_3d = qml.device('qulacs.simulator', wires=dev_wires_3d, gpu=True, shots=10000)
                qnode_3d = qml.QNode(convert_tensor_to_circuit, dev_3d)
                drawer = qml.draw(qnode_3d)
                dev_wires = dev_wires_3d
            
            circuit_rand = drawer(random_tensor, init_params_rand, n_qubits, one_q_gates, two_q_gates, dev_wires)
            circuit_rand_no_numbers = ''.join([i for i in circuit_rand if not (i.isdigit() or i == ',' or i == '.' or i == '(' or i == ')' or i == '\n')])
            circuit_rand_no_numbers = circuit_rand_no_numbers.replace("Probs", '')
            different = True            
            
            for j in range(self.pop_size):
                tensor1 = self.individuals[j].tensor
                num_params1 = get_num_params(tensor1, n_qubits, one_q_gates, two_q_gates)
                init_params1 = (2.*np.pi)*np.random.random(num_params1)
                
                circuit1 = drawer(tensor1, init_params1, n_qubits, one_q_gates, two_q_gates, dev_wires)
                circuit1_no_numbers = ''.join([i for i in circuit1 if not (i.isdigit() or i == ',' or i == '.' or i == '(' or i == ')' or i == '\n')])
                circuit1_no_numbers = circuit1_no_numbers.replace("Probs", '')
                
                # diff = abs(len(circuit1_no_numbers)-len(circuit_rand_no_numbers))
                # if diff < 30:
                #     threshold = different_ratio * (dimension + 4)
                # else:
                #     threshold = diff + different_ratio * (dimension-1)
                threshold = 500
                
                if levenshteinDistanceDP(circuit_rand_no_numbers, circuit1_no_numbers) < threshold:
                    different = False
                    print('repeating...')
                    # i -= 1
                    break
            
            if different:
                curr_circuit = IndividualCircuit(random_tensor, target, num_one_gates, num_two_gates, n_qubits, dimension)
                self.pop_size += 1
                self.individuals.append(curr_circuit)
                print('adding one circuit... total circuit ', self.pop_size)
            # else:
            #     i -= 1
        self.update()
        
        
#     def fill_population(self, pop_size, target, num_one_gates, num_two_gates, max_moments, n_qubits, dimension):       
        
#         for i in range(pop_size):
#             random_tensor = generate_random_tensor(n_qubits, num_one_gates, num_two_gates, max_moments)
#             num_params_rand = get_num_params(random_tensor, n_qubits, one_q_gates, two_q_gates)
#             init_params_rand = (2.*np.pi)*np.random.random(num_params_rand)
#             num_moment = len(random_tensor)
            
#             if dimension == 2:
#                 dev_wires_2d = [np.array(idx, requires_grad=True) for idx in range(qubits_2d)]
#                 dev_2d = qml.device('qulacs.simulator', wires=dev_wires_2d, gpu=True, shots=10000)
#                 qnode_2d = qml.QNode(convert_tensor_to_circuit, dev_2d)
#                 drawer = qml.draw(qnode_2d)
#                 dev_wires = dev_wires_2d
#             elif dimension == 3:
#                 dev_wires_3d = [np.array(idx, requires_grad=True) for idx in range(qubits_3d)]
#                 dev_3d = qml.device('qulacs.simulator', wires=dev_wires_3d, gpu=True, shots=10000)
#                 qnode_3d = qml.QNode(convert_tensor_to_circuit, dev_3d)
#                 drawer = qml.draw(qnode_3d)
#                 dev_wires = dev_wires_3d
            
#             circuit_rand = drawer(random_tensor, init_params_rand, n_qubits, one_q_gates, two_q_gates, dev_wires)
                
#             circuit_rand_no_numbers = ''.join([i for i in circuit_rand if not i.isdigit()])
#             different = True            
            
#             for j in range(self.pop_size):
#                 tensor1 = self.individuals[j].tensor
#                 num_params1 = get_num_params(tensor1, n_qubits, one_q_gates, two_q_gates)
#                 init_params1 = (2.*np.pi)*np.random.random(num_params1)
                
#                 circuit1 = drawer(tensor1, init_params1, n_qubits, one_q_gates, two_q_gates, dev_wires)
#                 circuit1_no_numbers = ''.join([i for i in circuit1 if not i.isdigit()])
                
#                 baseline = different_ratio * dimension + (num_moment-1) * different_ratio/2
            
#                 if textdistance.hamming.distance(circuit_rand_no_numbers, circuit1_no_numbers) < baseline:
#                     different = False
#                     break
            
#             if different:
#                 curr_circuit = IndividualCircuit(random_tensor, target, num_one_gates, num_two_gates, n_qubits, dimension)
#                 self.pop_size += 1
#                 self.individuals.append(curr_circuit)
#             else:
#                 i -= 1
#         self.update()
        
    
    
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
        self.update()
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
        # print('training all.. ')
        start_time = time.time()
        upper_limit = 500
        iterations = int(self.pop_size/upper_limit)
        new_individuals = []
        
        for j in range(iterations):
            args = []
            gpu_device = cycle(range(8))
            for i in range(upper_limit):
                args.append((self.individuals[i], str(next(gpu_device))))

            with mp.get_context("spawn").Pool(processes=upper_limit) as pool:
                results = pool.starmap(circuit_fitness, args)

            new_individuals += results
            
        remainder = self.pop_size % upper_limit

        args = []
        gpu_device = cycle(range(8))
        for i in range(iterations * upper_limit, iterations * upper_limit + remainder):
            args.append((self.individuals[i], str(next(gpu_device))))
            
        if remainder > 0:
            # print('args ', args)
            with mp.get_context("spawn").Pool(processes=remainder) as pool:
                results = pool.starmap(circuit_fitness, args)

            new_individuals += results
        
        self.individuals = new_individuals

        end_time = time.time()
        exec_time = end_time - start_time
        print("Execution time multiprocessing {}".format(exec_time))
        self.update()
                
    
    
    def get_fittests(self, num_fittests):
        """
        RETURN: copies of IndividualCircuit Objects
        """
        self.fittest = self.individuals[0]
        fittests = self.individuals[:num_fittests]
        return copy.deepcopy(fittests)