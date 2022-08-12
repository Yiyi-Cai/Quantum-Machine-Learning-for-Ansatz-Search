from itertools import cycle

from individual_circuit import *

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