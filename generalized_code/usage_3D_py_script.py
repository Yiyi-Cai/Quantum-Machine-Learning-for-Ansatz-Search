from file_reading_general import *
from circuit import *
from population import *
from optimizer import *
from plotting import *
import h5py
import gc
import pickle 


max_moments = 10
pop_size = 50

opt = CircuitOptimizer(pop_size, target_1d_3, one_q_gates, two_q_gates, max_moments, qubits_3d, dim)

opt_folder = '../more_gates/'+ str(dim)+'d/distributions/'

opt.population.train_all()
opt.print_info()
if not os.path.exists(opt_folder):
    os.makedirs(opt_folder,exist_ok=True)
with open(opt_folder + 'curr_opt'+ str(opt.generation) + '.p', 'wb') as file:
    pickle.dump(opt, file)
    
write_header(pop_size)
plot_distribution(opt)

with open(opt_folder + 'curr_opt0.p', 'rb') as file:
    opt = pickle.load(file)
    
    
curr_max_fitness = opt.population.fittest.get_fitness()
overall_max_fitness = curr_max_fitness
patience = 16
unchanged_num_gens = 2

while unchanged_num_gens < patience:
    with open(opt_folder + 'curr_opt'+ str(opt.generation) + '.p', 'rb') as file:
        opt = pickle.load(file)
    opt.generation += 1
    opt.selection()
    opt.crossover()
    
    for i in range(6):
        if random.random() < 0.8:
            opt.mutate()
            
    opt.train_offsprings()
    opt.add_offsprings()
    
    with open(opt_folder + 'curr_opt'+ str(opt.generation) + '.p', 'wb') as file:
        pickle.dump(opt, file)
        
    opt.print_info()
    plot_distribution(opt)
    print('num params: ', len(opt.population.fittest.params))
    
    curr_max_fitness = opt.population.fittest.get_fitness()
    if curr_max_fitness > overall_max_fitness:
        overall_max_fitness = curr_max_fitness
        unchanged_num_gens = 0
    else:
        unchanged_num_gens +=1
    gc.collect()