import csv
import seaborn as sns
import itertools
import h5py

from optimizer import *
from file_reading import *
from plotting import *
from correlation import *



num_one_gates = 3
num_two_gates = 1
max_moments = 5
n_qubits = 8
pop_size = 40



"""
RUNNING GENERATION 0
"""


opt = CircuitOptimizer(pop_size, target_1d, num_one_gates, num_two_gates, max_moments, n_qubits)

write_header(pop_size)
plot_distribution(opt)
plot_and_draw_everything(opt)
plot_correlations(opt)
write_params_and_tensors(opt)



"""
GENETIC ALGORITHM
"""
# for i in range(2):
#     opt.generation += 1
#     opt.selection()
#     opt.crossover()
    
#     for i in range(5):
#         if random.random() < 0.7:
#             opt.mutate()
    
#     opt.train_offsprings()
#     draw_and_save_offsprings(opt)
#     opt.add_offsprings()
    
#     plot_distribution(opt)
#     plot_and_draw_everything(opt)
#     plot_correlations(opt)
#     write_params_and_tensors(opt)
#     opt.print_info()

curr_max_fitness = opt.population.fittest.get_fitness()
overall_max_fitness = curr_max_fitness
patience = 15
unchanged_num_gens = 0

while unchanged_num_gens < patience:
    opt.generation += 1
    opt.selection()
    opt.crossover()
    
    for i in range(5):
        if random.random() < 0.7:
            opt.mutate()
    
    opt.train_offsprings()
    draw_and_save_offsprings(opt)
    opt.add_offsprings()
    
    plot_distribution(opt)
    plot_and_draw_everything(opt)
    plot_correlations(opt)
    write_params_and_tensors(opt)
    opt.print_info()
    
    curr_max_fitness = opt.population.fittest.get_fitness()
    if curr_max_fitness > overall_max_fitness:
        overall_max_fitness = curr_max_fitness
        unchanged_num_gens = 0
    else:
        unchanged_num_gens +=1
        
"""
RESULTS PROCESSING
"""
plt.rcParams['font.size'] = '20'
if opt.generation > 10:
    plt.rcParams['font.size'] = '17'
if opt.generation > 20:
    plt.rcParams['font.size'] = '15'
if opt.generation > 30:
    plt.rcParams['font.size'] = '12'
if opt.generation > 50:
    plt.rcParams['font.size'] = '9'
name = folder + 'fitnesses_2d.csv'
plt.figure( figsize=(20, 14))
values = pd.read_csv(name)
values = values.transpose()
boxplot = values.boxplot() 
boxplot.set_ylabel('fitnesses')
boxplot.set_xlabel('generations')
plt.title("Population Fitnesses Per Generation")
plt.savefig(folder + 'fitnesses.png', bbox_inches='tight')



        
