import os

import pennylane as qml
from sklearn.feature_selection import mutual_info_classif

from file_reading import *
from gates_setup import *

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


folder = '../optimizer_data/2D_run3/'
correlations_folder_name = folder + 'correlations/generation_'


states_mapping = [int(np.sum(np.asarray(s)*np.asarray([1,2,4,8,16,32,64,128]))) for s in list(itertools.product([0,1],repeat=8))]
def state_to_bin(state):
    #provided qubit state, returns the bin# it corresponds to
    return states_mapping.index(np.sum(np.asarray(state)*np.asarray([1,2,4,8,16,32,64,128])))
def qubits_to_var(sample):
    return bins_pt_mass[state_to_bin(sample)]
def qubits_to_pt(sample):
    return qubits_to_var(sample)[0]
def qubits_to_eta(sample):
    return qubits_to_var(sample)[1]


def plot_correlation(opt, circuit, index, device_number):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device_number)
    # title = "Generated Correlation"
    
    dev = qml.device('qulacs.simulator', wires=circuit.dev_wires, gpu=True, shots=10000)
    qnode = qml.QNode(convert_tensor_to_circuit, dev)

    sampling = qnode(circuit.tensor, circuit.params, circuit.n_qubits, circuit.num_one_gates, circuit.num_two_gates, circuit.dev_wires, prob=False)
    states_mapping = [int(np.sum(np.asarray(s)*np.asarray([1,2,4,8,16,32,64,128]))) for s in list(itertools.product([0,1],repeat=8))]

    sampling_pt_mass = np.apply_along_axis( qubits_to_var, 1, sampling)
    sampling_pt = sampling_pt_mass[:,0]
    sampling_mass = sampling_pt_mass[:,1]

    generated_df = pd.DataFrame(sampling_pt, columns=['jet_pt'])
    generated_df['jet_mass'] = sampling_mass
    
    

    directory = correlations_folder_name + str(opt.generation)
    if not os.path.exists(directory):
        os.makedirs(directory)

    gen_corr = generated_df.corr()

    test_j_pt = np.digitize(generated_df['jet_pt'], bins_pt_edge)
    test_j_mass = np.digitize(generated_df['jet_mass'], bins_mass_edge)

    
    sns.set(rc = {'figure.figsize':(15,15)})
    sns.set(font_scale = 2.5)   
    axis = sns.kdeplot( generated_df['jet_pt'],generated_df['jet_mass'], color = 'purple')
    axis.get_figure().savefig(directory + '/circuit_' + str(index+1) +'_correlation.png', bbox_inches='tight') 
    
    try:
        mutual_info = mutual_info_classif(test_j_pt[: ,np.newaxis], test_j_mass, discrete_features = True)
    except:
        gen_corr['mutual information, standard error)'] = [0,0]
        gen_corr.to_csv(directory + '/circuit_' + str(index+1) +'_correlation.csv')
    
    ### compute uncertainty of mutual info
    chunks = 10
    chunks_pt = np.reshape(test_j_pt, (chunks, int(len(test_j_pt)/chunks)))
    chunks_mass = np.reshape(test_j_mass, (chunks, int(len(test_j_mass)/chunks)))
    sub_mutual_infos = []
    for i in range(chunks):
        sub_mutual_infos.append(mutual_info_classif(chunks_pt[i][: ,np.newaxis], chunks_mass[i], discrete_features = True)[0])
    uncertainty = np.std(sub_mutual_infos)/np.sqrt(chunks)
        

    gen_corr['mutual information, standard error)'] = [mutual_info[0],uncertainty]
    gen_corr.to_csv(directory + '/circuit_' + str(index+1) +'_correlation.csv')



    
    
    