import os

import pennylane as qml
from sklearn.feature_selection import mutual_info_classif

from file_reading_general import *
from gates_setup_general import *
from dimension_setting import *

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


folder = '../more_gates/' + str(dim)+'d/'
correlations_folder_name = folder + 'correlations/generation_'

if dim==2:
    n_qubits = 8
elif dim == 3:
    n_qubits = 12
    

def compute_uncertainty(var1, var2, chunks=10):
    chunks = 10
    chunks_pt = np.reshape(var1, (chunks, int(len(var1)/chunks)))
    chunks_mass = np.reshape(var2, (chunks, int(len(var2)/chunks)))
    sub_mutual_infos = []
    for i in range(chunks):
        sub_mutual_infos.append(mutual_info_classif(chunks_pt[i][: ,np.newaxis], chunks_mass[i], discrete_features = True)[0])
    uncertainty = np.std(sub_mutual_infos)/np.sqrt(chunks)
    return uncertainty

states_mapping = [int(np.sum(np.asarray(s)*np.asarray([2**iq for iq in range(n_qubits)]))) for s in list(itertools.product([0,1],repeat=n_qubits))]

def state_to_bin(state):
    #provided qubit state, returns the bin# it corresponds to
    return states_mapping.index(np.sum(np.asarray(state)*np.asarray([2**iq for iq in range(n_qubits)])))


def qubits_to_var(sample):
    if dim == 2:
        return bins_pt_mass[state_to_bin(sample)]
    else:
        return bins_pt_mass_nsubj[state_to_bin(sample)]
def qubits_to_pt(sample):
    return qubits_to_var(sample)[0]
def qubits_to_mass(sample):
    return qubits_to_var(sample)[1]
def qubits_to_nsubj(sample):
    return qubits_to_var(sample)[2]


def plot_correlation(opt, circuit, index, device_number):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device_number)

    
    dev = qml.device('qulacs.simulator', wires=circuit.dev_wires, gpu=True, shots=10000)
    qnode = qml.QNode(convert_tensor_to_circuit, dev)

    
    sampling = qnode(circuit.tensor, circuit.params, circuit.n_qubits, circuit.one_q_gates, circuit.two_q_gates, circuit.dev_wires, prob=False)
    if dim == 2:
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

        sns.set(font_scale = 2)
        g=sns.PairGrid(generated_df)
        g.fig.set_size_inches(30,30)
        g.map_diag(sns.histplot,color='pink')
        g.map_upper(sns.histplot, color = '#C8A2C8')
        g.map_lower(sns.kdeplot,color='purple')
        
    
        g.fig.savefig(directory + '/circuit_' + str(index+1) +'_correlation.pdf', bbox_inches='tight') 
        
        sns.set(rc = {'figure.figsize':(15,15)})
        sns.set(font_scale = 2.5)   
        axis = sns.kdeplot( generated_df['jet_pt'],generated_df['jet_mass'], color = 'purple')
        axis.get_figure().savefig(directory + '/circuit_' + str(index+1) +'_correlation.pdf', bbox_inches='tight') 

        try:
            pt_mass_mutual_info = mutual_info_classif(test_j_pt[: ,np.newaxis], test_j_mass, discrete_features = True)
            pt_mass_uncert = compute_uncertainty(test_j_pt[: ,np.newaxis], test_j_mass)
        except:
            # gen_corr['uncertainty'] = [0,0]
            # gen_corr.to_csv(directory + '/circuit_' + str(index+1) +'_correlation.csv')
            return 
            
            
        MI_df = pd.DataFrame({'jet pt': [pt_mass_mutual_info, pt_mass_uncert]})

        MI_df.index = ['jet mass', 'uncertainty']
        MI_df.to_csv(directory + '/circuit_' + str(index+1) +'_correlation.csv')

    
    elif dim == 3:
        sampling_pt_mass_nsubj = np.apply_along_axis( qubits_to_var, 1, sampling)

        sampling_pt = sampling_pt_mass_nsubj[:,0]
        sampling_mass = sampling_pt_mass_nsubj[:,1]
        sampling_nsubj = sampling_pt_mass_nsubj[:,2]
        generated_df = pd.DataFrame(sampling_pt, columns=['jet pt'])
        generated_df['n-subjecttiness'] = sampling_nsubj
        generated_df['jet mass'] = sampling_mass

        sns.set(font_scale = 2)

        g=sns.PairGrid(generated_df)
        g.fig.set_size_inches(30,30)
        g.map_diag(sns.histplot,color='pink')
        g.map_upper(sns.histplot, color = '#C8A2C8')
        g.map_lower(sns.kdeplot,color='purple')



        directory = correlations_folder_name + str(opt.generation)
        if not os.path.exists(directory):
            os.makedirs(directory)

        g.fig.savefig(directory + '/circuit_' + str(index+1) +'_correlation.pdf', bbox_inches='tight') 


        test_j_pt = np.digitize(generated_df['jet pt'], bins_pt_edge)
        test_j_mass = np.digitize(generated_df['jet mass'], bins_mass_edge)
        test_j_tau = np.digitize(generated_df['n-subjecttiness'], bins_tau_edge)

        try:
            tau_pt_MI = mutual_info_classif(test_j_tau[: ,np.newaxis], test_j_pt, discrete_features = True)[0]
            tau_pt_uncert = compute_uncertainty(test_j_tau[: ,np.newaxis], test_j_pt)
            mass_pt_MI = mutual_info_classif(test_j_mass[: ,np.newaxis], test_j_pt, discrete_features = True)[0]
            mass_pt_uncert = compute_uncertainty(test_j_mass[: ,np.newaxis], test_j_mass)
            tau_mass_MI = mutual_info_classif(test_j_tau[: ,np.newaxis], test_j_mass, discrete_features = True)[0]
            tau_mass_uncert = compute_uncertainty(test_j_tau[: ,np.newaxis], test_j_mass)
        except: 
            return

        MI_df = pd.DataFrame({'jet pt': [mass_pt_MI, mass_pt_uncert, tau_pt_MI, tau_pt_uncert], 'jet mass': ['', '',  tau_mass_MI, tau_mass_uncert]})

        MI_df.index = ['jet mass', 'uncertainty', 'n-subjettiness', 'uncertainties']

        MI_df.to_csv(directory + '/circuit_' + str(index+1) +'_correlation.csv')



    
    
    