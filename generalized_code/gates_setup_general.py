import random

from pennylane import broadcast
import pennylane as qml
from pennylane import numpy as np

qubits_2d = 8
# dev_wires_2d = [np.array(idx, requires_grad=True) for idx in range(qubits_2d)]
# dev_2d = qml.device('qulacs.simulator', wires=dev_wires_2d, gpu=True, shots=10000)


qubits_3d = 12
# dev_wires_3d = [np.array(idx, requires_grad=True) for idx in range(qubits_3d)]
# dev_3d = qml.device('qulacs.simulator', wires=dev_wires_3d, gpu=True, shots=10000)


### keys represent gates, values represent number of parameters required
two_qubits_gates_with_params = {qml.CNOT:0, qml.CRX:1, qml.CRY:1, qml.CRZ:1, qml.CRot:3}
one_qubit_gates_with_params = {qml.RX:1, qml.RY:1, qml.RZ:1, qml.PhaseShift:1, qml.Rot: 3}
two_qubits_gates = list(two_qubits_gates_with_params.keys())
one_qubit_gates = list(one_qubit_gates_with_params.keys())

one_q_gates = len(one_qubit_gates)
two_q_gates = len(two_qubits_gates)




def get_num_params(tensor, n_qubits, one_q_gates, two_q_gates):
    num_params = 0
    total_gates = one_q_gates + two_q_gates * 2
    num_opts = {}
    for moment in tensor:
        for q_idx in range(n_qubits):
            curr_qubit = moment[q_idx]
            for gate_idx in range(0, total_gates):
                if not curr_qubit[gate_idx] == 0:
                    if gate_idx < one_q_gates:
                        gate = one_qubit_gates[gate_idx]
                        curr_num_params = one_qubit_gates_with_params[gate]
                        num_params += curr_num_params
                    else:
                        two_q_idx = gate_idx - one_q_gates
                        if two_q_idx % 2 == 0:  
                            gate = two_qubits_gates[int(two_q_idx/2)]
                            target_q = curr_qubit[gate_idx] - 1
                            curr_num_params = two_qubits_gates_with_params[gate]
                            num_params += curr_num_params
                    break
    return num_params



def convert_tensor_to_circuit(tensor, params, n_qubits, one_q_gates, two_q_gates, dev_wires, prob=True):
    total_gates = one_q_gates + two_q_gates * 2
    params_idx = 0
    num_opts = {}
    for moment in tensor:
        for q_idx in range(n_qubits):
            curr_qubit = moment[q_idx]
            for gate_idx in range(total_gates):
                if not curr_qubit[gate_idx] == 0:
                    if gate_idx < one_q_gates:
                        gate = one_qubit_gates[gate_idx]
                        curr_num_params = one_qubit_gates_with_params[gate]
                        if curr_num_params == 0:
                            gate(wires=dev_wires[q_idx])
                        else:
                            broadcast(unitary=gate, pattern="single", wires=dev_wires[q_idx], parameters=[params[params_idx:params_idx+curr_num_params]])
                            params_idx += curr_num_params
                            
                    else:
                        two_q_idx = gate_idx - one_q_gates
                        if two_q_idx % 2 == 0:  
                            gate = two_qubits_gates[int(two_q_idx/2)]
                            target_idx = curr_qubit[gate_idx] -1
                            curr_num_params = two_qubits_gates_with_params[gate]
                            if curr_num_params == 0:
                                gate(wires=[dev_wires[q_idx], dev_wires[target_idx]])
                            else:
                                broadcast(unitary=gate, pattern="double", wires=[dev_wires[q_idx], dev_wires[target_idx]], parameters=[params[params_idx:params_idx+curr_num_params]])
                                params_idx += curr_num_params
                    break
                
    
    if prob:
        return qml.probs(wires=dev_wires[0:n_qubits])
    else:
        return qml.sample(wires=dev_wires[0:n_qubits])
    
    
    
def generate_random_tensor(n_qubits, one_q_gates, two_q_gates, max_moments):
    tensor = []
    total_gates = one_q_gates + two_q_gates * 2
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
            if rand_gates_idx < one_q_gates:
                rand_qubit_idx = random.choice(qubit_options)
                moment[rand_qubit_idx][rand_gates_idx] = 1
                qubit_options.remove(rand_qubit_idx)
            else:
                if len(qubit_options) >= 2:
                    rand_control, rand_target = random.sample(qubit_options, k = 2)
                    qubits_used = [rand_control, rand_target]
                    if all(x in qubit_options for x in qubits_used):
                        if (rand_gates_idx - one_q_gates) % 2 == 0:
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
