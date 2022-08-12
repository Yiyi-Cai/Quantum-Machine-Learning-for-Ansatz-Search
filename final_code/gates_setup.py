import random

from pennylane import broadcast
import pennylane as qml


### keys represent gates, values represent number of parameters required
two_qubits_gates_with_params = {qml.CNOT:0}
one_qubit_gates_with_params = {qml.RX:1, qml.RY:1, qml.RZ:1}
two_qubits_gates = list(two_qubits_gates_with_params.keys())
one_qubit_gates = list(one_qubit_gates_with_params.keys())


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



def convert_tensor_to_circuit(tensor, params, n_qubits, num_one_gates, num_two_gates, dev_wires, prob=True):
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
    
    