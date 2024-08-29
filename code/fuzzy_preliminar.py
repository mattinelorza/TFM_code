import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Define the fuzzy variables
proof_of_transit_result = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'proof_of_transit_result')
num_nodes = ctrl.Antecedent(np.arange(0, 11, 1), 'num_nodes')
node_attestation = ctrl.Antecedent(np.arange(0, 11, 1), 'node_attestation')
network_trust = ctrl.Consequent(np.arange(0, 11, 1), 'network_trust')

# Define the membership functions for Proof of Transit Result
proof_of_transit_result['low'] = fuzz.trimf(proof_of_transit_result.universe, [0, 0, 0.5])
proof_of_transit_result['high'] = fuzz.trimf(proof_of_transit_result.universe, [0.5, 1, 1])

# Define the membership functions for Number of Nodes
num_nodes['few'] = fuzz.trimf(num_nodes.universe, [0, 0, 5])
num_nodes['moderate'] = fuzz.trimf(num_nodes.universe, [0, 5, 8])
num_nodes['many'] = fuzz.trimf(num_nodes.universe, [5, 10, 10])

# Define the membership functions for Node Attestation // puede haber nodos que no soporten atestación
node_attestation['low'] = fuzz.trimf(node_attestation.universe, [0, 0, 4])
node_attestation['medium'] = fuzz.trimf(node_attestation.universe, [0, 4, 8])
node_attestation['high'] = fuzz.trimf(node_attestation.universe, [4, 10, 10])

# Define the membership functions for Network Trust
network_trust['very-low'] = fuzz.trimf(network_trust.universe, [0, 0, 3])
network_trust['low'] = fuzz.trimf(network_trust.universe, [0, 3, 5])
network_trust['medium'] = fuzz.trimf(network_trust.universe, [3, 5, 10])
network_trust['high'] = fuzz.trimf(network_trust.universe, [5, 10, 10])

# Define the fuzzy rules
rules = [
    # Rules for low Proof of Transit Result
    ctrl.Rule(proof_of_transit_result['low'] & num_nodes['few'] & node_attestation['low'], network_trust['very-low']),
    ctrl.Rule(proof_of_transit_result['low'] & num_nodes['few'] & node_attestation['medium'], network_trust['very-low']),
    ctrl.Rule(proof_of_transit_result['low'] & num_nodes['few'] & node_attestation['high'], network_trust['very-low']),
    
    ctrl.Rule(proof_of_transit_result['low'] & num_nodes['moderate'] & node_attestation['low'], network_trust['very-low']),
    ctrl.Rule(proof_of_transit_result['low'] & num_nodes['moderate'] & node_attestation['medium'], network_trust['low']),
    ctrl.Rule(proof_of_transit_result['low'] & num_nodes['moderate'] & node_attestation['high'], network_trust['medium']),
    
    ctrl.Rule(proof_of_transit_result['low'] & num_nodes['many'] & node_attestation['low'], network_trust['low']),
    ctrl.Rule(proof_of_transit_result['low'] & num_nodes['many'] & node_attestation['medium'], network_trust['low']),
    ctrl.Rule(proof_of_transit_result['low'] & num_nodes['many'] & node_attestation['high'], network_trust['medium']),

    # Rules for high Proof of Transit Result
    ctrl.Rule(proof_of_transit_result['high'] & num_nodes['few'] & node_attestation['low'], network_trust['medium']),
    ctrl.Rule(proof_of_transit_result['high'] & num_nodes['few'] & node_attestation['medium'], network_trust['medium']),
    ctrl.Rule(proof_of_transit_result['high'] & num_nodes['few'] & node_attestation['high'], network_trust['high']),
    
    ctrl.Rule(proof_of_transit_result['high'] & num_nodes['moderate'] & node_attestation['low'], network_trust['medium']),
    ctrl.Rule(proof_of_transit_result['high'] & num_nodes['moderate'] & node_attestation['medium'], network_trust['medium']),
    ctrl.Rule(proof_of_transit_result['high'] & num_nodes['moderate'] & node_attestation['high'], network_trust['high']),
    
    ctrl.Rule(proof_of_transit_result['high'] & num_nodes['many'] & node_attestation['low'], network_trust['high']),
    ctrl.Rule(proof_of_transit_result['high'] & num_nodes['many'] & node_attestation['medium'], network_trust['high']),
    ctrl.Rule(proof_of_transit_result['high'] & num_nodes['many'] & node_attestation['high'], network_trust['high'])
]

# Create the control system
network_trust_ctrl = ctrl.ControlSystem(rules)
network_trust_sim = ctrl.ControlSystemSimulation(network_trust_ctrl)

# Function to evaluate the fuzzy logic system
def evaluate_trust(proof_of_transit_result_value, num_nodes_value, node_attestation_value):
    network_trust_sim.input['proof_of_transit_result'] = proof_of_transit_result_value
    network_trust_sim.input['num_nodes'] = num_nodes_value
    network_trust_sim.input['node_attestation'] = node_attestation_value
    network_trust_sim.compute()
    return network_trust_sim.output['network_trust']

# Main function to get user input and evaluate trust
def main():
    try:
        proof_of_transit_result_value = float(input("Enter the Proof of Transit Result (0-1): "))
        num_nodes_value = float(input("Enter the Number of Nodes using Proof of Transit (0-10): "))
        node_attestation_value = float(input("Enter the Number of Nodes that are attested (0-10): "))

        if not (0 <= proof_of_transit_result_value <= 1) or not (0 <= num_nodes_value <= 10) or not (0 <= node_attestation_value <= 10):
            print("Please enter values within the range 0 to 1 for Proof of Transit, 0 to 10 for Number of Nodes using Proof of Transit, and 0 to 10 for Number of Nodes that are attested.")
            return
        
        # Calculate the trust based on all three inputs
        trust = evaluate_trust(proof_of_transit_result_value, num_nodes_value, node_attestation_value)
        print(f'For PoT result = {proof_of_transit_result_value}, number of nodes = {num_nodes_value}, and node attestation = {node_attestation_value}, the network trust is {trust:.2f}')
    except ValueError:
        print("Invalid input. Please enter numerical values between 0 and 1 for Proof of Transit, and between 0 and 10 for Number of Nodes and Node Attestation.")

if __name__ == "__main__":
    main()
