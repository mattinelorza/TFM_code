import numpy as np
import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Define the fuzzy variables and membership functions
proof_of_transit_result = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'proof_of_transit_result')
num_nodes = ctrl.Antecedent(np.arange(0, 11, 1), 'num_nodes')
node_attestation = ctrl.Antecedent(np.arange(0, 11, 1), 'node_attestation')
network_trust = ctrl.Consequent(np.arange(0, 11, 1), 'network_trust')

proof_of_transit_result['low'] = fuzz.trimf(proof_of_transit_result.universe, [0, 0, 0.5])
proof_of_transit_result['high'] = fuzz.trimf(proof_of_transit_result.universe, [0.5, 1, 1])

num_nodes['few'] = fuzz.trimf(num_nodes.universe, [0, 0, 5])
num_nodes['moderate'] = fuzz.trimf(num_nodes.universe, [5, 8, 10])
num_nodes['many'] = fuzz.trimf(num_nodes.universe, [8, 10, 10])

node_attestation['low'] = fuzz.trimf(node_attestation.universe, [0, 0, 5])
node_attestation['medium'] = fuzz.trimf(node_attestation.universe, [5, 7, 8])
node_attestation['high'] = fuzz.trimf(node_attestation.universe, [8, 10, 10])

network_trust['low'] = fuzz.trimf(network_trust.universe, [0, 0, 5])
network_trust['medium'] = fuzz.trimf(network_trust.universe, [5, 7.5, 9])
network_trust['high'] = fuzz.trimf(network_trust.universe, [8.5, 10, 10])

rules = [
    ctrl.Rule(proof_of_transit_result['low'] & num_nodes['few'] & node_attestation['low'], network_trust['low']),
    ctrl.Rule(proof_of_transit_result['low'] & num_nodes['few'] & node_attestation['medium'], network_trust['low']),
    ctrl.Rule(proof_of_transit_result['low'] & num_nodes['few'] & node_attestation['high'], network_trust['low']),
    
    ctrl.Rule(proof_of_transit_result['low'] & num_nodes['moderate'] & node_attestation['low'], network_trust['low']),
    ctrl.Rule(proof_of_transit_result['low'] & num_nodes['moderate'] & node_attestation['medium'], network_trust['low']),
    ctrl.Rule(proof_of_transit_result['low'] & num_nodes['moderate'] & node_attestation['high'], network_trust['medium']),
    
    ctrl.Rule(proof_of_transit_result['low'] & num_nodes['many'] & node_attestation['low'], network_trust['low']),
    ctrl.Rule(proof_of_transit_result['low'] & num_nodes['many'] & node_attestation['medium'], network_trust['medium']),
    ctrl.Rule(proof_of_transit_result['low'] & num_nodes['many'] & node_attestation['high'], network_trust['medium']),

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

network_trust_ctrl = ctrl.ControlSystem(rules)
network_trust_sim = ctrl.ControlSystemSimulation(network_trust_ctrl)

# Function to evaluate the fuzzy logic system
def evaluate_trust(proof_of_transit_result_value, num_nodes_value, node_attestation_value):
    network_trust_sim.input['proof_of_transit_result'] = proof_of_transit_result_value
    network_trust_sim.input['num_nodes'] = num_nodes_value
    network_trust_sim.input['node_attestation'] = node_attestation_value
    try:
        network_trust_sim.compute()
        return network_trust_sim.output['network_trust']
    except ValueError as e:
        print(f"Error: {e}")
        return None

# Generate dataset
np.random.seed(42)  # For reproducibility
n_samples = 100

# Generate random values for each input
proof_of_transit_results = np.random.uniform(0, 1, n_samples)
num_nodes_values = np.random.randint(0, 11, n_samples)
node_attestations = np.random.randint(0, 11, n_samples)

# Calculate network trust for each combination, ensuring valid results
data = []
for pot, nn, na in zip(proof_of_transit_results, num_nodes_values, node_attestations):
    trust = evaluate_trust(pot, nn, na)
    # Ensure valid trust value is added to the list
    data.append([pot, nn, na, trust])

# Create a DataFrame to store the dataset
df = pd.DataFrame(data, columns=['Proof_of_Transit_Result', 'Number_of_Nodes', 'Node_Attestation', 'Network_Trust'])

# Save the dataset to a CSV file
df.to_csv('network_trust_dataset.csv', index=False)

print("Dataset generated and saved as 'network_trust_dataset.csv'.")
