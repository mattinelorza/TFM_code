import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
from skfuzzy import control as ctrl



# Variables Fuzzy
proof_of_transit_result = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'proof_of_transit_result')
num_nodes = ctrl.Antecedent(np.arange(0, 11, 1), 'num_nodes')
node_attestation = ctrl.Antecedent(np.arange(0, 11, 1), 'node_attestation')
network_trust = ctrl.Consequent(np.arange(0, 11, 1), 'network_trust')

# Proof of Transit Result
proof_of_transit_result['failed'] = fuzz.trimf(proof_of_transit_result.universe, [0, 0, 0.5])
proof_of_transit_result['successful'] = fuzz.trimf(proof_of_transit_result.universe, [0.5, 1, 1])

# Number of Nodes
num_nodes['few'] = fuzz.trimf(num_nodes.universe, [0, 0, 5])
num_nodes['moderate'] = fuzz.trimf(num_nodes.universe, [0, 5, 8])
num_nodes['many'] = fuzz.trapmf(num_nodes.universe, [5, 8, 10,10])

# Node Attestation
node_attestation['low'] = fuzz.trimf(node_attestation.universe, [0, 0, 4])
node_attestation['medium'] = fuzz.trimf(node_attestation.universe, [0, 4, 8])
node_attestation['high'] = fuzz.trapmf(node_attestation.universe, [4, 8, 10,10])

# Network Trust
network_trust['very-low'] = fuzz.trimf(network_trust.universe, [0, 0, 3])
network_trust['low'] = fuzz.trimf(network_trust.universe, [0, 3, 5])
network_trust['medium'] = fuzz.trimf(network_trust.universe, [3, 5, 8])
network_trust['high'] = fuzz.trapmf(network_trust.universe, [5, 8,10, 10])


# Plot Proof of Transit Result
plt.figure(figsize=(10, 8))

plt.subplot(2, 2, 1)
plt.plot(proof_of_transit_result.universe, proof_of_transit_result['failed'].mf, label='Failed')
plt.plot(proof_of_transit_result.universe, proof_of_transit_result['successful'].mf, label='Successful')
plt.title('Proof of Transit Result')
plt.xlabel('Value')
plt.ylabel('Membership Degree')
plt.legend()

# Plot Number of Nodes
plt.subplot(2, 2, 2)
plt.plot(num_nodes.universe, num_nodes['few'].mf, label='Few')
plt.plot(num_nodes.universe, num_nodes['moderate'].mf, label='Moderate')
plt.plot(num_nodes.universe, num_nodes['many'].mf, label='Many')
plt.title('Number of Nodes with PoT')
plt.xlabel('Value')
plt.ylabel('Membership Degree')
plt.legend()

# Plot Node Attestation
plt.subplot(2, 2, 3)
plt.plot(node_attestation.universe, node_attestation['low'].mf, label='Low')
plt.plot(node_attestation.universe, node_attestation['medium'].mf, label='Medium')
plt.plot(node_attestation.universe, node_attestation['high'].mf, label='High')
plt.title('Number of Attested nodes')
plt.xlabel('Value')
plt.ylabel('Membership Degree')
plt.legend()

# Plot Network Trust
plt.subplot(2, 2, 4)
plt.plot(network_trust.universe, network_trust['very-low'].mf, label='Very-low')
plt.plot(network_trust.universe, network_trust['low'].mf, label='Low')
plt.plot(network_trust.universe, network_trust['medium'].mf, label='Medium')
plt.plot(network_trust.universe, network_trust['high'].mf, label='High')
plt.title('Network LoT')
plt.xlabel('Value')
plt.ylabel('Membership Degree')
plt.legend()

plt.tight_layout()
plt.show()
