import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# Define the fuzzy variables
proof_of_transit_result = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'proof_of_transit_result')
num_nodes = ctrl.Antecedent(np.arange(0, 11, 1), 'num_nodes')
network_attestation = ctrl.Consequent(np.arange(0, 11, 1), 'network_attestation')

# Define the membership functions for Proof of Transit Result
proof_of_transit_result['low'] = fuzz.trimf(proof_of_transit_result.universe, [0, 0, 0.5])
proof_of_transit_result['high'] = fuzz.trimf(proof_of_transit_result.universe, [0.5, 1, 1])

# Define the membership functions for Number of Nodes
num_nodes['few'] = fuzz.trimf(num_nodes.universe, [0, 0, 5])
num_nodes['moderate'] = fuzz.trimf(num_nodes.universe, [0, 9, 10])
num_nodes['many'] = fuzz.trimf(num_nodes.universe, [5, 10, 10])

# Define the membership functions for Network Attestation
network_attestation['low'] = fuzz.trimf(network_attestation.universe, [0, 0, 5])
network_attestation['medium'] = fuzz.trimf(network_attestation.universe, [0, 5, 10])
network_attestation['high'] = fuzz.trimf(network_attestation.universe, [9, 10, 10])

# Calculate the centroid for Network Attestation
def calculate_centroid(variable):
    membership_values = {
        'low': variable['low'].mf,
        'medium': variable['medium'].mf,
        'high': variable['high'].mf
    }
    x = variable.universe
    areas = {k: np.trapz(membership_values[k] * x, x) for k in membership_values}
    total_area = sum(areas.values())
    centroid = sum(areas[k] * np.mean(x) for k in areas) / total_area if total_area > 0 else np.nan
    return centroid, total_area

# Plot Membership Functions
def plot_membership_functions():
    fig, axs = plt.subplots(3, 1, figsize=(12, 15))

    # Plot Proof of Transit Result membership functions
    axs[0].plot(proof_of_transit_result.universe, proof_of_transit_result['low'].mf, label='Low')
    axs[0].plot(proof_of_transit_result.universe, proof_of_transit_result['high'].mf, label='High')
    axs[0].set_title('Proof of Transit Result Membership Functions')
    axs[0].set_xlabel('Proof of Transit Result')
    axs[0].set_ylabel('Membership Degree')
    axs[0].legend()

    # Plot Number of Nodes membership functions
    axs[1].plot(num_nodes.universe, num_nodes['few'].mf, label='Few')
    axs[1].plot(num_nodes.universe, num_nodes['moderate'].mf, label='Moderate')
    axs[1].plot(num_nodes.universe, num_nodes['many'].mf, label='Many')
    axs[1].set_title('Number of Nodes Membership Functions')
    axs[1].set_xlabel('Number of Nodes')
    axs[1].set_ylabel('Membership Degree')
    axs[1].legend()

    # Plot Network Attestation membership functions
    axs[2].plot(network_attestation.universe, network_attestation['low'].mf, label='Low')
    axs[2].plot(network_attestation.universe, network_attestation['medium'].mf, label='Medium')
    axs[2].plot(network_attestation.universe, network_attestation['high'].mf, label='High')
    axs[2].set_title('Network Attestation Membership Functions')
    axs[2].set_xlabel('Network Attestation')
    axs[2].set_ylabel('Membership Degree')
    axs[2].legend()

    # Calculate and plot centroid for Network Attestation
    centroid, total_area = calculate_centroid(network_attestation)
    print(f'Centroid of Network Attestation: {centroid:.2f}')
    print(f'Total Area under the curve: {total_area:.2f}')

    axs[2].axvline(centroid, color='r', linestyle='--', label=f'Centroid = {centroid:.2f}')
    axs[2].legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_membership_functions()
