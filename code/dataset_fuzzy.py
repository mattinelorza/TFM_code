import numpy as np
import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# Variables Fuzzy 
proof_of_transit_result = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'proof_of_transit_result')
num_nodes = ctrl.Antecedent(np.arange(0, 11, 1), 'num_nodes')
node_attestation = ctrl.Antecedent(np.arange(0, 11, 1), 'node_attestation')
network_trust = ctrl.Consequent(np.arange(0, 11, 1), 'network_trust')

# Membership functions for Proof of Transit Result
proof_of_transit_result['failed'] = fuzz.trimf(proof_of_transit_result.universe, [0, 0, 0.5])
proof_of_transit_result['successful'] = fuzz.trimf(proof_of_transit_result.universe, [0.5, 1, 1])

# Membership functions for Number of Nodes
num_nodes['few'] = fuzz.trimf(num_nodes.universe, [0, 0, 5])
num_nodes['moderate'] = fuzz.trimf(num_nodes.universe, [0, 5, 8])
num_nodes['many'] = fuzz.trapmf(num_nodes.universe, [5, 8, 10,10])

# Membership functions for Node Attestation
node_attestation['low'] = fuzz.trimf(node_attestation.universe, [0, 0, 4])
node_attestation['medium'] = fuzz.trimf(node_attestation.universe, [0, 4, 8])
node_attestation['high'] = fuzz.trapmf(node_attestation.universe, [4, 8, 10,10])

# Membership functions for Network Trust
network_trust['very-low'] = fuzz.trimf(network_trust.universe, [0, 0, 3])
network_trust['low'] = fuzz.trimf(network_trust.universe, [0, 3, 5])
network_trust['medium'] = fuzz.trimf(network_trust.universe, [3, 5, 8])
network_trust['high'] = fuzz.trapmf(network_trust.universe, [5, 8,10, 10])

# Membership functions PoT
plt.figure(figsize=(10, 8))

plt.subplot(2, 2, 1)
plt.plot(proof_of_transit_result.universe, proof_of_transit_result['failed'].mf, label='Failed')
plt.plot(proof_of_transit_result.universe, proof_of_transit_result['successful'].mf, label='Successful')
plt.title('Proof of Transit Result')
plt.xlabel('Value')
plt.ylabel('Membership Degree')
plt.legend()

# Membership functions PoT
plt.subplot(2, 2, 2)
plt.plot(num_nodes.universe, num_nodes['few'].mf, label='Few')
plt.plot(num_nodes.universe, num_nodes['moderate'].mf, label='Moderate')
plt.plot(num_nodes.universe, num_nodes['many'].mf, label='Many')
plt.title('Number of Nodes with PoT')
plt.xlabel('Value')
plt.ylabel('Membership Degree')
plt.legend()

# Membership Node Attestation
plt.subplot(2, 2, 3)
plt.plot(node_attestation.universe, node_attestation['low'].mf, label='Low')
plt.plot(node_attestation.universe, node_attestation['medium'].mf, label='Medium')
plt.plot(node_attestation.universe, node_attestation['high'].mf, label='High')
plt.title('Number of Attested nodes')
plt.xlabel('Value')
plt.ylabel('Membership Degree')
plt.legend()

# Membership Network Trust
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


# Reglas Fuzzy
rules = [
    # Reglas failed Proof of Transit Result
    ctrl.Rule(proof_of_transit_result['failed'] & num_nodes['few'] & node_attestation['low'], network_trust['very-low']),
    ctrl.Rule(proof_of_transit_result['failed'] & num_nodes['few'] & node_attestation['medium'], network_trust['very-low']),
    ctrl.Rule(proof_of_transit_result['failed'] & num_nodes['few'] & node_attestation['high'], network_trust['very-low']),
    
    ctrl.Rule(proof_of_transit_result['failed'] & num_nodes['moderate'] & node_attestation['low'], network_trust['very-low']),
    ctrl.Rule(proof_of_transit_result['failed'] & num_nodes['moderate'] & node_attestation['medium'], network_trust['low']),
    ctrl.Rule(proof_of_transit_result['failed'] & num_nodes['moderate'] & node_attestation['high'], network_trust['medium']),
    
    ctrl.Rule(proof_of_transit_result['failed'] & num_nodes['many'] & node_attestation['low'], network_trust['low']),
    ctrl.Rule(proof_of_transit_result['failed'] & num_nodes['many'] & node_attestation['medium'], network_trust['low']),
    ctrl.Rule(proof_of_transit_result['failed'] & num_nodes['many'] & node_attestation['high'], network_trust['medium']),

    # Reglas successfull Proof of Transit Result
    ctrl.Rule(proof_of_transit_result['successful'] & num_nodes['few'] & node_attestation['low'], network_trust['medium']),
    ctrl.Rule(proof_of_transit_result['successful'] & num_nodes['few'] & node_attestation['medium'], network_trust['medium']),
    ctrl.Rule(proof_of_transit_result['successful'] & num_nodes['few'] & node_attestation['high'], network_trust['medium']),
    
    ctrl.Rule(proof_of_transit_result['successful'] & num_nodes['moderate'] & node_attestation['low'], network_trust['medium']),
    ctrl.Rule(proof_of_transit_result['successful'] & num_nodes['moderate'] & node_attestation['medium'], network_trust['medium']),
    ctrl.Rule(proof_of_transit_result['successful'] & num_nodes['moderate'] & node_attestation['high'], network_trust['high']),
    
    ctrl.Rule(proof_of_transit_result['successful'] & num_nodes['many'] & node_attestation['low'], network_trust['high']),
    ctrl.Rule(proof_of_transit_result['successful'] & num_nodes['many'] & node_attestation['medium'], network_trust['high']),
    ctrl.Rule(proof_of_transit_result['successful'] & num_nodes['many'] & node_attestation['high'], network_trust['high'])
]

network_trust_ctrl = ctrl.ControlSystem(rules)
network_trust_sim = ctrl.ControlSystemSimulation(network_trust_ctrl)

# Función para la evaluación del sistema de lógica difusa
# Se evalua implicitamente a un crisp value la salida del sistema de inferencia con el network_trust_sim.compute()
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

# Generación del dataset
np.random.seed(42) 
n_samples = 100 # 1000 muestras, valor representativo

# Generación de valores por cada input
proof_of_transit_results = np.random.randint(0, 2, n_samples)
num_nodes_values = np.random.randint(0, 11, n_samples)
node_attestations = np.random.randint(0, 11, n_samples)

# Calculo del network trust para el parámetro 1
dataset = []
for pot, nn, na in zip(proof_of_transit_results, num_nodes_values, node_attestations):
    trust = evaluate_trust(pot, nn, na)
    if trust is not None:
        pot_text = "successful" if pot == 1 else "not successful"
        dataset.append([pot_text, nn, na, trust])


# Creación del DataFrame para almacenar el dataset y almacenamiento en un excel
df = pd.DataFrame(dataset, columns=['Proof_of_Transit_Result', 'Number_of_Nodes', 'Node_Attestation', 'Network_Trust'])
df.to_excel('network_trust_dataset.xlsx', index=False)
print("Dataset generado 'network_trust_dataset.xlsx'.")

# Crear la gráfica separada successfull not successfull
plt.figure(figsize=(10, 6))

# Separar los datos en "successful" y "not successful"
successful = df[df['Proof_of_Transit_Result'] == "successful"]
not_successful = df[df['Proof_of_Transit_Result'] == "not successful"]

# Crear puntos 
y_successful = successful['Network_Trust']
y_not_successful = not_successful['Network_Trust']
x_successful = np.random.normal(loc=2, scale=0.04, size=len(y_successful))
x_not_successful = np.random.normal(loc=1, scale=0.04, size=len(y_not_successful))

# Dibujar puntos
plt.scatter(x_not_successful, y_not_successful, color='red', label='not successful', alpha=0.6)
plt.scatter(x_successful, y_successful, color='blue', label='successful', alpha=0.6)

# Configurar ejes y título
plt.xticks([1, 2], ['not successful', 'successful'])
plt.xlabel('Proof of Transit Result')
plt.ylabel('Network Trust')
plt.title('Network Trust by Proof of Transit Result')
plt.legend()


plt.show()

# Mostrar la gráfica conjunta successfull not successfull
plt.figure(figsize=(10, 6))
colors = df['Proof_of_Transit_Result'].map({'successful': 'blue', 'not successful': 'red'})

# Crear puntos en el eje Y
y_values = df['Network_Trust']

# Crear el eje X con índices de los puntos
x_values = range(len(y_values))

# Dibujar puntos
plt.scatter(x_values, y_values, color=colors, alpha=0.6)

unique_labels = df['Proof_of_Transit_Result'].unique()
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[df['Proof_of_Transit_Result'] == label].iloc[0], markersize=10) for label in unique_labels]

plt.legend(handles, unique_labels, title="Proof of Transit Result",loc='upper left')

# Etiquetas y título
plt.xlabel('PoT results')
plt.ylabel('Network Trust')
plt.title('Network Trust by Proof of Transit Result')
plt.xticks([])  # Ocultar los valores del eje X

plt.show()

# Crear la tercera gráfica
plt.figure(figsize=(10, 6))

# Number of Nodes vs Network Trust
plt.scatter(df['Number_of_Nodes'], df['Network_Trust'], color='blue', alpha=0.6)

# Etiquetas y título
plt.xlabel('Number of Nodes with PoT')
plt.ylabel('Network Trust')
plt.title('Network Trust by Number of Nodes with PoT')

plt.show()

# Crear la cuarta gráfica
plt.figure(figsize=(10, 6))

# Number of Nodes vs Network Trust
plt.scatter(df['Node_Attestation'], df['Network_Trust'], color='black', alpha=0.6)

# Etiquetas y título
plt.xlabel('Number of attested Nodes')
plt.ylabel('Network Trust')
plt.title('Network Trust by Attested Nodes')


plt.show()

