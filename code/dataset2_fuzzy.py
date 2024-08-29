import numpy as np
import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# Variables Fuzzy 2
direct_reputation = ctrl.Antecedent(np.arange(0, 11, 1), 'direct_reputation')
indirect_reputation = ctrl.Antecedent(np.arange(0, 11, 1), 'indirect_reputation')
node_integrity = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'node_integrity')
node_reputation = ctrl.Consequent(np.arange(0, 11, 1), 'node_reputation')

# Membership functions for Direct reputation
direct_reputation['very-bad'] = fuzz.trapmf(direct_reputation.universe, [0, 0, 1,3])
direct_reputation['bad'] = fuzz.trimf(direct_reputation.universe, [1, 3, 5])
direct_reputation['neutral'] = fuzz.trimf(direct_reputation.universe, [3, 5, 7])
direct_reputation['good'] = fuzz.trimf(direct_reputation.universe, [5, 7, 9])
direct_reputation['very-good'] = fuzz.trapmf(direct_reputation.universe, [7, 9, 10,10])

# Membership functions for Indirect reputation
indirect_reputation['bad'] = fuzz.trapmf(indirect_reputation.universe, [0, 0, 2, 5])
indirect_reputation['neutral'] = fuzz.trimf(indirect_reputation.universe, [2, 5, 8])
indirect_reputation['good'] = fuzz.trapmf(indirect_reputation.universe, [5, 8, 10,10])

# Membership functions for node_integrity
node_integrity['not guaranteed'] = fuzz.trimf(node_integrity.universe, [0, 0, 0.5])
node_integrity['guaranteed'] = fuzz.trimf(node_integrity.universe, [0.5, 1, 1])

# Membership functions for Node reputation
node_reputation['very-bad'] = fuzz.trimf(node_reputation.universe, [0, 0, 3])
node_reputation['bad'] = fuzz.trimf(node_reputation.universe, [0, 3, 5])
node_reputation['neutral'] = fuzz.trimf(node_reputation.universe, [3, 5, 8])
node_reputation['good'] = fuzz.trapmf(node_reputation.universe, [5, 8,10, 10])

# Membership functions Direct Reputation
plt.figure(figsize=(10, 8))

plt.subplot(2, 2, 1)
plt.plot(direct_reputation.universe, direct_reputation['very-bad'].mf, label='Very-bad')
plt.plot(direct_reputation.universe, direct_reputation['bad'].mf, label='Bad')
plt.plot(direct_reputation.universe, direct_reputation['neutral'].mf, label='Neutral')
plt.plot(direct_reputation.universe, direct_reputation['good'].mf, label='Good')
plt.plot(direct_reputation.universe, direct_reputation['very-good'].mf, label='Very-good')
plt.title('Direct reputation')
plt.xlabel('Value')
plt.ylabel('Membership Degree')
plt.legend(bbox_to_anchor=(0, 0.5), loc='center left')

# Membership functions Indirect reputation
plt.subplot(2, 2, 2)
plt.plot(indirect_reputation.universe, indirect_reputation['bad'].mf, label='Bad')
plt.plot(indirect_reputation.universe, indirect_reputation['neutral'].mf, label='Neutral')
plt.plot(indirect_reputation.universe, indirect_reputation['good'].mf, label='Good')
plt.title('Indirect reputation')
plt.xlabel('Value')
plt.ylabel('Membership Degree')
plt.legend()

# Membership functions Node integrity
plt.subplot(2, 2, 3)
plt.plot(node_integrity.universe, node_integrity['not guaranteed'].mf, label='Not guaranteed')
plt.plot(node_integrity.universe, node_integrity['guaranteed'].mf, label='Guaranteed')
plt.title('Node integrity')
plt.xlabel('Value')
plt.ylabel('Membership Degree')
plt.legend()

# Membership functions Node reputation
plt.subplot(2, 2, 4)
plt.plot(node_reputation.universe, node_reputation['very-bad'].mf, label='Very-bad')
plt.plot(node_reputation.universe, node_reputation['bad'].mf, label='Bad')
plt.plot(node_reputation.universe, node_reputation['neutral'].mf, label='Neutral')
plt.plot(node_reputation.universe, node_reputation['good'].mf, label='Good')
plt.title('Node reputation')
plt.xlabel('Value')
plt.ylabel('Membership Degree')
plt.legend()

plt.tight_layout()
plt.show()


# Reglas Fuzzy
rules = [
    ctrl.Rule(direct_reputation['very-bad'] & indirect_reputation['bad'] & node_integrity['not guaranteed'], node_reputation['very-bad']),
    ctrl.Rule(direct_reputation['very-bad'] & indirect_reputation['bad'] & node_integrity['guaranteed'], node_reputation['very-bad']),
    ctrl.Rule(direct_reputation['very-bad'] & indirect_reputation['neutral'] & node_integrity['not guaranteed'], node_reputation['very-bad']),
    ctrl.Rule(direct_reputation['very-bad'] & indirect_reputation['neutral'] & node_integrity['guaranteed'], node_reputation['very-bad']),
    ctrl.Rule(direct_reputation['very-bad'] & indirect_reputation['good'] & node_integrity['not guaranteed'], node_reputation['very-bad']),
    ctrl.Rule(direct_reputation['very-bad'] & indirect_reputation['good'] & node_integrity['guaranteed'], node_reputation['bad']),

    ctrl.Rule(direct_reputation['bad'] & indirect_reputation['bad'] & node_integrity['not guaranteed'], node_reputation['very-bad']),
    ctrl.Rule(direct_reputation['bad'] & indirect_reputation['bad'] & node_integrity['guaranteed'], node_reputation['bad']),
    ctrl.Rule(direct_reputation['bad'] & indirect_reputation['neutral'] & node_integrity['not guaranteed'], node_reputation['very-bad']),
    ctrl.Rule(direct_reputation['bad'] & indirect_reputation['neutral'] & node_integrity['guaranteed'], node_reputation['bad']),
    ctrl.Rule(direct_reputation['bad'] & indirect_reputation['good'] & node_integrity['not guaranteed'], node_reputation['bad']),
    ctrl.Rule(direct_reputation['bad'] & indirect_reputation['good'] & node_integrity['guaranteed'], node_reputation['bad']),

    ctrl.Rule(direct_reputation['neutral'] & indirect_reputation['bad'] & node_integrity['not guaranteed'], node_reputation['bad']),
    ctrl.Rule(direct_reputation['neutral'] & indirect_reputation['bad'] & node_integrity['guaranteed'], node_reputation['neutral']),
    ctrl.Rule(direct_reputation['neutral'] & indirect_reputation['neutral'] & node_integrity['not guaranteed'], node_reputation['bad']),
    ctrl.Rule(direct_reputation['neutral'] & indirect_reputation['neutral'] & node_integrity['guaranteed'], node_reputation['neutral']),
    ctrl.Rule(direct_reputation['neutral'] & indirect_reputation['good'] & node_integrity['not guaranteed'], node_reputation['neutral']),
    ctrl.Rule(direct_reputation['neutral'] & indirect_reputation['good'] & node_integrity['guaranteed'], node_reputation['neutral']),

    ctrl.Rule(direct_reputation['good'] & indirect_reputation['bad'] & node_integrity['not guaranteed'], node_reputation['bad']),
    ctrl.Rule(direct_reputation['good'] & indirect_reputation['bad'] & node_integrity['guaranteed'], node_reputation['neutral']),
    ctrl.Rule(direct_reputation['good'] & indirect_reputation['neutral'] & node_integrity['not guaranteed'], node_reputation['bad']),
    ctrl.Rule(direct_reputation['good'] & indirect_reputation['neutral'] & node_integrity['guaranteed'], node_reputation['neutral']),
    ctrl.Rule(direct_reputation['good'] & indirect_reputation['good'] & node_integrity['not guaranteed'], node_reputation['bad']),
    ctrl.Rule(direct_reputation['good'] & indirect_reputation['good'] & node_integrity['guaranteed'], node_reputation['neutral']),

    ctrl.Rule(direct_reputation['very-good'] & indirect_reputation['bad'] & node_integrity['not guaranteed'], node_reputation['neutral']),
    ctrl.Rule(direct_reputation['very-good'] & indirect_reputation['bad'] & node_integrity['guaranteed'], node_reputation['good']),
    ctrl.Rule(direct_reputation['very-good'] & indirect_reputation['neutral'] & node_integrity['not guaranteed'], node_reputation['neutral']),
    ctrl.Rule(direct_reputation['very-good'] & indirect_reputation['neutral'] & node_integrity['guaranteed'], node_reputation['good']),
    ctrl.Rule(direct_reputation['very-good'] & indirect_reputation['good'] & node_integrity['not guaranteed'], node_reputation['neutral']),
    ctrl.Rule(direct_reputation['very-good'] & indirect_reputation['good'] & node_integrity['guaranteed'], node_reputation['good']),
]

node_reputation_ctrl = ctrl.ControlSystem(rules)
node_reputation_sim = ctrl.ControlSystemSimulation(node_reputation_ctrl)

# Función para la evaluación del sistema de lógica difusa
# Se evalua implicitamente a un crisp value la salida del sistema de inferencia con el node_reputation_sim.compute()
def evaluate_reputation(node_integrity_value, direct_reputation_value, indirect_reputation_value):
    node_reputation_sim.input['node_integrity'] = node_integrity_value
    node_reputation_sim.input['direct_reputation'] = direct_reputation_value
    node_reputation_sim.input['indirect_reputation'] = indirect_reputation_value
    try:
        node_reputation_sim.compute()
        return node_reputation_sim.output['node_reputation']
    except ValueError as e:
        print(f"Error: {e}")
        return None

# Generación del dataset
np.random.seed(42) 
n_samples = 1000 # 1000 muestras, valor representativo

# Generación de valores por cada input
node_integrity = np.random.randint(0, 2, n_samples)
direct_reputation_values = np.random.randint(0, 11, n_samples)
indirect_reputations = np.random.randint(0, 11, n_samples)

# Calculo del node reputation para el parámetro 2
dataset = []
for intg, dr, ir in zip(node_integrity, direct_reputation_values, indirect_reputations):
    reputation = evaluate_reputation(intg, dr, ir)
    if reputation is not None:
        integrity_text = "Guaranteed" if intg == 1 else "Not Guaranteed"
        dataset.append([integrity_text, dr, ir, reputation])


# Creación del DataFrame para almacenar el dataset y almacenamiento en un excel
df = pd.DataFrame(dataset, columns=['node_integrity', 'direct_reputation', 'indirect_reputation', 'node_reputation'])
df.to_excel('node_reputation_dataset.xlsx', index=False)
print("Dataset generado 'node_reputation_dataset.xlsx'.")


# Colores para los diferentes grados de reputación directa
colors = {
    'very-bad': 'red',
    'bad': 'orange',
    'neutral': 'yellow',
    'good': 'green',
    'very-good': 'blue'
}

# Asignar categorías a cada valor de direct_reputation
def get_reputation_category(dr):
    if dr <= 2:
        return 'very-bad'
    elif dr <= 4:
        return 'bad'
    elif dr <= 6:
        return 'neutral'
    elif dr <= 8:
        return 'good'
    else:
        return 'very-good'

# Asignar categorías a cada valor de direct_reputation en el DataFrame
df['direct_reputation_category'] = df['direct_reputation'].apply(get_reputation_category)

# Crear la gráfica
plt.figure(figsize=(10, 6))

# Dibujar puntos para cada categoría de reputación directa
for category, color in colors.items():
    subset = df[df['direct_reputation_category'] == category]
    x_values = np.random.normal(loc=2, scale=0.04, size=len(subset))  # Distribución normal para separación
    plt.scatter(x_values, subset['node_reputation'], color=color, label=f'{category}', alpha=0.6)

# Ejes y título
plt.xlabel('Direct Reputation')
plt.ylabel('Node Reputation')
plt.title('Node Reputation by Direct Reputation')
plt.xticks([])
plt.legend(title='Direct Reputation Categories')

plt.show()

# Colores para los diferentes niveles de reputación directa
colors = {
    
    'bad': 'orange',
    'neutral': 'purple',
    'good': 'green',
    
}

# Asignar categorías a cada valor de direct_reputation
def get_reputation_category(dr):
    if dr <= 4:
        return 'bad'
    elif dr <= 6:
        return 'neutral'
    else:
        return 'good'

# Asignar categorías a cada valor de direct_reputation en el DataFrame
df['indirect_reputation_category'] = df['indirect_reputation'].apply(get_reputation_category)

# Crear la gráfica
plt.figure(figsize=(10, 6))

# Dibujar puntos para cada categoría de reputación directa
for category, color in colors.items():
    subset = df[df['indirect_reputation_category'] == category]
    x_values = np.random.normal(loc=2, scale=0.04, size=len(subset))  # Distribución normal para separación
    plt.scatter(x_values, subset['node_reputation'], color=color, label=f'{category}', alpha=0.6)

# Ejes y título
plt.xlabel('Indirect Reputation')
plt.ylabel('Node Reputation')
plt.title('Node Reputation by Indirect Reputation')
plt.xticks([])
plt.legend(title='Indirect Reputation Categories')

plt.show()


# Gráfica separada successfull not successfull
plt.figure(figsize=(10, 6))

# Separar los datos en "successful" y "not successful"
successful = df[df['node_integrity'] == "Guaranteed"]
not_successful = df[df['node_integrity'] == "Not Guaranteed"]

# Crear puntos 
y_successful = successful['node_reputation']
y_not_successful = not_successful['node_reputation']
x_successful = np.random.normal(loc=2, scale=0.04, size=len(y_successful))
x_not_successful = np.random.normal(loc=1, scale=0.04, size=len(y_not_successful))

# Dibujar puntos
plt.scatter(x_not_successful, y_not_successful, color='red', label='Not Guaranteed', alpha=0.6)
plt.scatter(x_successful, y_successful, color='blue', label='Guaranteed', alpha=0.6)

# Etiquetas y título
plt.xticks([1, 2], ['not guaranteed', 'guaranteed'])
plt.xlabel('Node Integrity')
plt.ylabel('Node Reputation')
plt.title('Node Reputation by node integrity')
plt.legend()


plt.show()



