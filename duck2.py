import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier






# Load your dataset here
# X = Features, y = Labels

path = "D:\IDS\IDS_data.csv"
data = pd.read_csv(path)

X = data.drop(columns=[' Label'])
y = data[' Label']

y = np.where(y == 'Benign', 0, 1)

# Parameters
duck_population_size = 50
num_features = X.shape[1]
max_iterations = 10
mutation_rate = 0.1
crossover_rate = 0.8

asn = 0
# Test Train Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize duck population randomly
def initialize_duck_population(size, num_features):
    return np.random.randint(2, size=(size, num_features))

# Evaluate fitness (accuracy) of a feature subset
def evaluate_tastiness(feature_subset):
    global asn
    selected_feature_indices = np.where(feature_subset)[0]
    X_train_selected = X_train.iloc[:, selected_feature_indices]
    X_test_selected = X_test.iloc[:, selected_feature_indices]

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_selected, y_train)

    accuracy = model.score(X_test_selected, y_test)
    asn += 1
    print("Evaluate Accuracy Iter ", asn)

    return accuracy



def let_ducks_wander(population):
    return initialize_duck_population(len(population), num_features)


def preen_and_update_ducks(ducks, selected_indices, crossover_rate, mutation_rate):
    new_generation = ducks.copy()
    num_parents = len(selected_indices)
    
    # Ensure an even number of parents for pairing
    if num_parents % 2 != 0:
        num_parents -= 1
    
    for i in range(0, num_parents, 2):
        parent1 = ducks[selected_indices[i]]
        parent2 = ducks[selected_indices[i + 1]]

        if np.random.rand() < crossover_rate:
            crossover_point = np.random.randint(1, num_features)
            new_generation[i, :crossover_point] = parent1[:crossover_point]
            new_generation[i + 1, crossover_point:] = parent2[crossover_point:]
        
        for j in range(num_features):
            if np.random.rand() < mutation_rate:
                new_generation[i, j] = 1 - new_generation[i, j]
            if np.random.rand() < mutation_rate:
                new_generation[i + 1, j] = 1 - new_generation[i + 1, j]
    
    return new_generation



# Duck Optimization loop
def duck_optimization():
    duck_population = initialize_duck_population(duck_population_size, num_features)
    duck_wisdom = np.zeros(duck_population_size)  # Shared memory for tastiness
    
    for iteration in range(max_iterations):
        tastiness_values = np.array([evaluate_tastiness(duck) for duck in duck_population])
        selected_indices = select_mating_ducks(tastiness_values)
        duck_population = preen_and_update_ducks(duck_population, selected_indices, crossover_rate, mutation_rate)
        update_duck_wisdom(duck_wisdom, duck_population, tastiness_values)
    
    best_duck_index = np.argmax(duck_wisdom)
    best_duck = duck_population[best_duck_index]
    best_tastiness = duck_wisdom[best_duck_index]
    
    return best_duck, best_tastiness

# Select ducks for mating based on tastiness (fitness) scores
def select_mating_ducks(tastiness_values):
    num_mating = len(tastiness_values) // 2
    selected_indices = np.argsort(tastiness_values)[-num_mating:]
    return selected_indices

# Update duck wisdom based on shared memory
def update_duck_wisdom(duck_wisdom, ducks, tastiness_values):
    for i, duck in enumerate(ducks):
        if tastiness_values[i] > duck_wisdom[i]:
            duck_wisdom[i] = tastiness_values[i]



# Run the Duck Optimization algorithm
best_duck, best_fitness = duck_optimization()

print("Best Feature Subset (Duck):", best_duck)
print("Best Fitness (Accuracy):", best_fitness)

# Let's train a model using the best feature subset
# best_duck_indices = np.where(best_duck == 1)[0]

# selected_feature_indices = np.where(best_duck == 1)[0]
# X_train_selected = X_train.iloc[:, best_duck_indices]
# X_test_selected = X_test.iloc[:, best_duck_indices]  # Use iloc for consistent indexing




# model = tf.keras.models.Sequential([
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dense(64, activation='relu'),
#     tf.keras.layers.Dense(1, activation='sigmoid')
# ])

# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# model.fit(X_train_selected, y_train, epochs=10, verbose=0)
# test_accuracy = model.evaluate(X_test_selected, y_test, verbose=0)[1]


# print("Model accuracy using the best feature subset:", test_accuracy)

