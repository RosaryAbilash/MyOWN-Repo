import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder




iter_X = 0
# Load your dataset here
# X = Features, y = Labels

path = "D:\IDS\IDS_data.csv"
data = pd.read_csv(path)

label_encoder = LabelEncoder()

X = data.drop(columns=[' Label'])
y = data[' Label']

y_encoded = label_encoder.fit_transform(y)
original_column_names = X.columns

# Parameters
duck_population_size = 11
num_features = X.shape[1]
max_iterations = 10
mutation_rate = 0.1
crossover_rate = 0.8

asn = 0
# Test Train Split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)



# Classifiers
classifiers = [
    RandomForestClassifier(n_estimators=100, random_state=42),
    SVC(kernel='linear'),
    KNeighborsClassifier(n_neighbors=5)
]




# Initialize duck population randomly
def initialize_duck_population(size, num_features):
    return np.random.randint(2, size=(size, num_features))

# Evaluate fitness (accuracy) of a feature subset
def evaluate_tastiness(feature_subset):
    global iter_X
    selected_feature_indices = np.where(feature_subset)[0]
    X_train_selected = X_train.iloc[:, selected_feature_indices]
    X_test_selected = X_test.iloc[:, selected_feature_indices]

    best_accuracy = 0
    best_classifier = None
    lst = []
    for clf in classifiers:
        print("Model = ", clf, "Best Accuracy : ", best_accuracy)
        clf.fit(X_train_selected, y_train)
        y_pred = clf.predict(X_test_selected)
        accuracy = accuracy_score(y_test, y_pred)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_classifier = clf
            lst.append(best_accuracy)
    iter_X += 1
    print("Best Accuracy List: ", lst)
    print("Iteration = ", iter_X, "Accuracy", accuracy)
    return best_accuracy



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
# Duck Optimization loop
def duck_optimization():
    duck_population = initialize_duck_population(duck_population_size, num_features)
    duck_wisdom = np.zeros(duck_population_size)  # Shared memory for tastiness
    prev_best_tastiness = 0.0  # Store previous best fitness value
    
    for iteration in range(max_iterations):
        tastiness_values = np.array([evaluate_tastiness(duck) for duck in duck_population])
        selected_indices = select_mating_ducks(tastiness_values)
        duck_population = preen_and_update_ducks(duck_population, selected_indices, crossover_rate, mutation_rate)
        update_duck_wisdom(duck_wisdom, duck_population, tastiness_values)
        
        best_duck_index = np.argmax(duck_wisdom)
        best_tastiness = duck_wisdom[best_duck_index]
        
        # Check for convergence or improvement threshold
        if best_tastiness - prev_best_tastiness < improvement_threshold:
            break  # Terminate if improvement is small
        prev_best_tastiness = best_tastiness
    
    best_duck_index = np.argmax(duck_wisdom)
    best_duck = duck_population[best_duck_index]
    best_tastiness = duck_wisdom[best_duck_index]
    
    selected_feature_indices = np.where(best_duck)[0]
    selected_column_names = original_column_names[selected_feature_indices]
    
    return selected_column_names, best_tastiness


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

# Set the improvement_threshold based on your problem
improvement_threshold = 0.001  # Example threshold, adjust as needed

# Run the Duck Optimization algorithm
selected_column_names, best_fitness = duck_optimization()

print("Best Feature Subsets (Duck):", selected_column_names)
print("Best Fitness (Accuracy):", best_fitness)








# Evaluate fitness (accuracy) of a feature subset using different classifiers
def evaluate_tastiness(feature_subset):
    selected_feature_indices = np.where(feature_subset)[0]
    X_train_selected = X_train.iloc[:, selected_feature_indices]
    X_test_selected = X_test.iloc[:, selected_feature_indices]

    best_accuracy = 0
    best_classifier = None
    
    for clf in classifiers:
        clf.fit(X_train_selected, y_train)
        y_pred = clf.predict(X_test_selected)
        accuracy = accuracy_score(y_test, y_pred)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_classifier = clf
    
    return best_accuracy
