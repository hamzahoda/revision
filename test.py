-----------------------------------------
A STAR
----------------------------------------

import heapq

def a_star_search(graph, start, goal, heuristic):
    open_list = [(0, start)]  # Priority queue with f-score and node
    closed_list = set()
    g_scores = {node: float('inf') for node in graph}
    g_scores[start] = 0
    parents = {}

    while open_list:
        _, current_node = heapq.heappop(open_list)

        if current_node == goal:
            return reconstruct_path(parents, start, goal)

        closed_list.add(current_node)

        if current_node in graph:
            for neighbor, edge_cost in graph[current_node].items():
                if neighbor in closed_list:
                    continue

                new_g_score = g_scores[current_node] + edge_cost

                if new_g_score < g_scores[neighbor] or neighbor not in [node for _, node in open_list]:
                    g_scores[neighbor] = new_g_score
                    parents[neighbor] = current_node
                    f_score = new_g_score + heuristic[neighbor]
                    heapq.heappush(open_list, (f_score, neighbor))

    return None

def reconstruct_path(parents, start, goal):
    path = [goal]
    current_node = goal

    while current_node != start:
        current_node = parents[current_node]
        path.append(current_node)

    path.reverse()
    return path

# Example usage:
graph = {
    'Arad': {'Zerind': 75, 'Timisoara': 118, 'Sibiu': 140},
    'Zerind': {'Arad': 75, 'Oradea': 71},
    'Oradea': {'Zerind': 71, 'Sibiu': 151},
    'Timisoara': {'Arad': 118, 'Lugoj': 111},
    'Lugoj': {'Timisoara': 111, 'Mehadia': 70},
    'Mehadia': {'Lugoj': 70, 'Drobeta': 75},
    'Drobeta': {'Mehadia': 75, 'Craiova': 120},
    'Craiova': {'Drobeta': 120, 'Rimnicu': 146, 'Pitesti': 138},
    'Rimnicu': {'Craiova': 146, 'Sibiu': 80, 'Pitesti': 97},
    'Sibiu': {'Arad': 140, 'Oradea': 151, 'Rimnicu': 80, 'Fagaras': 99},
    'Fagaras': {'Sibiu': 99, 'Bucharest': 211},
    'Pitesti': {'Rimnicu': 97, 'Craiova': 138, 'Bucharest': 101},
    'Bucharest': {'Fagaras': 211, 'Pitesti': 101, 'Giurgiu': 90, 'Urziceni': 85},
    'Giurgiu': {'Bucharest': 77},
    'Urziceni': {'Bucharest': 85, 'Hirsova': 98, 'Vaslui': 142},
    'Hirsova': {'Urziceni': 98, 'Eforie': 86},
    'Eforie': {'Hirsova': 86},
    'Vaslui': {'Urziceni': 142, 'Iasi': 92},
    'Iasi': {'Vaslui': 92, 'Neamt': 87},
    'Neamt': {'Iasi': 87}
}

heuristic = {
    'Arad': 366,
    'Bucharest': 0,
    'Craiova': 160,
    'Drobeta': 242,
    'Eforie': 161,
    'Fagaras': 178,
    'Giurgiu': 90,
    'Hirsova': 151,
    'Iasi': 226,
    'Lugoj': 244,
    'Mehadia': 241,
    'Neamt': 234,
    'Oradea': 380,
    'Pitesti': 98,
    'Rimnicu': 193,
    'Sibiu': 253,
    'Timisoara': 329,
    'Urziceni': 80,
    'Vaslui': 199,
    'Zerind': 374
}

start_node = 'Arad'
goal_node = 'Bucharest'

shortest_path = a_star_search(graph, start_node, goal_node, heuristic)
if shortest_path:
    print("Shortest path:", shortest_path)
else:
    print("No path found from", start_node, "to", goal_node)








*******************************************************************************
DFS BFS GREEDY
*******************************************************************************
import heapq


def dfs(graph, start, goal):
    visited = set()
    stack = [start]
    print("Start", start)
    while stack:
        node = stack.pop()
        print("Visiting", node)
        if node == goal:
            print("Reached Goal")
            return True
        if node not in visited:
            visited.add(node)
            stack.extend(graph[node])
    return False


def bfs(graph, start, goal):
    visited = set()
    queue = [start]

    while queue:
        node = queue.pop(0)

        if node == goal:
            print("Reached Goal")
            return True

        if node not in visited:
            visited.add(node)
            if node in graph:
                queue.extend(graph[node])

    return False


def greedy_search(graph, start, goal, heuristic):
    visited = set()
    # Priority queue using heuristic value as priority
    queue = [(heuristic[start], start)]
    print("Start", start)
    while queue:
        _, node = heapq.heappop(queue)
        print("Visiting", node)
        if node == goal:
            print("Reached Goal", visited)
            return True

        if node not in visited:
            visited.add(node)

            if node in graph:  # Check if node exists in the graph dictionary
                for neighbor in graph[node]:
                    if neighbor not in visited:
                        heapq.heappush(queue, (heuristic[neighbor], neighbor))

    return False


graph = {
    "A": ["B", "D"],
    "B": ["C", "E"],
    "C": [],
    "D": ["E", "G"],
    "E": ["C", "F"],
    "F": [],
    "G": []
}

heuristic = {
    "A": 5,
    "B": 4,
    "C": 2,
    "D": 3,
    "E": 2,
    "F": 1,
    "G": 0
}

print(greedy_search(graph, "A", "G", heuristic))
# print(dfs(graph, "A", "G"))




*******************************************************************************
EVOLTUION
*******************************************************************************

import random

# Problem-specific evaluation function
def evaluate_solution(solution):
    # Calculate fitness or objective value of the solution
    # Return the fitness value
    return sum(solution)

# Evolutionary Algorithm
def evolutionary_algorithm(population_size, num_generations):
    # Generate an initial population of random solutions
    population = [random.choices([0, 1], k=10) for _ in range(population_size)]

    for generation in range(num_generations):
        # Evaluate fitness of each solution in the population
        fitness_values = [evaluate_solution(solution) for solution in population]

        # Select parents for reproduction (e.g., tournament selection)
        parents = random.choices(population, weights=fitness_values, k=population_size)

        # Create offspring through crossover (e.g., one-point crossover)
        offspring = []
        for i in range(population_size):
            parent1 = random.choice(parents)
            parent2 = random.choice(parents)
            crossover_point = random.randint(0, len(parent1))
            child = parent1[:crossover_point] + parent2[crossover_point:]
            offspring.append(child)

        # Apply mutation to the offspring (e.g., bit-flip mutation)
        for i in range(population_size):
            for j in range(len(offspring[i])):
                if random.random() < mutation_rate:
                    offspring[i][j] = 1 - offspring[i][j]

        # Replace the current population with the offspring
        population = offspring

    # Return the best solution found
    best_solution = max(population, key=lambda x: evaluate_solution(x))
    return best_solution

# Example usage
population_size = 1000
num_generations = 50
mutation_rate = 0.01

best_solution = evolutionary_algorithm(population_size, num_generations)
print("Best solution:", best_solution)
print("Fitness:", evaluate_solution(best_solution))



*******************************************************************************
Genetic Algo
*******************************************************************************
#GENETIC ALGO SIMPLEST
import random

# Genetic Algorithm Parameters
population_size = 50
chromosome_length = 10
generations = 100

# Generate Initial Population
population = []
for _ in range(population_size):
    lst = []
    for _ in range(chromosome_length):
        lst.append(random.randint(0,1))
    population.append(lst)

# Fitness Function (Modify according to your problem)
def fitness_function(chromosome):
    target = [1, 1, 0, 1, 0, 0, 1, 0, 1, 0]  # Example target chromosome
    return sum(c1 == c2 for c1, c2 in zip(chromosome, target))

# Genetic Algorithm
for _ in range(generations):
    population = sorted(population, key=fitness_function, reverse=True)
    parents = population[:population_size // 2]

    new_population = []
    for parent1, parent2 in zip(parents[::2], parents[1::2]):
        
        crossover_point = 4
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        mutated_child1 = child1[:3] + [1-child1[3]] + child1[4:]
        mutated_child2 = child2[:3] + [1-child2[3]] + child2[4:]
        new_population.extend([mutated_child1, mutated_child2])

    population = new_population

# Find the best chromosome in the final population
best_solution = max(population, key=fitness_function)

print("Best Solution:", best_solution)
print("Fitness:", fitness_function(best_solution))

*******************************************************************************
PSO
*******************************************************************************
#Particle Swarm Optimization WITH ERROR
import random

# Particle Swarm Optimization Parameters
num_particles = 50
num_dimensions = 10
max_iterations = 100
c1 = 2.0  # Cognitive parameter
c2 = 2.0  # Social parameter
w = 0.7   # Inertia weight

# Initialize Particle Positions, Velocities, and Best Positions
particles = [[random.uniform(0, 1) for _ in range(num_dimensions)] for _ in range(num_particles)]
velocities = [[random.uniform(-1, 1) for _ in range(num_dimensions)] for _ in range(num_particles)]
best_positions = particles.copy()

# Fitness Function (Modify according to your problem)
def fitness_function(position):
    target = [1, 1, 0, 1, 0, 0, 1, 0, 1, 0]  # Example target position
    return sum(c1 == c2 for c1, c2 in zip(position, target))

# Particle Swarm Optimization
global_best_position = None
global_best_fitness = float('-inf')

for _ in range(max_iterations):
    for i in range(num_particles):
        particle = particles[i]
        velocity = velocities[i]
        best_position = best_positions[i]

        # Update Particle Velocity
        for j in range(num_dimensions):
            r1 = random.uniform(0, 1)
            r2 = random.uniform(0, 1)
            velocity[j] = (w * velocity[j]) + (c1 * r1 * (best_position[j] - particle[j])) + (c2 * r2 * (global_best_position[j] - particle[j]))

        # Update Particle Position
        for j in range(num_dimensions):
            particle[j] = particle[j] + velocity[j]

        # Update Best Positions
        fitness = fitness_function(particle)
        if fitness > fitness_function(best_position):
            best_positions[i] = particle

        # Update Global Best
        if fitness > global_best_fitness:
            global_best_fitness = fitness
            global_best_position = particle

# Print the Best Solution and its Fitness
print("Best Solution:", global_best_position)
print("Fitness:", global_best_fitness)

******************************************************************************
ACO
*****************************************************************************
import random

# Ant Colony Optimization Parameters
num_ants = 50
num_iterations = 100
alpha = 1.0  # Pheromone factor
beta = 2.0   # Heuristic factor
evaporation_rate = 0.5

# Graph representation (Modify according to your problem)
graph = [
    [0, 2, 4, 1],
    [2, 0, 1, 5],
    [4, 1, 0, 3],
    [1, 5, 3, 0]
]

num_cities = len(graph)

# Initialize Pheromone Matrix
pheromone = [[1.0 for _ in range(num_cities)] for _ in range(num_cities)]

# Ant Colony Optimization
best_path = None
best_distance = float('inf')

for _ in range(num_iterations):
    paths = []

    # Construct Solutions
    for _ in range(num_ants):
        start_city = random.randint(0, num_cities - 1)
        path = [start_city]
        visited = [False] * num_cities
        visited[start_city] = True

        for _ in range(num_cities - 1):
            current_city = path[-1]
            next_city = None
            probabilities = []

            # Compute Probabilities for the Next City
            for city in range(num_cities):
                if not visited[city]:
                    pheromone_value = pheromone[current_city][city]
                    heuristic_value = 1.0 / graph[current_city][city]
                    probability = pheromone_value * alpha * heuristic_value * beta
                    probabilities.append((city, probability))

            total_probability = sum(prob for _, prob in probabilities)
            probabilities = [(city, prob / total_probability) for city, prob in probabilities]

            # Choose Next City based on Probability
            random_value = random.uniform(0, 1)
            cumulative_probability = 0.0
            for city, probability in probabilities:
                cumulative_probability += probability
                if random_value <= cumulative_probability:
                    next_city = city
                    break

            path.append(next_city)
            visited[next_city] = True

        paths.append(path)

    # Update Pheromone Matrix
    for i in range(num_cities):
        for j in range(num_cities):
            if i != j:
                pheromone[i][j] *= (1 - evaporation_rate)

    for path in paths:
        distance = sum(graph[path[i]][path[i+1]] for i in range(num_cities - 1))
        if distance < best_distance:
            best_distance = distance
            best_path = path

        for i in range(num_cities - 1):
            pheromone[path[i]][path[i+1]] += 1.0 / distance

# Print the Best Path and its Distance
print("Best Path:", best_path)
print("Distance:", best_distance)

*******************************************************************************
HILL CLIMB
*******************************************************************************

import random

def objective_function(x, y):
    """
    Objective function to evaluate the state.
    """
    return (1 - x)**2 + 100*(y - x**2)**2

def generate_neighbor(x, y):
    """
    Generates a random neighbor state by making a small change to the current state.
    """
    neighbor_x = x + random.uniform(-0.1, 0.1)
    neighbor_y = y + random.uniform(-0.1, 0.1)
    return neighbor_x, neighbor_y

def hill_climbing(initial_x, initial_y):
    """
    Hill Climbing algorithm implementation.
    """
    current_x = initial_x
    current_y = initial_y
    current_score = objective_function(current_x, current_y)
    
    while True:
        neighbor_x, neighbor_y = generate_neighbor(current_x, current_y)
        neighbor_score = objective_function(neighbor_x, neighbor_y)
        
        if neighbor_score < current_score:
            current_x = neighbor_x
            current_y = neighbor_y
            current_score = neighbor_score
        else:
            break
    
    return current_x, current_y, current_score

# Example usage
initial_x = random.uniform(-2, 2)
initial_y = random.uniform(-2, 2)
final_x, final_y, final_score = hill_climbing(initial_x, initial_y)
print("Final State (x, y):", final_x, final_y)
print("Final Score:", final_score)


*******************************************************************************
LAB 4 
*******************************************************************************
import random

def drawBoard(board):
    print('   |   |')
    print(' ' + board[7] + ' | ' + board[8] + ' | ' + board[9])
    print('   |   |')
    print('-----------')
    print('   |   |')
    print(' ' + board[4] + ' | ' + board[5] + ' | ' + board[6])
    print('   |   |')
    print('-----------')
    print('   |   |')
    print(' ' + board[1] + ' | ' + board[2] + ' | ' + board[3])
    print('   |   |')

def inputPlayerLetter():
    return ['X', 'O']

def whoGoesFirst():
    return random.choice(['computer', 'player'])

def playAgain():
    return input('Do you want to play again? (yes or no) ').lower().startswith('y')

def makeMove(board, letter, move):
    board[move] = letter

def isWinner(board, letter):
    return ((board[7] == letter and board[8] == letter and board[9] == letter) or 
    (board[4] == letter and board[5] == letter and board[6] == letter) or 
    (board[1] == letter and board[2] == letter and board[3] == letter) or 
    (board[7] == letter and board[4] == letter and board[1] == letter) or 
    (board[8] == letter and board[5] == letter and board[2] == letter) or 
    (board[9] == letter and board[6] == letter and board[3] == letter) or 
    (board[7] == letter and board[5] == letter and board[3] == letter) or 
    (board[9] == letter and board[5] == letter and board[1] == letter))

def getBoardCopy(board):
    return board[:]

def isSpaceFree(board, move):
    return board[move] == ' '

def getPlayerMove(board):
    move = ' '
    while move not in '1 2 3 4 5 6 7 8 9'.split() or not isSpaceFree(board, int(move)):
        move = input('What is your next move? (1-9) ')
    return int(move)

def chooseRandomMoveFromList(board, movesList):
    possibleMoves = []
    for i in movesList:
        if isSpaceFree(board, i):
            possibleMoves.append(i)
    if len(possibleMoves) != 0:
        return random.choice(possibleMoves)
    else:
        return None
    
def getComputerMove(board, computerLetter):
    if computerLetter == 'X':
        playerLetter = 'O'
    else:
        playerLetter = 'X'

    for i in range(1, 10):
        copy = getBoardCopy(board)
        if isSpaceFree(copy, i):
            makeMove(copy, computerLetter, i)
            if isWinner(copy, computerLetter):
                return i

    for i in range(1, 10):
        copy = getBoardCopy(board)
        if isSpaceFree(copy, i):
            makeMove(copy, playerLetter, i)
            if isWinner(copy, playerLetter):
                return i

    move = chooseRandomMoveFromList(board, [1, 3, 7, 9])
    if move != None:
        return move

    if isSpaceFree(board, 5):
        return 5

    return chooseRandomMoveFromList(board, [2, 4, 6, 8])

def isBoardFull(board):
    for i in range(1, 10):
        if isSpaceFree(board, i):
            return False
    return True

print('Welcome to Tic Tac Toe!')

while True:
    theBoard = [' '] * 10
    playerLetter, computerLetter = inputPlayerLetter()
    turn = whoGoesFirst()
    print('The ' + turn + ' will go first.')
    gameIsPlaying = True

    while gameIsPlaying:
        if turn == 'player':
            drawBoard(theBoard)
            move = getPlayerMove(theBoard)
            makeMove(theBoard, playerLetter, move)
            if isWinner(theBoard, playerLetter):
                drawBoard(theBoard)
                print('You have won the game!')
                gameIsPlaying = False
            else:
                if isBoardFull(theBoard):
                    drawBoard(theBoard)
                    print('The game is a tie!')
                    break
                else:
                    turn = 'computer'
        else:
            move = getComputerMove(theBoard, computerLetter)
            makeMove(theBoard, computerLetter, move)
            if isWinner(theBoard, computerLetter):
                drawBoard(theBoard)
                print('The computer has beaten you! You lose.')
                gameIsPlaying = False
            else:
                if isBoardFull(theBoard):
                    drawBoard(theBoard)
                    print('The game is a tie!')
                    break
                else:
                    turn = 'player'

    if not playAgain():
        break


*******************************************************************************
KNN
*******************************************************************************
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris = load_iris()
x = iris.data
y = iris.target
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, random_state=42)
num_neighbors = [1, 3, 5, 7, 9]
accuracies = []

for k in num_neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(xtrain, ytrain)
    ypred = knn.predict(xtest)
    accuracy = accuracy_score(ytest, ypred)
    accuracies.append(accuracy)
    print(k, accuracy)

# Plotting the results
plt.plot(num_neighbors, accuracies, marker='o')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Accuracy')
plt.title('Accuracy of KNN for Different Values of k')
plt.show()

*******************************************************************************
Naive Bias simple
*******************************************************************************
#NAIEVE BIAS WALA
import pandas as pd, numpy as np

ds =  pd.read_excel("weatherTemp.xlsx")
print(ds.head())
x = ds.iloc[:, 0:2].values #for input values 
y = ds.iloc[:,2].values #for output value

#Import LabelEncoder

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
x[:,0] = le.fit_transform(x[:,0])
x[:,1] = le.fit_transform(x[:,1])
y = le.fit_transform(y)


print ("Weather:", x[:,0])
print("Temp ",x[:,1])
print("Play ",y)

#Import Gaussian Naive Bayes model

from sklearn.naive_bayes import GaussianNB

#Create & Gaussian Classifier

model  = GaussianNB()

model.fit(x, y)

#predict Output

predicted  = model.predict([[2,1]])
print(predicted)

*******************************************************************************
Naiv bias Big
*******************************************************************************
#NAIVE BIAS + f1_score, accuracy_score METRICS

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score, accuracy_score

url = "dermatology.csv"
column_names = ["erythema", "scaling", "definite borders", "itching", "koebner phenomenon", "polygonal papules", "follicular papules", "oral mucosal involvement", "knee and elbow involvement", "scalp involvement", "family history", "melanin incontinence", "eosinophils in the infiltrate", "PNL infiltrate", "fibrosis of the papillary dermis", "exocytosis", "acanthosis",
                "hyperkeratosis", "parakeratosis", "clubbing of the rete ridges", "elongation of the rete ridges", "thinning of the suprapapillary epidermis", "spongiform pustule", "munro microabcess", "focal hypergranulosis", "disappearance of the granular layer", "vacuolisation and damage of basal layer", "spongiosis",
                "saw-tooth appearance of retes", "follicular horn plug", "perifollicular parakeratosis", "inflammatory monoluclear inflitrate", "band-like infiltrate",
                "Age", "class"]

ds = pd.read_csv(url, names=column_names)
ds = ds.drop(columns=["erythema"])
ds = ds.replace("?", float("NaN"))
ds = ds.dropna()

x = ds.iloc[:, :-1]
y = ds.iloc[:, -1]


model = GaussianNB()
k = 5

scores = cross_val_score(model, x, y, cv=k)
mean_accuracy = scores.mean()
std_accuracy = scores.std()

print(f"Mean Accuracy (k={k}): {mean_accuracy}")
print(f"Standard Deviation of Accuracy (k={k}): {std_accuracy}")
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,
                                                    random_state=42)


model.fit(x_train, y_train)


y_pred = model.predict(x_test)

accuracy = model.score(x_test, y_test)
a = accuracy_score()
confusion = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:")
print(confusion)

*******************************************************************************
LINEAR REGRESSION
*******************************************************************************


# Import the necessary libraries
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# Input data - X represents the independent variable and y represents the dependent variable
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 5, 4, 5])

# Create an instance of the Linear Regression model
model = LinearRegression()

# Train the model using the input data
model.fit(X, y)

# Predict the output for a new input value
new_X = np.array([[6]])
predicted_y = model.predict(new_X)

# Calculate the error rate
mae = mean_absolute_error(y, model.predict(X))

# Visualize the data and the regression line
plt.scatter(X, y, color='blue', label='Actual')
plt.plot(X, model.predict(X), color='red', label='Regression Line')
plt.scatter(new_X, predicted_y, color='green', label='Predicted')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression')
plt.legend()
plt.show()

# Print the predicted output and error rate
print("Predicted y:", predicted_y)
print("Mean Absolute Error:", mae)


******************************************************************************
    SVM + Confusion matric
******************************************************************************
#SVM + CONFUSION MATRIX 

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

# Step 1: Download the dataset and load it into a DataFrame
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt'
df = pd.read_csv(url, header=None)

# Step 2: Preprocess the data (if necessary)
# Assuming the dataset is already preprocessed and does not require further preprocessing.

# Step 3: Split the dataset into features (X) and labels (y)
X = df.iloc[:, :-1]  # Features are all columns except the last one
y = df.iloc[:, -1]   # Labels are the last column

# Step 4: Apply Support Vector Machine (SVM) algorithm
# Split the data into train and test sets using a 70-30 split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create an instance of SVM classifier
svm_classifier = SVC()

# Fit the classifier to the training data
svm_classifier.fit(X_train, y_train)

# Predict the labels for the test data
y_pred = svm_classifier.predict(X_test)

# Step 5: Evaluate the performance - Confusion Matrix
confusion_mat = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(confusion_mat)

# Step 6: Perform k-fold cross-validation
# Assuming k = 5 for demonstration purposes
k = 5
cross_val_scores = cross_val_score(svm_classifier, X, y, cv=k)
avg_accuracy = cross_val_scores.mean()
print("Average Accuracy:", avg_accuracy)

# Step 7: Vary the train/test split
train_sizes = [0.5, 0.7, 0.9]  # Train sizes (percentages)
test_sizes = [1 - size for size in train_sizes]  # Corresponding test sizes

for train_size in train_sizes:
    # Split the data into train and test sets using the specified sizes
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-train_size, random_state=42)

    # Fit the classifier to the training data
    svm_classifier.fit(X_train, y_train)

    # Predict the labels for the test data
    y_pred = svm_classifier.predict(X_test)

    # Compute and display the confusion matrix
    confusion_mat = confusion_matrix(y_test, y_pred)
    print(f"\nTrain Size: {train_size * 100}%, Test Size: {(1 - train_size) * 100}%")
    print("Confusion Matrix:")
    print(confusion_mat)

******************************************************************************
SVM + Naiv Bias +confusion matrix
****************************************************************************
#SVM + NAIEVE BIAS + f1_score, accuracy_score + CROSS VALIDATION 
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score

# Step 1: Load the dataset into a DataFrame
# Assuming the dataset is already loaded into a DataFrame called 'df'
# If not, you can load the dataset using pandas' read_csv() function

# Step 2: Preprocess the data (if necessary)
# Assuming the dataset is already preprocessed and does not require further preprocessing

# Step 3: Split the dataset into features (X) and labels (y)
X = df.iloc[:, :-1]  # Features are all columns except the last one
y = df.iloc[:, -1]   # Labels are the last column

# Step 4: Split the data into train and test sets using a 70-30 split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 5: Apply Naïve Bayes algorithm
naive_bayes_classifier = GaussianNB()

# Fit the classifier to the training data
naive_bayes_classifier.fit(X_train, y_train)

# Predict the labels for the test data
y_pred_nb = naive_bayes_classifier.predict(X_test)

# Calculate F-measure and accuracy for Naïve Bayes
f_measure_nb = f1_score(y_test, y_pred_nb)
accuracy_nb = accuracy_score(y_test, y_pred_nb)

# Step 6: Apply SVM algorithm
svm_classifier = SVC()

# Fit the classifier to the training data
svm_classifier.fit(X_train, y_train)

# Predict the labels for the test data
y_pred_svm = svm_classifier.predict(X_test)

# Calculate F-measure and accuracy for SVM
f_measure_svm = f1_score(y_test, y_pred_svm)
accuracy_svm = accuracy_score(y_test, y_pred_svm)

# Step 7: Perform 10-fold cross-validation for Naïve Bayes and SVM
k = 10
cross_val_scores_nb = cross_val_score(naive_bayes_classifier, X, y, cv=k)
cross_val_scores_svm = cross_val_score(svm_classifier, X, y, cv=k)

# Calculate average F-measure and accuracy for Naïve Bayes using cross-validation
avg_f_measure_nb = cross_val_scores_nb.mean()
avg_accuracy_nb = cross_val_scores_nb.mean()

# Calculate average F-measure and accuracy for SVM using cross-validation
avg_f_measure_svm = cross_val_scores_svm.mean()
avg_accuracy_svm = cross_val_scores_svm.mean()

# Step 8: Display the performance metrics
print("Naïve Bayes Performance:")
print("F-measure:", f_measure_nb)
print("Accuracy:", accuracy_nb)
print("Average F-measure (Cross-Validation):", avg_f_measure_nb)
print("Average Accuracy (Cross-Validation):", avg_accuracy_nb)

print("\nSVM Performance:")
print("F-measure:", f_measure_svm)
print("Accuracy:", accuracy_svm)
print("Average F-measure (Cross-Validation):", avg_f_measure_svm)
print("Average Accuracy (Cross-Validation):", avg_accuracy_svm)

*******************************************************************************
LINEAR REGRESSION V2 
*******************************************************************************


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error

df = pd.read_excel("weather.xlsx")
# print(df.head(5))

y = df["Basel Temperature [2 m elevation corrected]"].values.reshape(-1,1) # column vector conversion
x = df["Basel Growing Degree Days [2 m elevation corrected]"].values.reshape(-1,1)

print(x.shape)
print(y.shape)

plt.scatter(x[::1000],y[::1000])
plt.title('Basel Temp vs Basel Growing Degree Days ')
plt.xlabel("Basel Growing Degree Days")
plt.ylabel("Basel Temp")
plt.show()

x = x[~np.isnan(x).any(axis=1)]
y = y[~np.isnan(y).any(axis=1)]

print(x.shape)
print(y.shape)

xTrain , xTest , yTrain , yTest = train_test_split(x,y,test_size=0.2,random_state = 0)

lReg = LinearRegression()
lReg.fit(xTrain,yTrain)

yPrediction = lReg.predict(X=xTest)

yPrediction = pd.DataFrame ({"Actual": yTest.flatten(), "Predict":yPrediction.flatten()})
print(yPrediction.head(5))


plt.scatter(xTrain[::1000] , yTrain[::1000] , color = "red")
plt.plot(xTrain[::500] , lReg.predict(xTrain)[::500], color = "blue")
plt.title('Basel Temp vs Basel Growing Degree Days ')
plt.xlabel("Basel Growing Degree Days")
plt.ylabel("Basel Temp")
plt.show()


print("Mean Absolute Error:", mean_absolute_error(xTrain ,yTrain))
print("Mean Squared Error:",mean_squared_error(xTrain ,yTrain))
print("Root Mean Squared Error:", np.sqrt(mean_squared_error(xTrain ,yTrain)))




******************************************************************************
MinMax smol
******************************************************************************
def minimax(game_state, depth, maximizing_player):
    if depth == 0 or game_state.is_terminal():
        return game_state.evaluate()

    if maximizing_player:
        max_eval = float('-inf')
        for child_state in game_state.generate_children():
            eval = minimax(child_state, depth - 1, False)
            max_eval = max(max_eval, eval)
        return max_eval
    else:
        min_eval = float('inf')
        for child_state in game_state.generate_children():
            eval = minimax(child_state, depth - 1, True)
            min_eval = min(min_eval, eval)
        return min_eval

# Usage example
game_state = GameState()  # Initialize the game state
best_move = None
best_eval = float('-inf')

for child_state in game_state.generate_children():
    eval = minimax(child_state, depth=3, maximizing_player=False)
    if eval > best_eval:
        best_eval = eval
        best_move = child_state.move

print("Best move:", best_move)



*******************************************************************************
MINMAX
*******************************************************************************

import random

def print_board(board):
    """
    Prints the Tic-Tac-Toe board.
    """
    print("---------")
    for row in board:
        print("|", end="")
        for cell in row:
            print(cell, end="|")
        print("\n---------")

def is_full(board):
    """
    Checks if the board is full.
    """
    for row in board:
        for cell in row:
            if cell == "-":
                return False
    return True

def get_winner(board):
    """
    Determines the winner of the game.
    Returns 'X' if X wins, 'O' if O wins, or '-' if there is no winner yet.
    """
    # Check rows
    for row in board:
        if row[0] == row[1] == row[2] != "-":
            return row[0]

    # Check columns
    for col in range(3):
        if board[0][col] == board[1][col] == board[2][col] != "-":
            return board[0][col]

    # Check diagonals
    if board[0][0] == board[1][1] == board[2][2] != "-":
        return board[0][0]
    if board[0][2] == board[1][1] == board[2][0] != "-":
        return board[0][2]

    return "-"

def get_empty_cells(board):
    """
    Returns a list of empty cells on the board.
    """
    empty_cells = []
    for i in range(3):
        for j in range(3):
            if board[i][j] == "-":
                empty_cells.append((i, j))
    return empty_cells

def evaluate(board):
    """
    Evaluates the current state of the board.
    Returns +1 if X wins, -1 if O wins, or 0 for a tie.
    """
    winner = get_winner(board)
    if winner == "X":
        return 1
    elif winner == "O":
        return -1
    else:
        return 0

def minimax(board, depth, is_maximizing):
    """
    Minimax algorithm implementation.
    """
    if get_winner(board) != "-":
        return evaluate(board)

    if is_full(board):
        return 0

    if is_maximizing:
        best_score = float("-inf")
        for cell in get_empty_cells(board):
            i, j = cell
            board[i][j] = "X"
            score = minimax(board, depth + 1, False)
            board[i][j] = "-"
            best_score = max(score, best_score)
        return best_score

    else:
        best_score = float("inf")
        for cell in get_empty_cells(board):
            i, j = cell
            board[i][j] = "O"
            score = minimax(board, depth + 1, True)
            board[i][j] = "-"
            best_score = min(score, best_score)
        return best_score

def get_best_move(board):
    """
    Finds the best move for the AI using the Minimax algorithm.
    """
    best_score = float("-inf")
    best_move = None
    for cell in get_empty_cells(board):
        i, j = cell
        board[i][j] = "X"
        score = minimax(board, 0, False)
        board[i][j] = "-"
        if score > best_score:
            best_score = score
            best_move = cell
    return best_move

def play_game():
    """
    Plays a simplified Tic-Tac-Toe game against the AI.
    """
    board = [["-" for _ in range(3)] for _ in range(3)]
    print("Welcome to Simplified Tic-Tac-Toe!")
    print_board(board)

    while True:
        # Player's turn
        position = int(input("Enter the position on the board (1-9): "))
        row = (position - 1) // 3
        col = (position - 1) % 3
        if board[row][col] == "-":
            board[row][col] = "O"
        else:
            print("Invalid move! Try again.")
            continue

        # Check for game over
        winner = get_winner(board)
        if winner == "O":
            print("Congratulations! You won!")
            break
        elif is_full(board):
            print("It's a tie!")
            break

        # AI's turn
        print("AI is thinking...")
        ai_move = get_best_move(board)
        board[ai_move[0]][ai_move[1]] = "X"

        # Check for game over
        winner = get_winner(board)
        if winner == "X":
            print("AI wins!")
            break
        elif is_full(board):
            print("It's a tie!")
            break

        print_board(board)

play_game()






    
