import numpy as np

path = "tic_tac_toe.csv"

items = []

with open(path, "r") as file:
    for line in file:
        items.append(line.rstrip().split(","))
items = items[1::]

for idx in range(len(items)):
    items[idx] = (items[idx][0], items[idx][1][0] + items[idx][2][0])

items.sort(key=lambda x: x[1])

items = np.array(items)
print(items.shape)

"""Parameters (Changed by User)"""
#Dimension of Hypervector
dimensions = 100

#Population count
population_count = 200

#Generations
generations = 300

#Mutation in Percent (Recommended at 8)
mutations = 8

#Hypervector Count
num = 16

#Possible Board Entries
entries = sorted(["o", "x", "-"])

#Ratio for Accuracy (Between 0 and 1)
ratio = 0.75
"""Parameters Complete"""


"""Initializing Genetic Algorithm Items"""
populations = {}
vector = np.ones((dimensions)) - 2 * np.random.randint(0, 2, (dimensions))
np.savetxt("vector.csv", vector.astype(int))
hypervectors = np.full((num, dimensions), vector, dtype=int)
primitives = sorted(["W", "L", "T"])
combinations = [f"{primitive}{remoteness}" for remoteness in range(num - 6) for primitive in primitives]
total_comb_count = np.array([0 for __ in range(len(combinations))])
for i in items:
    total_comb_count[combinations.index(i[1])] += 1
#Check Solved Game Values, Comment out if unecessary
print(combinations)
print(total_comb_count)
primitives = {primitives[i] : i for i in range(len(primitives))}

for i in range(population_count):
    populations.update({i: np.random.randint(0, 2, (num, dimensions))})
    for v in range(len(populations[i])):
        if np.count_nonzero(populations[i][v]) > dimensions/2:
            indices = np.nonzero(populations[i][v])
            np.random.shuffle(indices)
            for idx in range(len(indices) - (dimensions//2)):
                populations[i][v][idx] = 0
            
"""Initialization Complete"""


"""Helper Functions for Hyperdimensional Computing"""
def bit_flip(hypervectors, reference):
    return hypervectors + -2 * np.multiply(hypervectors, reference)

def multiply(v1, v2):
    return np.multiply(v1, v2)

def add(v1, v2):
    v3 = v1 + v2
    return np.clip(v3, -1, 1, dtype=int)

def cosim(v1, v2):
    return np.dot(v1, v2)/(np.linalg.norm(v1) * np.linalg.norm(v2))

def accuracy(dataset, hypervectors):
    translated_combinations = np.array([np.multiply(hypervectors[num - 6 + primitives[combinations[idx][0]]], hypervectors[int(combinations[idx][1])]) for idx in range(len(combinations))])
    acc = 0
    for data in range(len(dataset)):
        board_vector = np.zeros(dimensions, dtype=int)
        hypervector_indices = [num - 3 + entries.index(item) for item in dataset[data][0]]
        for idx, hypervector_idx in enumerate(hypervector_indices):
            board_vector = add(board_vector, np.multiply(hypervectors[hypervector_idx], hypervectors[idx]))
        values = [cosim(board_vector, v2) for v2 in translated_combinations]
        best_idx = values.index(max(values))
        if combinations[best_idx] == dataset[data][1]:
            acc += 1
    return acc / len(dataset)

def wAcc(dataset, hypervectors):
    translated_combinations = np.array([np.multiply(hypervectors[num - 6 + primitives[combinations[idx][0]]], hypervectors[int(combinations[idx][1])]) for idx in range(len(combinations))])
    length = len(combinations)
    sol_total = [0 for __ in range(length)]
    acc_sol = [0 for __ in range(length)]
    for data in range(len(dataset)):
        board_vector = np.zeros(dimensions, dtype=int)
        hypervector_indices = [num - 3 + entries.index(item) for item in dataset[data][0]]
        for idx, hypervector_idx in enumerate(hypervector_indices):
            board_vector = add(board_vector, np.multiply(hypervectors[hypervector_idx], hypervectors[idx]))
        values = [cosim(board_vector, v2) for v2 in translated_combinations]
        best_idx = values.index(max(values))
        sol_total[best_idx] += 1
        if combinations[best_idx] == dataset[data][1]:
            acc_sol[best_idx] += 1
    acc = 0.0
    for idx in range(length):
        if acc_sol[idx] == 0:
            if sol_total == 0:
                acc += 1.0
        else:
            acc += acc_sol[idx] / sol_total[idx]
    return acc / length

def wSim(hypervectors):
    norms = [np.linalg.norm(v) for v in hypervectors]
    similarity = 1
    length = len(hypervectors)
    for idx_1 in range(length - 1):
        for idx_2 in range(idx_1 + 1, length):
            similarity *= ((np.dot(hypervectors[idx_1], hypervectors[idx_2])) / (norms[idx_1] * norms[idx_2]))
    return 1 - (similarity ** (1/length))

def mutate(matrix):
    for idx in range(len(matrix)):
        if (np.random.randint(0, 100) - mutations) < 0:
            flips = np.random.randint(0, dimensions, np.random.randint(1, dimensions // 2))
            matrix[idx] = [1 if i in flips else 0 for i in range(len(matrix[0]))]
    return matrix

def mate(m1, m2, target):
    child_m_idx = np.random.randint(0, 2, len(m1))
    populations[target] = mutate(np.array([m1[idx] if child_m_idx[idx] == 0 else m2[idx] for idx in range(len(child_m_idx))]))
    
def updatePopulation(fit_order):
    half_length = len(fit_order) // 2
    first_half, second_half = fit_order[:half_length], fit_order[half_length:]
    for idx in range(len(first_half)):
        mate_idx = (idx + np.random.randint(0, half_length)) % half_length
        mate(populations[first_half[idx]], populations[first_half[mate_idx]], second_half.pop())
"""Helper Funtions Complete"""

"""
for g in range(generations):
    accuracy_list = []
    print("Selecting...")
    for i in range(population_count):
        result_hypervectors = bit_flip(hypervectors, populations[i])
        accuracy_list.append(accuracy(items, result_hypervectors))
    sorted_indices = sorted([i for i in range(population_count)], key=lambda x: accuracy_list[x])[::-1]
    updatePopulation(sorted_indices)
    print(f"Generation {g}: {accuracy_list[sorted_indices[0]]}")
    np.savetxt("hypervectors.csv", populations[sorted_indices[0]].astype(int))
"""

for g in range(generations):
    accuracy_list = []
    similarity_list = []
    print("Selecting...")
    for i in range(population_count):
        result_hypervectors = bit_flip(hypervectors, populations[i])
        accuracy_list.append(wAcc(items, result_hypervectors))
        similarity_list.append(wSim(result_hypervectors))
    print(accuracy_list)
    sorted_indices = sorted([i for i in range(len(populations))], key=lambda x: (accuracy_list[x] * ratio) + ((1 - ratio) * similarity_list[x]))[::-1]
    updatePopulation(sorted_indices)
    print(f"Generation {g}: {accuracy_list[sorted_indices[0]]}")
    np.savetxt("hypervectors.csv", populations[sorted_indices[0]].astype(int))