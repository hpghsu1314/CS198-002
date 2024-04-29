import numpy as np
import multiprocessing
import pickle

num_cores = multiprocessing.cpu_count()
print(f"Number of CPU cores: {num_cores}")


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
population_count = 500

#Generations
generations = 500

#Mutation in Percent (Recommended at 8)
mutations = 8

#Board Size
board_size = 9

#Remoteness Maximum
remote = 9

#Possible Board Entries, one character each
entries = sorted(["o", "x"])

#Possible Primitives, one character each
primitives = sorted(["W", "L", "T"])

#Ratio for Accuracy (Between 0 and 1)
ratio = 0.75
"""Parameters Complete"""


"""Initializing Genetic Algorithm Items"""
remote = remote + 1
num = board_size + remote + len(entries) + len(primitives) + 1
populations = {}
hypervectors = np.ones((num, dimensions))
combinations = [f"{primitive}{remoteness}" for remoteness in range(remote) for primitive in primitives]
total_comb_count = np.array([0 for __ in range(len(combinations))])
for i in items:
    total_comb_count[combinations.index(i[1])] += 1
#Check Solved Game Values, Comment out if unecessary
print(combinations)
print(total_comb_count)
primitives = {primitives[i] : i for i in range(len(primitives))}
entries = {entries[i]: i for i in range(len(entries))}

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
    return v1 + v2

def clip(v):
    return np.clip(v, -1, 1, dtype=int)

def cosim(v1, v2):
    if np.linalg.norm(v1) == 0:
        return 0
    else:
        return np.dot(v1, v2)/(np.linalg.norm(v1) * np.linalg.norm(v2))

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

def wAcc(resulting_vectors):
    """
    Vectors are stacked as follows:
        1. Board Location
        2. Piece
        3. Remoteness
        4. Primitive Value
    The last vector represents an empty board.
    """
    board_states = len(entries.keys())
    vBoard, vPiece, vRemote, vPrim = resulting_vectors[0:board_size], resulting_vectors[board_size:board_size + board_states], resulting_vectors[board_size + board_states:board_size + board_states + remote], resulting_vectors[board_size + board_states + remote:-1]
    translated_comb = np.array([multiply(r, p) for r in vRemote for p in vPrim])
    acc = 0
    for state in items:
        board = state[0]
        state_vec = resulting_vectors[-1]
        for idx in range(board_size):
            if board[idx] in entries.keys():
                state_vec = add(state_vec, np.multiply(vPiece[entries[board[idx]]], vBoard[idx]))
        state_vec = clip(state_vec)
        similarities = [0 if total_comb_count[idx] == 0 else cosim(state_vec, translated_comb[idx]) for idx in range(len(translated_comb))]
        classification = similarities.index(max(similarities))
        if state[1] == combinations[classification]:
            acc += 1
    return acc / len(items)
        
        
def wSim(resulting_vectors):
    summation = 0.0
    result_length = len(resulting_vectors)
    for n in range(result_length-1):
        for k in range(n+1, result_length):
            number = abs(np.dot(resulting_vectors[n], resulting_vectors[k]) / (np.linalg.norm(resulting_vectors[n]) * np.linalg.norm(resulting_vectors[k])))
            summation += number
    return 1 - (summation / ((result_length * (result_length - 1)) / 2))

def fitness(accuracy, similarity):
    return [(ratio * accuracy[i]) + ((1-ratio) * similarity[i]) for i in range(population_count)]

def evaluate_fitness(member):
    result = np.array(bit_flip(hypervectors, member), dtype=np.int32)
    return (1-ratio)*wAcc(result) + ratio*wSim(result)

"""Helper Funtions Complete"""

for g in range(generations):
    acc_lst = []
    sim_lst = []
    print("Selecting...")
    for p in range(population_count):
        result = np.array(bit_flip(hypervectors, populations[p]), dtype=np.int32)
        acc_lst.append(wAcc(result))
        sim_lst.append(wSim(result))
    fitness_lst = fitness(acc_lst, sim_lst)
    rankings = sorted([i for i in range(population_count)], key=lambda x: fitness_lst[x], reverse=True)
    updatePopulation(rankings)
    print(f"Generation {g} | {acc_lst[rankings[0]]} | {sim_lst[rankings[0]]}")
    np.savetxt("flips.csv", populations[rankings[0]].astype(int))
    file = open("checkpoint.pkl", "wb")
    pickle.dump(populations, file)
    file.close()
    

"""
if __name__ == "__main__":
    multiprocessing.freeze_support()
    for g in range(generations):
        acc_lst = []
        sim_lst = []
        print("Selecting...")
        for idx in range(population_count):
            fitness_lst = evaluate_fitness(populations[idx])
        print(fitness_lst)
        rankings = sorted([i for i in range(population_count)], key=lambda x: fitness_lst[x], reverse=True)
        updatePopulation(rankings)
        print(f"Generation {g} | {acc_lst[rankings[0]]} | {sim_lst[rankings[0]]}")
        np.savetxt("flips.csv", populations[rankings[0]].astype(int))
"""
