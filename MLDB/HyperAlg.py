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
#Epochs
epochs = 400

#Dimension of Hypervector
dimensions = 100

#Board Size
board_size = 9

#Remoteness Maximum
remote = 9

#Possible Board Entries, one character each
entries = sorted(["o", "x"])

#Possible Primitives, one character each
primitives = sorted(["W", "L", "T"])

"""Parameters Complete"""


"""Initializing Genetic Algorithm Items"""
remote = remote + 1
num = board_size + remote + len(entries) + len(primitives) + 1
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

flips = np.random.randint(0, 2, (num, dimensions))


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

def wAcc(resulting_vectors):
    """
    Vectors are stacked as follows:
        1. Board Location
        2. Piece
        3. Remoteness
        4. Primitive Value
    The last vector represents an empty board.
    """
    
    return 
        
        
def wSim(resulting_vectors):
    
    return 



"""Helper Funtions Complete"""

for e in range(epochs):
    
    flip_vec = bit_flip(hypervectors, flips)
    
    
    file = open("checkpoint.pkl", "wb")
    #Save vectors here
    file.close()
    