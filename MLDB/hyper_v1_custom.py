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
print(items)
"""Parameters (Changed by User)"""
#Dimension of Hypervector
dimensions = 5

#Hypervector Count
num = 28

#Possible Board Entries
entries = sorted(["o", "x"])

"""Parameters Complete"""


"""Initializing Algorithm Items"""
vector = np.ones((dimensions))
ones = np.full((num, dimensions), vector, dtype=int)
flip_matrix = np.random.randint(0, 2, (num, dimensions))

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

"""Helper Funtions Complete"""

