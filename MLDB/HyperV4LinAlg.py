import numpy as np

path = "tic_tac_toe.csv"

items = []
remote_prim_val = set()

with open(path, "r") as file:
    for line in file:
        items.append(line.rstrip().split(","))
items = items[1::]

for idx in range(len(items)):
    items[idx] = (items[idx][0], items[idx][1][0] + items[idx][2][0])

items.sort(key=lambda x: x[1])

items = np.array(items)
print(items.shape)

def bit_flip(hypervectors, reference):
    return hypervectors + -2 * np.multiply(hypervectors, reference)

def multiply(v1, v2):
    return np.multiply(v1, v2)

def add(v1, v2):
    return v1 + v2

def clip(v):
    return np.clip(v, -1, 1)

def cosim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

for i in items:
    remote_prim_val.add(i[1])

remote_prim_val = sorted(list(remote_prim_val))

print(items)

print(remote_prim_val)

"""End of Setup"""


"""Tune Values Here"""

dimensions = 10

num_of_hypervectors = len(remote_prim_val) + 9 + 2 + 1

"""End of Values"""

populations = {}
hypervectors = np.ones((num_of_hypervectors, dimensions))

reference = np.random.randint(0, 2, (num_of_hypervectors, dimensions))

print(reference)

res = bit_flip(hypervectors, reference)

combinations = {}

prim_vec = res[0:len(remote_prim_val)]
positions = res[len(remote_prim_val):len(remote_prim_val) + 9]
items = res[len(remote_prim_val) + 9: len(remote_prim_val) + 9 + 2]
board = res[len(remote_prim_val) + 9 + 2:]

for v1 in range(len(positions)):
    for v2 in range(len(items)):
        res_v = multiply(positions[v1], items[v2])
        combinations.update({(v1, v2): res_v})
        
print(combinations)

