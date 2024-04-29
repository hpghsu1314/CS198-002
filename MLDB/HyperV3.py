import numpy as np
import multiprocessing



dimensions = 100
population_count = 1000
generations = 300
mutations = 8
board_size = 9
remote = 9
entries = sorted(["o", "x"])
primitives = sorted(["W", "L", "T"])
ratio = 0.10



def bit_flip(hypervectors, reference):
    return hypervectors + -2 * np.multiply(hypervectors, reference)

def multiply(v1, v2):
    return np.multiply(v1, v2)

def add(v1, v2):
    v3 = v1 + v2
    return np.clip(v3, -1, 1, dtype=int)

def cosim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def mutate(matrix, mutations):
    for idx in range(len(matrix)):
        if (np.random.randint(0, 100) - mutations) < 0:
            flips = np.random.randint(0, len(matrix[0]), np.random.randint(1, len(matrix[0]) // 2))
            matrix[idx] = [1 if i in flips else 0 for i in range(len(matrix[0]))]
    return matrix

def mate(m1, m2, target, populations):
    child_m_idx = np.random.randint(0, 2, len(m1))
    populations[target] = mutate(np.array([m1[idx] if child_m_idx[idx] == 0 else m2[idx] for idx in range(len(child_m_idx))]), mutations)

def updatePopulation(fit_order, populations):
    half_length = len(fit_order) // 2
    first_half, second_half = fit_order[:half_length], fit_order[half_length:]
    for idx in range(len(first_half)):
        mate_idx = (idx + np.random.randint(0, half_length)) % half_length
        mate(populations[first_half[idx]], populations[first_half[mate_idx]], second_half.pop(), populations)

def wAcc(resulting_vectors, items, entries):
    board_states = len(entries.keys())
    vBoard, vPiece, vRemote, vPrim = resulting_vectors[0:len(entries)], resulting_vectors[len(entries):len(entries) + board_states], resulting_vectors[len(entries) + board_states:len(entries) + board_states + remote], resulting_vectors[len(entries) + board_states + remote:-1]
    translated_comb = np.array([multiply(r, p) for r in vRemote for p in vPrim])
    acc = 0
    for state in items:
        board = state[0]
        state_vec = resulting_vectors[-1]
        for idx in range(len(board)):
            if board[idx] in entries.keys():
                state_vec = add(state_vec, np.multiply(vPiece[entries[board[idx]]], vBoard[idx]))
        similarities = [0 if total_comb_count[idx] == 0 else cosim(state_vec, translated_comb[idx]) for idx in range(len(translated_comb))]
        classification = similarities.index(max(similarities))
        if state[1] == combinations[classification]:
            acc += 1
    return acc / len(items)

def wSim(resulting_vectors):
    summation = 0.0
    result_length = len(resulting_vectors)
    for n in range(result_length - 1):
        for k in range(n + 1, result_length):
            number = abs(np.dot(resulting_vectors[n], resulting_vectors[k]) / (np.linalg.norm(resulting_vectors[n]) * np.linalg.norm(resulting_vectors[k])))
            summation += number
    return 1 - (summation / ((result_length * (result_length - 1)) / 2))

def fitness(accuracy, similarity, ratio):
    return [(ratio * accuracy[i]) + ((1 - ratio) * similarity[i]) for i in range(population_count)]

def splitDict(dictionary, num_of_parts):
    return {i % num_of_parts: {idx: dictionary[idx] for idx in dictionary.keys() if idx % num_of_parts == i} for i in range(num_of_parts)}

def evaluate_fitness(member, hypervectors, items, entries, total_comb_count):
    result = bit_flip(hypervectors, member)
    return (1 - ratio) * wAcc(result, items, entries) + ratio * wSim(result)

def main():
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

    remote = remote + 1
    num = board_size + remote + len(entries) + len(primitives) + 1
    populations = {i: np.random.randint(0, 2, (num, dimensions)) for i in range(population_count)}
    hypervectors = np.ones((num, dimensions))
    combinations = [f"{primitive}{remoteness}" for remoteness in range(remote) for primitive in primitives]
    total_comb_count = np.array([0 for __ in range(len(combinations))])
    for i in items:
        total_comb_count[combinations.index(i[1])] += 1

    print(combinations)
    print(total_comb_count)
    primitives = {primitives[i]: i for i in range(len(primitives))}
    entries = {entries[i]: i for i in range(len(entries))}

    for g in range(generations):
        acc_lst = []
        sim_lst = []
        print("Selecting...")

        with multiprocessing.Pool(processes=8) as pool:
            fitness_lst = pool.starmap(evaluate_fitness, [(populations[idx], hypervectors, items, entries, total_comb_count) for idx in range(population_count)])

        print(fitness_lst)
        rankings = sorted([i for i in range(population_count)], key=lambda x: fitness_lst[x], reverse=True)
        updatePopulation(rankings, populations)
        print(f"Generation {g} | {acc_lst[rankings[0]]} | {sim_lst[rankings[0]]}")
        np.savetxt("flips.csv", populations[rankings[0]].astype(int))

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
