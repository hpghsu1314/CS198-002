import z3
import numpy as np
import HomeFunGame2 as g
import HomeFunSolver as s
import math

class MatrixFinder:
    
    def __init__(self, solved_memo, remoteness, row, col):
        self.solved_memo = solved_memo
        self.nextMoveVec = {}
        self.remoteness = remoteness
        self.solution = np.array([[z3.Real(f"s_{c}_{r}") for c in range(col)] for r in range(row)])
        self.solver = z3.Solver()
        self.X = z3.Real("X")
        self.O = z3.Real("O")
        self.N = z3.Real("N")
        self.result = [z3.Real(f"R_{i}") for i in range(col)]
        
    
    def translateMemo(self):
        for key in self.solved_memo.keys():
            if g.PrimitiveValue(key) == "Not Primitive":
                lst_of_moves = g.GenerateMoves(key)
                boards = []
                for move in lst_of_moves:
                    boards.append(s.find_canonical(g.check_symmetry(g.DoMove(key, move))))
                scenario, remoteness = self.bestScenario(boards)
                for b in range(len(boards)):
                    if self.solved_memo[boards[b]] == scenario and self.remoteness[boards[b]] == remoteness:
                        position = lst_of_moves[b][0] * 3 + lst_of_moves[b][1]
                        vec = [0 if i != position else 1 for i in range(9)]
                        self.nextMoveVec.update({self.posToVec(key): vec})
                        break
    
    def posToVec(self, pos):
        returnVec = []
        for p in pos:
            returnVec.append(self.X if p == "X" else self.O if p == "O" else self.N)
        return tuple(returnVec)
    
    def bestScenario(self, list_of_moves):
        scenario = "Win the Game"
        for move in list_of_moves:
            if self.solved_memo[move] == "Lose the Game":
                scenario = "Lose the Game"
                break
            elif self.solved_memo[move] == "Draw the Game":
                scenario = "Draw the Game"
        if scenario == "Lose the Game":
            remoteness = math.inf
            for m in list_of_moves:
                if self.solved_memo[m] == "Lose the Game":
                    remoteness = min(self.remoteness[m], remoteness)
        elif scenario == "Draw the Game":
            remoteness = 0
            for m in list_of_moves:
                if self.solved_memo[m] == "Draw the Game":
                    remoteness = max(self.remoteness[m], remoteness)
        else:
            remoteness = 0
            for m in list_of_moves:
                if self.solved_memo[m] == "Win the Game":
                    remoteness = max(self.remoteness[m], remoteness)
            
        return scenario, remoteness
    
    def findSol(self, positions):
        for i in positions:
            move = self.findIndex(i)
            sol_vec = self.solution[move]
            for k in self.nextMoveVec.keys():
                v = self.multVec(sol_vec, k)
                if self.nextMoveVec[k][move] == 1:
                    self.solver.add(v >= self.result[move])
                else:
                    self.solver.add(v < self.result[move])
                      
    def multVec(self, v1, v2):
        temp = 0
        for i in range(len(v1)):
            temp += v1[i] * v2[i]
        return z3.simplify(temp)

    def findIndex(self, vector):
        for i in range(len(vector)):
            if vector[i] == 1:
                return i
            

s.Solve(g.init_pos)
memo, remoteness = s.memo, s.remoteness
mFinder = MatrixFinder(memo, remoteness, 9, 9)
mFinder.translateMemo()
mFinder.findSol([[1,0,0,0,0,0,0,0,0]])
print("done execution")
solved = mFinder.solver.check()
print(mFinder.nextMoveVec)
print(solved)
if solved == z3.sat:
    z3.set_option(max_args=10000000, max_lines=1000000, max_depth=10000000, max_visited=1000000)
    m = mFinder.solver.model()
    print(m)
    file = open("Satisfiability.txt", "w+")
    file.write(str(m))