import HomeFunGame2 as Game;
import HomeFunSolver as Solver
import math

#GameModes
gameModes = ["Normal", "Misere", "OnlyX", "MisereOnlyX", "OrderFirstChaos", "ChaosFirstOrder"]

class playGame:
    
    def __init__(self, cpu=False, first=True):
        self.gameModes = gameModes
        self.solvedGame = None
        self.prevPositions = []
        self.position = Solver.init_pos
        self.p1Turn = True
        self.cpu = cpu
        self.remoteness = None
        if cpu and not first:
            self.p1Turn = False
    
    def chooseMode(self):
        modes = ""
        for i in range(len(self.gameModes)):
            modes += f"\n{i}: {self.gameModes[i]}"
        gameMode = ""
        while gameMode == "":
            print(modes)
            gameMode = input("Please enter the number corresponding to the game mode that you would like to play.:\n")
            try:
                if int(gameMode) >= 0 and int(gameMode) < len(self.gameModes):
                    self.mode = self.gameModes[int(gameMode)]
                else:
                    print("Not a valid game mode")
                    gameMode = ""
            except:
                print("Not a valid game mode")
                gameMode = ""
        gameMode = int(gameMode)
        if gameMode == 1 or gameMode == 3: Game.MISERE = True 
        if gameMode == 2 or gameMode == 3: Game.ONLYX = True
        if gameMode == 4 or gameMode == 5: Game.ORDER_CHAOS = True
        if gameMode == 4: Game.ORDER_FIRST = True
        
        Solver.Solve(Solver.init_pos)
        self.solvedGame = Solver.memo
        self.remoteness = Solver.remoteness
        
    def analyzeGame(self):
        Solver.analysis()
        
    def printCurrPos(self):
        pos = Game.str_to_matrix(self.position)
        position = ""
        for row in pos:
            for entry in row:
                if entry != None:
                    position += " [" + entry + "] "
                else:
                    position += " [ ] "
            position += "\n"
        print(position)
    
            
    def bestScenario(self, list_of_moves):
        scenario = "Win the Game"
        for move in list_of_moves:
            if Solver.memo[move] == "Lose the Game":
                scenario = "Lose the Game"
                break
            elif Solver.memo[move] == "Draw the Game":
                scenario = "Draw the Game"
        if scenario == "Lose the Game":
            remoteness = math.inf
            for m in list_of_moves:
                if Solver.memo[m] == "Lost the Game":
                    remoteness = min(Solver.remoteness[m], remoteness)
        elif scenario == "Draw the Game":
            remoteness = 0
            for m in list_of_moves:
                if Solver.memo[m] == "Draw the Game":
                    remoteness = max(Solver.remoteness[m], remoteness)
        else:
            remoteness = 0
            for m in list_of_moves:
                if Solver.memo[m] == "Lose the Game":
                    remoteness = max(Solver.remoteness[m], remoteness)
            
        return scenario, remoteness
    
    def doMove(self):
        p = Solver.find_canonical(Game.check_symmetry(self.position))
        print(Solver.memo[p] + " in " + str(Solver.remoteness[p]) + "\n")
        legalMoves = Game.GenerateMoves(self.position)
        if self.cpu and not self.p1Turn:
            possible_moves = []
            best_hash = []
            for move in legalMoves:
                possible_moves.append(Game.DoMove(self.position, move))
            for board in possible_moves:
                best_hash.append(Solver.find_canonical(Game.check_symmetry(board)))
            scenario, remoteness = self.bestScenario(best_hash)
            true_index = 0
            for index in range(len(possible_moves)):
                if Solver.memo[best_hash[index]] == scenario and Solver.remoteness[best_hash[index]] == remoteness:
                    true_index = index
                    break
            self.position = Game.DoMove(self.position, legalMoves[true_index])
            self.p1Turn = not self.p1Turn
            if Game.PrimitiveValue(self.position) != "Not Primitive":
                self.printCurrPos()
                print(f"You have {Game.PrimitiveValue(self.position)}")
                return
            
        legalMoves = Game.GenerateMoves(self.position)
        p1, p2 = "Player 1", "Player 2"
        moves = ""
        for ind in range(len(legalMoves)):
            moves += f"{ind}: {legalMoves[ind]}\n"
        playerMove = ""
        while playerMove == "":
            print(moves)
            self.printCurrPos()
            playerMove = input(f"{p1 if self.p1Turn else p2} please make a move:\n")
            try:
                if int(playerMove) < 0 or int(playerMove) >= len(legalMoves):
                    print("Not a valid move\n")
                    playerMove = ""
            except:
                print("Not a valid move\n")
                playerMove = ""
        playerMove = int(playerMove)
        self.prevPositions.append(self.position)
        self.position = Game.DoMove(self.position, legalMoves[playerMove])
        self.p1Turn = not self.p1Turn
        undo = input("Please type yes if the previous player would like to undo move, any other response will be ignored:\n")
        if undo == "yes":
            self.undoMove()
            self.p1Turn = not self.p1Turn
        if Game.PrimitiveValue(self.position) != "Not Primitive":
            print(f"{p1 if self.p1Turn else p2} {Game.PrimitiveValue(self.position)}")
            
    def initialize(self):
        self.position = Solver.init_pos
        
    def undoMove(self):
        self.position = self.prevPositions.pop()
        

g = playGame()
g.chooseMode()
while Game.PrimitiveValue(g.position) == "Not Primitive":
    g.doMove()