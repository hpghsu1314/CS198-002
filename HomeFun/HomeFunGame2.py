#Tic-Tac-Toe
ROW = 3
COLUMN = 3
IN_A_ROW = 3


#Different game modes
MISERE = False
ONLYX = False
MISERE_ONLYX = False
if MISERE_ONLYX:
    MISERE, ONLYX = True, True
ORDER_CHAOS = False
ORDER_FIRST = False
ORDER_CHAOS_SYMMETRIES = False

def matrix_to_str(matrix):
    string = ""
    for row in matrix:
        for item in row:
            if item != None:
                string += str(item)
            else:
                string += "N"
    return string

def str_to_matrix(string):
    index = 0
    matrix = [[None for __ in range(ROW)] for __ in range(COLUMN)]
    for row in range(COLUMN):
        for col in range(ROW):
            if string[index] != "N":
                matrix[row][col] = string[index]
            index += 1
    return matrix

def DoMove(position, move):
    position = str_to_matrix(position)
    assert position[move[0]][move[1]] == None, "Position not valid"
    
    empty = 0
    for row in position:
        for val in row:
            if val == None:
                empty += 1
    if ONLYX:
        position[move[0]][move[1]] = "X"
    elif ORDER_CHAOS:
        position[move[0]][move[1]] = move[2]
    else:
        if empty % 2 == 1:
            position[move[0]][move[1]] = "O"
        else:
            position[move[0]][move[1]] = "X"
        
    position = matrix_to_str(position)
    
    return position


def GenerateMoves(position):
    position = str_to_matrix(position)
    temp = []
    for row in range(len(position)):
        for val in range(len(position[row])):
            if position[row][val] == None:
                if ORDER_CHAOS:
                    temp.append((row, val, "X"))
                    temp.append((row, val, "O"))
                else:
                    temp.append((row, val))
    return temp


def PrimitiveValue(position):
    if ORDER_CHAOS:
        moves = GenerateMoves(position)
    position = str_to_matrix(position)
    lose = check_row(position) or check_col(position) or check_left_diag(position) or check_right_diag(position)
    
    if ORDER_CHAOS:
        if (len(moves)//2) % 2 == 1:
            if lose:
                return "Win" if ORDER_FIRST else "Lost"
            if len(moves) == 0:
                return "Lost" if ORDER_FIRST else "Win"
        elif (len(moves)//2) % 2 == 0:
            if lose:
                return "Lost" if ORDER_FIRST else "Win"
            if len(moves) == 0:
                return "Win" if ORDER_FIRST else "Lost"
        return "Not Primitive"
    
    if lose:
        if MISERE:
            return "Win"
        return "Lost"
    else:
        for row in position:
            for value in row:
                if value == None:
                    return "Not Primitive"
        return "Draw"


def check_row(position):
    if IN_A_ROW <= ROW:
        for adjustment in range(ROW - IN_A_ROW + 1):
            for row in position:
                lose = True
                for index in range(IN_A_ROW-1):
                    if row[index+adjustment] != row[index+adjustment+1] or row[index+adjustment] == None:
                        lose = False
                        break
                if lose == True:
                    return True
    return False

def check_col(position):
    if IN_A_ROW <= COLUMN:
        for adjustment in range(COLUMN - IN_A_ROW + 1):
            for col in range(len(position[0])):
                lose = True
                for index in range(IN_A_ROW-1):
                    if position[index + adjustment][col] != position[index + adjustment + 1][col] or position[index + adjustment][col] == None:
                        lose = False
                        break
                if lose == True:
                    return True
    return False

def check_left_diag(position):
    if IN_A_ROW <= COLUMN and IN_A_ROW <= ROW:
        for adjustment in range(COLUMN - IN_A_ROW + 1):
            for horizontal in range(ROW - IN_A_ROW + 1):
                lose = True
                for index in range(IN_A_ROW - 1):
                    if position[index + adjustment][index + horizontal] != position[index + adjustment + 1][index + horizontal + 1] or position[index + adjustment][index + horizontal] == None:
                        lose = False
                        break
                if lose == True:
                    return True
    return False

#Done
def check_right_diag(position):
    if IN_A_ROW <= COLUMN and IN_A_ROW <= ROW:
        for adjustment in range(COLUMN - IN_A_ROW + 1):
            for horizontal in range(ROW - IN_A_ROW + 1):
                lose = True
                for index in range(IN_A_ROW - 1):
                    if position[IN_A_ROW - index - 1 + adjustment][index + horizontal] != position[IN_A_ROW - index - 2 + adjustment][index + 1 + horizontal] or position[IN_A_ROW - index - 1 + adjustment][index + horizontal] == None:
                        lose = False
                        break
                if lose == True:
                    return True
    return False

def check_symmetry(board):
    board = str_to_matrix(board)
    result = []
    temp = [n.copy() for n in board]
    mirrored = [[] for __ in range(len(board))]
    result.append(temp)
    
    for row in range(len(board)):
        mirrored[row] = temp[row][::-1]
    
    result.append(mirrored)
    
    if ROW == COLUMN:
        result = result + rotation(result[0]) + rotation(result[1])
    else:
        result = result + rotate180(result[0]) + rotate180(result[1])
    
    if ORDER_CHAOS_SYMMETRIES and ORDER_CHAOS:
        more_results = []
        for item in result:
            temp = [n.copy() for n in item]
            for row in temp:
                for index in range(len(row)):
                    row[index] = "O" if row[index] == "X" else "X" if row[index] == "O" else None
            more_results.append(temp)
        result = result + more_results

    for item in range(len(result)):
        result[item] = matrix_to_str(result[item])
        
    return result


def rotation(board):
    rotated_boards = []
    temp = [[None for __ in range(len(board))] for __ in range(len(board))]
    for __ in range(3):
        for r in range(len(board)):
            for c in range(len(board)):
                temp[c][len(board)-r-1] = board[r][c]
        new_board = [n.copy() for n in temp]
        rotated_boards.append(new_board)
        board = new_board
    return rotated_boards

def rotate180(board):
    rotated_boards = []
    temp = [[None for __ in range(ROW)] for __ in range(COLUMN)]
    for r in range(ROW):
        for c in range(COLUMN):
            temp[c][r] = board[COLUMN - c - 1][ROW - r - 1]
    new_board = [n.copy() for n in temp]
    rotated_boards.append(new_board)
    return rotated_boards

init_pos = [[None for __ in range(ROW)] for __ in range(COLUMN)]
init_pos = matrix_to_str(init_pos)