import HomeFunGame2 as Game

memo = {}
remoteness = {}

opposites = {"Win the Game": "Lose the Game", "Draw the Game": "Draw the Game", "Lose the Game": "Win the Game"}
prim_value_chart = {"Lost": "Lose the Game", "Win": "Win the Game", "Draw": "Draw the Game"}
total_prim = {"Win": 0, "Lost": 0, "Draw": 0, "Total" : 0}
opp_prim_value_chart = {"Lose the Game" : "Lost", "Win the Game": "Win", "Draw the Game": "Draw"}

symmetry = True

def update_remoteness(position, results, value):
    if value == "Lose the Game":
        remote_num = max([remoteness[r[0]] for r in results if r[1] == value]) + 1
    else:
        remote_num = min([remoteness[r[0]] for r in results if r[1] == value]) + 1
    remoteness.update({position : remote_num})

def find_canonical(list_of_items):
    item = list_of_items[0]
    item_hash = hash(item)
    for i in list_of_items:
        if abs(hash(i)) < abs(item_hash):
            item_hash = hash(i)
            item = i
    return item

def check_res(results):
    win = "Win the Game"
    draw = "Draw the Game"
    lose = "Lose the Game"
    return win if win in results else draw if draw in results else lose

def Solve(position):
    prim_value = Game.PrimitiveValue(position)
    if prim_value != "Not Primitive":
        total_prim[prim_value] += 1
        total_prim["Total"] += 1
        remoteness.update({position : 0})
        return prim_value_chart[prim_value]
    else:
        legal_moves = Game.GenerateMoves(position)
        result = []
        for move in legal_moves:
            new_pos = Game.DoMove(position, move)
            
            if symmetry:
                positions = Game.check_symmetry(new_pos)
                new_pos = find_canonical(positions)
                
            if new_pos not in memo.keys():
                res = Solve(new_pos)
                memo.update({new_pos: res})
                result.append((new_pos, opposites[res]))
            else:
                result.append((new_pos, opposites[memo[new_pos]]))
        
        curr_cond = check_res([res[1] for res in result])
        memo.update({position: curr_cond})
        update_remoteness(position, result, curr_cond)
        return curr_cond

init_pos = Game.init_pos

def analysis():
    max_value, min_value = max([remoteness[key] for key in remoteness.keys()]), min([remoteness[key] for key in remoteness.keys()])
    print("Remote  Win     Lose    Tie     Total")
    for remote_num in range(max_value, min_value - 1, -1):
        win = 0
        lose = 0
        draw = 0
        for key in remoteness.keys():
            if remoteness[key] == remote_num:
                if memo[key] == "Win the Game":
                    win += 1
                elif memo[key] == "Lose the Game":
                    lose += 1
                else:
                    draw += 1
        total = win + lose + draw
        print(f"{remote_num}" + " " * (8 - len(f"{remote_num}")) + f"{win}" + " " * (8 - len(f"{win}")) + f"{lose}" + " " * (8 - len(f"{lose}")) + f"{draw}" + " " * (8 - len(f"{draw}")) + f"{total}" + " " * (8 - len(f"{total}")))

    draw = 0
    win = 0
    lose = 0
    for i in memo.keys():
        if memo[i] == "Draw the Game":
            draw += 1
        elif memo[i] == "Win the Game":
            win += 1
        else:
            lose += 1
    total = f"{win + lose + draw}"
    win = f"{win}"
    lose = f"{lose}"
    draw = f"{draw}"
    print("Total   " + win + " " * (8 - len(win)) + lose + " " * (8 - len(lose)) + draw + " " * (8 - len(draw)) + total)


Solve(init_pos)
analysis()
print(memo)