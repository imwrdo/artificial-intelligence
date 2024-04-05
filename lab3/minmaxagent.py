import copy, sys
from exceptions import AgentException
import math
 
def basic_static_eval(connect4, player="o"):
    if(player == connect4.wins):
        return float('inf')
    elif(player != connect4.wins and connect4.wins is not None):
        return -float('inf')
    elif((connect4.game_over is True) and (connect4.wins is None)):
        return 0
    else:
        countMy = 0
        countNotMy = 0
        for four in connect4.iter_fours():
            if player == 'o':
                target = 'x'
            else:
                target = 'o'
            if four.count(player)==3:
                countMy+=1
            if four.count(target)==3:
                countNotMy+=1
                
        return countMy - countNotMy
 
 
def advanced_static_eval(connect4, player="o"):
    # TODO
    return 0  # return score for player
 
 
class MinMaxAgent:
    def __init__(self, my_token="o", heuristic_func=basic_static_eval):
        self.my_token = my_token
        self.heuristic_func = heuristic_func
 
    def decide(self, connect4):
        if connect4.who_moves != self.my_token:
            raise AgentException("not my round")
 
        best_move, best_score = self.minmax(connect4)
        return best_move
 
    def minmax(self, connect4, depth=4, maximizing=True):
        if depth == 0 or connect4.game_over:
            return None, self.heuristic_func(connect4, self.my_token)

        possible_drops = connect4.possible_drops()

        if not possible_drops:
            return None, 0 
        best_move = None
        if maximizing:
            best_score = -float('inf')
            for column in possible_drops:
                new_connect = copy.deepcopy(connect4)
                new_connect.drop_token(column)
                _, score = self.minmax(new_connect, depth - 1, not maximizing)
                if score > best_score:
                    best_move = column
                    best_score = score
        else:
            best_score = float('inf')
            for column in possible_drops:
                new_connect = copy.deepcopy(connect4)
                new_connect.drop_token(column)
                _, score = self.minmax(new_connect, depth - 1, not maximizing)
                if score < best_score:
                    best_move = column
                    best_score = score

        return best_move, best_score
