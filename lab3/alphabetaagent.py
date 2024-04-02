from exceptions import AgentException
import math 

class AlphaBetaAgent:
    def __init__(self, token):
        self.my_token = token

    def situation(self,connect4):
        fieldRestriction = [(0, 0), (0, connect4.width - 1), (connect4.height - 1, 0), (connect4.height - 1, connect4.width - 1)]
        count = sum(1 for row, col in fieldRestriction if connect4.board[row][col] == self.my_token)
        return 0.2 * count
    
    def alphabeta(self,connect4,player,depth,alpha,beta):
        if(self.my_token == connect4.wins):
            return 1
        if(self.my_token != connect4.wins and connect4.wins is not None):
            return -1
        if((connect4.game_over is True) and (connect4.wins is None)):
            return 0
        if(depth==0):
            return self.situation(connect4)
        alphaBuffer = alpha
        betaBuffer = beta
        best_move = -math.inf if player == 1 else math.inf
        for column in connect4.possible_drops():
            connect4.drop_token(column)
            score = self.alphabeta(connect4,1-player,depth-1,alphaBuffer,betaBuffer)
            connect4.undo_last_move()
            if player == 1:
                best_move = max(score, best_move)
                alphaBuffer = max(alphaBuffer,best_move)
                if best_move >= betaBuffer:
                    break
            if player == 0:
                best_move = min(score, best_move)
                betaBuffer = min(betaBuffer,best_move)
                if best_move <= alphaBuffer:
                    break
        return best_move
    
    def decide(self,connect4):
        if connect4.who_moves != self.my_token:
            raise AgentException('I can not move')
        player = 1
        depth = 4
        best_move = -math.inf
        alpha = -math.inf
        best_state = 0
        possible_drops = connect4.possible_drops()
        for column in possible_drops:
            connect4.drop_token(column)
            score = self.alphabeta(connect4,player,depth-1,alpha,math.inf)
            connect4.undo_last_move()
            if(score > best_move):
                best_move=score
                best_state = possible_drops.index(column)
        return possible_drops[best_state]