from random import choice, choices, randint
from numpy import Infinity, ndarray
from numpy.core.fromnumeric import choose
import time
import copy
import numpy as np
from Board import Board

class Player:
    def __init__(self):
        self.name = "Player"
        self.player_result = {}
        self.opponent_result = {}
        self.weight_board = np.array([[ 40, -6, 22, 16, 16, 22, -6, 40],
                                      [ -6,-14, -8,  2,  2, -8,-14, -6],
                                      [ 22, -8,  4,  4,  4,  4, -8, 22],
                                      [ 16,  2,  4, -6, -6,  4,  2, 16],
                                      [ 16,  2,  4, -6, -6,  4,  2, 16],
                                      [ 22, -8,  4,  4,  4,  4, -8, 22],
                                      [ -6,-14, -8,  2,  2, -8,-14, -6],
                                      [ 40, -6, 22, 16, 16, 22, -6, 40]])
            
    def set_board(self, board_inf):
        cur_board = Board(None, None)
        cur_board._Board__valid_moves = {1: [], -1: []}
        cur_board._Board__valid_moves_loc = []
        cur_board._Board__state = board_inf[1]
        cur_board.current_player = board_inf[2]
        cur_board.player_no = board_inf[2]
        cur_board.opponent_no = -board_inf[2]
        cur_board.total_step = board_inf[3]

        for row in range(8):
            for col in range(8):
                if board_inf[1][row, col] != 0:
                    for d in cur_board.directions:
                        this_row, this_col = row + d[0], col + d[1]
                        if (
                            cur_board.isInside(this_row, this_col)
                            and board_inf[1][this_row, this_col] == 0
                            and (this_row, this_col) not in cur_board._Board__valid_moves_loc
                        ):
                            cur_board._Board__valid_moves_loc.append((this_row, this_col))
            
        cur_board._Board__valid_moves[1] = cur_board.compute_available_move(1)
        cur_board._Board__valid_moves[-1] = cur_board.compute_available_move(-1)
        return cur_board

    def get_score(self, cur_board):
        player_point = (cur_board._Board__state == cur_board.player_no).sum()
        opponent_point = (cur_board._Board__state == cur_board.opponent_no).sum()
        
        player_move = cur_board._Board__valid_moves[cur_board.player_no]
        opponent_move = cur_board._Board__valid_moves[cur_board.opponent_no]

        draw = (player_point == opponent_point) and len(player_move) == 0 and len(opponent_move) == 0
        if player_point == 0 or draw:
            return -5000
        if opponent_point == 0:
            return 5000
        
        player_weight = 0
        opponent_weight = 0
        for move in player_move:
            player_weight += self.weight_board[move[0]][move[1]]
        for move in opponent_move:
            opponent_weight += self.weight_board[move[0]][move[1]]

        score = int((player_point - opponent_point) * cur_board.total_step / 10)
        score += int((player_weight - opponent_weight) * 2) 
        return score
        
    def alphabeta(self, cur_board, isMax=True, depth_left=3, alpha=-Infinity, beta=Infinity):
        #print(cur_board._Board__state)
        if cur_board.is_game_finished(cur_board.current_player)[0] or depth_left == 0:
            return self.get_score(cur_board), None

        valid_moves = cur_board.get_valid_state(cur_board.current_player)
        best_move = None
        best_score = -Infinity if isMax else Infinity

        if isMax:
            if valid_moves == []:
                next_board = copy.deepcopy(cur_board)
                next_board.current_player = -next_board.current_player
                score, _ = self.alphabeta(next_board, not isMax, depth_left-1, alpha, beta) 
                if score > best_score:
                    best_score, best_move = score, None
                if score >= beta:
                    return best_score, best_move
                alpha = max(alpha, score)

            for move in valid_moves:
                next_board = copy.deepcopy(cur_board)
                next_board._Board__action(move)
                next_board.current_player = -next_board.current_player
                
                score, _ = self.alphabeta(next_board, not isMax, depth_left-1, alpha, beta) 
                score += self.weight_board[move[0], move[1]]

                if score > best_score:
                    best_score, best_move = score, move
                if score >= beta:
                    break
                alpha = max(alpha, score)

        else:
            if valid_moves == []:
                next_board = copy.deepcopy(cur_board)
                next_board.current_player = -next_board.current_player
                score, _ = self.alphabeta(next_board, not isMax, depth_left-1, alpha, beta) 
                if score < best_score:
                    best_score, best_move = score, None
                if score <= alpha:
                    return best_score, best_move
                beta = min(beta, score)

            for move in valid_moves:
                next_board = copy.deepcopy(cur_board)
                next_board._Board__action(move)
                next_board.current_player = -next_board.current_player

                score, _ = self.alphabeta(next_board, not isMax, depth_left-1, alpha, beta) 
                score += self.weight_board[move[0], move[1]]

                if score < best_score:
                    best_score, best_move = score, move
                if score <= alpha:
                    break
                beta = min(beta, score)
        #print(best_score, best_move)
        return best_score, best_move

    def move(self, board_inf):
        """
        Args:
            board_inf:
                 [0] - valid_moves: 可以下的地方，它會是一個二維的 list ex:[[1,2],[2,2]]
                 [1] - board_state: 當前棋盤狀況
                 [2] - player_no  : 你先攻的話就是 1(O),反之 -1(X)
                 [3] - total_step : 現在下到第幾步 (Hint: 對於黑白棋而言，解尾盤比較重要)
        return:
            your moves: 你要下哪裡，它會是一個一維的 list ex:[1,2]
        """
        if board_inf[3] < 32:
            dl = 1
        elif board_inf[3] < 48:
            dl = 4
        else:
            dl = 6
        return self.alphabeta(self.set_board(board_inf), depth_left=dl)[1]