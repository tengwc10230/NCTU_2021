import numpy as np
from random import choice


class RandomAI:
    def __init__(self):
        self.name = "Random_AI"

    def get_valid_move(self, board_status) -> list:
        return np.where(board_status == 0)[0]

    def random_move(self, board_status) -> int:
        return choice(self.get_valid_move(board_status))

    def move(self, board_status) -> int:
        return self.random_move(board_status)


class Judge:
    def __init__(self, who_Turn):
        self.n_player_lose = 0
        self.n_player_win = 0
        self.tie = 0
        self.who_Turn = who_Turn

    def is_game_finished(self, board_status):
        result = check_win(board_status)
        if result == 0:
            return False, 0
        if result == -1:
            return True, -1
        return True, 1


class Board:
    def __init__(self, player, opponent, judge):
        self.board_status = None
        self.player = player
        self.opponent = opponent
        self.judge = judge

    def first_move(self) -> int:
        """
        To Any Player, their first move we set to random
        """
        return choice(range(9))

    def show_available_move(self):
        return np.where(self.board_status == 0)[0]

    def play(self):
        # Start empty board
        self.board_status = np.zeros((9,))

        # 1 的話玩家先走 ,-1 的話 AI 先走
        if self.judge.who_Turn == 1:
            self.board_status[self.first_move()] = 1
        else:
            self.board_status[self.first_move()] = -1
        self.judge.who_Turn *= -1

        # 下完第一步之後，接下來最多走 8 步
        for i in range(8):
            if self.judge.who_Turn == 1:
                current_player = self.player
            else:
                current_player = self.opponent

            # 1 是玩家下的地方，-1是對手下的地方
            self.board_status[
                current_player.move(self.board_status)
            ] = self.judge.who_Turn
            self.judge.who_Turn *= -1
            isWin, result = self.judge.is_game_finished(self.board_status)

            # AI or Player Win
            if i >= 3 and isWin:
                if result == -1:
                    self.judge.n_player_lose += 1
                    self.draw_game(f"AI WIN {self.judge.n_player_lose}:")
                if result == 1:
                    self.judge.n_player_win += 1
                    self.draw_game(f"PLAYER　WIN {self.judge.n_player_win}:")
                break
            # Tie
            if result == 0 and len(np.where(self.board_status == 0)[0]) == 0:
                self.judge.tie += 1
                self.draw_game(f"Tie {self.judge.tie}:")
                break

    def draw_game(self, who):
        map_ = {-1: "X", 1: "O", None: " - "}
        res = np.array([map_.get(v) for v in self.board_status]).reshape(3, 3)
        res[res == None] = "-"
        print(f"{who}\n{res}\n")


def check_win(board_status, size=3) -> int:

    if board_status[4] != 0:
        if (
            board_status[0] == board_status[4] == board_status[8]
            or board_status[2] == board_status[4] == board_status[6]
        ):
            return board_status[4]

    for i in range(size):
        # row
        if (
            board_status[i * 3] != 0
            and board_status[i * 3]
            == board_status[i * 3 + 1]
            == board_status[i * 3 + 2]
        ):
            return board_status[i * 3]
            # column
        if (
            board_status[i] != 0
            and board_status[i] == board_status[i + 3 * 1] == board_status[i + 3 * 2]
        ):
            return board_status[i]
        # diagonal
    return 0
