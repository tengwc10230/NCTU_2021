import numpy as np


class Board:
    def __init__(self, player, opponent):
        """

        Player representation: (first_player: O, second_player: X)

        Args:
            player: an Object which has required functions
            opponent: an Object which has required functions

        """
        ## Public ##
        self.n_side = 8
        self.current_player = 0  # Record current_player ID, may be 1 or -1
        self.player_no = (
            0  # Record your player ID, if you go first this num will be 1 otherwise -1
        )
        self.opponent_no = 0  # Record AI ID
        self.total_step = 0
        self.directions = (
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        )
        self.isInside = (
            lambda r, c: True
            if (0 <= r < self.n_side and 0 <= c < self.n_side)
            else False
        )
        ## Private ##
        self.__player = player
        self.__opponent = opponent
        self.__valid_moves = {1: [], -1: []}
        self.__valid_moves_loc = []
        self.__state = None
        self.reset()

    def __set_state(self):
        self.__state = np.zeros((self.n_side, self.n_side))
        self.__state[self.n_side // 2 - 1, self.n_side // 2 - 1] = -1
        self.__state[self.n_side // 2, self.n_side // 2] = -1
        self.__state[self.n_side // 2 - 1, self.n_side // 2] = 1
        self.__state[self.n_side // 2, self.n_side // 2 - 1] = 1

    def reset(self):
        """
        重製局面

        """
        self.__set_state()
        self.current_player = 1
        self.total_step = 0
        self.__valid_moves = {1: [], -1: []}
        self.__valid_moves_loc = []
        for row, col in np.argwhere(self.__state != 0):
            for d in self.directions:
                this_row = row + d[0]
                this_col = col + d[1]
                if (
                    self.isInside(this_row, this_col)
                    and self.__state[this_row, this_col] == 0
                    and (this_row, this_col) not in self.__valid_moves_loc
                ):
                    self.__valid_moves_loc.append((this_row, this_col))
        self.__valid_moves[1] = self.compute_available_move(1)
        self.__valid_moves[-1] = self.compute_available_move(-1)

    def __action(self, action):
        """
        決定下棋的位置

        Args:
            action: A tuple of location on board (row, col)
        """
        self.total_step += 1
        row, col = action
        assert self.__state[row, col] == 0, "There has been already set"
        assert action in self.__valid_moves[self.current_player][:, :2], "wrong choose"

        self.__state[row, col] = self.current_player

        self.__valid_moves_loc.remove((row, col))

        for d in self.directions:
            this_row, this_col = row + d[0], col + d[1]
            if (
                self.isInside(this_row, this_col)
                and self.__state[this_row, this_col] == 0
                and (this_row, this_col) not in self.__valid_moves_loc
            ):
                self.__valid_moves_loc.append((this_row, this_col))

        flip_direction = np.where(
            (self.__valid_moves[self.current_player][:, :2] == [row, col]).all(axis=1)
        )
        for i in self.__valid_moves[self.current_player][flip_direction]:
            d = self.directions[i[-1]]
            this_row, this_col = row, col
            while True:
                this_row += d[0]
                this_col += d[1]

                if self.isInside(this_row, this_col):
                    if self.__state[this_row, this_col] == -self.current_player:
                        self.__state[this_row, this_col] = self.current_player
                    else:
                        break
                else:
                    break
        self.__valid_moves[1] = self.compute_available_move(1)
        self.__valid_moves[-1] = self.compute_available_move(-1)

    def compute_available_move(self, chose_player):
        """
        計算玩家能夠下棋的位置

        Args:
            chose_player: An integer to stand for player.
        Return:
            a list of (row, col)
        """
        valid_moves = []

        for row, col in self.__valid_moves_loc:
            for idx, d in enumerate(self.directions):
                this_row, this_col = row + d[0], col + d[1]
                if (
                    self.isInside(this_row, this_col)
                    and self.__state[this_row, this_col] == -chose_player
                ):
                    while True:
                        this_row += d[0]
                        this_col += d[1]
                        if self.isInside(this_row, this_col):
                            if self.__state[this_row, this_col] == chose_player:
                                valid_moves.append((row, col, idx))
                                break
                            elif self.__state[this_row, this_col] == -chose_player:
                                continue
                            else:
                                break
                        else:
                            break
        valid_moves = np.array(valid_moves)
        return valid_moves

    def is_game_finished(self, chose_player):
        """
        確認該局是否結束，若結束則確認 chose_player 是否獲勝。

        Args:
            chose_player: An integer to stand for player.
        Returns:
            (Boolean, Integer): (是否結束, 獲勝玩家 1: 黑棋, -1: 白棋, 0: 平手)
        """
        opponent = -chose_player
        chose_player_valid_moves = self.compute_available_move(chose_player)
        opponent_valid_moves = self.compute_available_move(opponent)

        if (
            chose_player_valid_moves.shape[0] == 0
            and opponent_valid_moves.shape[0] == 0
        ):
            state_count_chose = (self.__state == chose_player).sum()
            state_count_opponent = (self.__state == opponent).sum()

            if state_count_chose > state_count_opponent:
                return (True, chose_player, [state_count_chose, state_count_opponent])
            elif state_count_chose == state_count_opponent:
                return (True, 0, [state_count_chose, state_count_opponent])
            else:
                return (True, opponent, [state_count_chose, state_count_opponent])
        else:
            return (False, 0, [999, 999])

    def play(self, player_first=False):
        """
        一局遊戲的執行函式

        Args:
            player_first: Boolean
        """
        self.reset()
        isFinished = (False, None)
        players = [self.__player, self.__opponent]

        if player_first:
            offset = 1
            self.player_no = 1
            self.opponent_no = -1
        else:
            offset = 0
            self.player_no = -1
            self.opponent_no = 1

        while isFinished[0] == False:
            offset = (offset + 1) % 2
            current_player_ = players[offset]
            vaild = self.get_valid_state(self.current_player)
            if vaild != []:
                idx = current_player_.move(self.get_information(self.current_player))
                self.__action(idx)
            self.current_player = -self.current_player
            isFinished = self.is_game_finished(self.current_player)

        self.print_state()
        if isFinished[1] == self.player_no:
            return (True, isFinished[1], isFinished[2])
        else:
            return (False, isFinished[1], isFinished[2])

    def print_state(self):
        """
        輸出局面 O:第一位玩家, X:第二位玩家, #:未下棋位置

        e.g.
            ['X' 'X' 'O' 'X' 'X' 'X' 'O' 'O']
            ['X' 'X' 'X' 'X' 'X' 'X' 'O' 'O']
            ['X' 'X' 'X' 'X' 'X' 'X' 'X' 'O']
            ['O' 'O' 'X' 'X' 'X' 'X' 'X' 'O']
            ['O' 'X' 'O' 'X' 'O' 'X' 'X' 'O']
            ['X' 'O' 'O' 'O' 'O' 'O' 'O' 'O']
            ['O' 'X' 'X' 'O' 'X' 'O' 'O' 'O']
            ['O' 'O' 'O' 'X' 'O' 'O' 'O' 'O']
        """
        for i in self.__state:
            print(np.where(i == 1, "O", np.where(i == -1, "X", "#")))
        print("\n")

    def get_valid_state(self, chose_player) -> list:
        """
        取得玩家可以下棋的位置

        Args:
            chose_player: An integer to stand for player.
        return:
            a list of (row, col)
        """
        if self.__valid_moves[chose_player] != []:
            return np.unique(self.__valid_moves[chose_player][:, :2], axis=0)
        else:
            return []

    def get_information(self, chose_player) -> list:
        """
        給你一些用得上的資訊

        return:
            valid_moves ：可以走的步
            board_state ：當前棋盤狀態
            player_no   ：Player ID
            total_step  ：目前到第幾步
        """
        valid_moves = self.get_valid_state(chose_player)
        board_state = self.__state.copy()
        return [valid_moves, board_state, self.player_no, self.total_step]
