from random import choice


class RandomAI:
    def __init__(self):
        self.name = "RandomAI"

    def move(self, board_inf):
        return choice(board_inf[0])
