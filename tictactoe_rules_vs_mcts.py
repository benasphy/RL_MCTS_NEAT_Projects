import copy
import math
import random

# ==========================================
# 1. TIC-TAC-TOE BOARD STATE MANAGEMENT
# ==========================================
# Representing players: 1 = X (AI), -1 = O (Opponent), 0 = Empty
EMPTY = 0
PLAYER_X = 1
PLAYER_O = -1


def is_win(board, player):
    # Check rows, columns, and diagonals
    for i in range(3):
        if all(board[i][j] == player for j in range(3)):
            return True
        if all(board[j][i] == player for j in range(3)):
            return True
    if board[0][0] == player and board[1][1] == player and board[2][2] == player:
        return True
    if board[0][2] == player and board[1][1] == player and board[2][0] == player:
        return True
    return False


def get_legal_moves(board):
    return [(r, c) for r in range(3) for c in range(3) if board[r][c] == EMPTY]


def make_move(board, move, player):
    new_board = copy.deepcopy(board)
    new_board[move[0]][move[1]] = player
    return new_board

# ==========================================
# 2. TRADITIONAL RULE-BASED APPROACH (Win -> Block -> Enumeration)
# ==========================================
class TraditionalRuleAgent:

    def __init__(self, player):
        self.player = player
        self.opponent = -player

    def find_win_or_block(self, board, target_player):
        """Helper to find if a specific player has an immediate winning line."""
        for move in get_legal_moves(board):
            sim_board = make_move(board, move, target_player)
            if is_win(sim_board, target_player):
                return move
        return None

    def select_move(self, board):
        # Rule 1: If a win move is available, take it
        winning_move = self.find_win_or_block(board, self.player)
        if winning_move:
            print(" -> [Rule Triggered] Immediate Win Found!")
            return winning_move

        # Rule 2: Else if a block move is available, take it
        blocking_move = self.find_win_or_block(board, self.opponent)
        if blocking_move:
            print(" -> [Rule Triggered] Immediate Block Required!")
            return blocking_move

        # Rule 3: Traditional Enumeration / Fallback
        # In a real engine, this would be a deep Minimax lookahead enumerating all branches,
        # or massive nested If-Then logic. Here we fallback to show it hit the dead-zone.
        print(" -> [Rule Failed] No Win/Block. Must fall back to raw tree enumeration.")
        return get_legal_moves(board)[0]  # Dumb fallback for illustration
