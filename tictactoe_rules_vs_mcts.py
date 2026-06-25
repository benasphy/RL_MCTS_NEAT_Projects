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

# ==========================================
# 3. MONTE CARLO TREE SEARCH APPROACH (Sampling)
# ==========================================
class MCTSNode:

    def __init__(self, board, player_to_move, parent=None, last_move=None):
        self.board = board
        self.player_to_move = player_to_move
        self.parent = parent
        self.last_move = last_move
        self.children = []
        self.visits = 0
        self.wins = 0.0


class MCTSAgent:

    def __init__(self, player, iterations=150):
        self.player = player
        self.opponent = -player
        self.iterations = iterations

    def select_move(self, board):
        root = MCTSNode(board, self.player)
        total_rollouts = 0

        for _ in range(self.iterations):
            node = root
            sim_board = copy.deepcopy(board)
            curr_player = self.player

            # --- Selection & Expansion ---
            # If nodes are already expanded, dig down. Otherwise, expand a new child.
            legal_moves = get_legal_moves(sim_board)
            if not is_win(sim_board, self.opponent) and legal_moves:
                # Simple expansion: if children don't match legal moves, create one
                if len(node.children) < len(legal_moves):
                    untried_moves = [
                        m for m in legal_moves if m not in [c.last_move for c in node.children]
                    ]
                    move = random.choice(untried_moves)
                    sim_board = make_move(sim_board, move, curr_player)
                    new_node = MCTSNode(
                        sim_board, -curr_player, parent=node, last_move=move
                    )
                    node.children.append(new_node)
                    node = new_node
                else:
                    # Select using simplified UCB1
                    node = max(
                        node.children,
                        key=lambda c: (c.wins / c.visits)
                        + 1.4 * math.sqrt(math.log(node.visits) / c.visits),
                    )
                    sim_board = node.board
                curr_player = node.player_to_move

            # --- Rollout (SAMPLING the opposing moves randomly) ---
            rollout_player = curr_player
            while not is_win(sim_board, PLAYER_X) and not is_win(sim_board, PLAYER_O) and get_legal_moves(sim_board):
                move = random.choice(get_legal_moves(sim_board))
                sim_board = make_move(sim_board, move, rollout_player)
                rollout_player = -rollout_player
                total_rollouts += 1

            # --- Backpropagation ---
            # Score outcome from perspective of Root Player (X)
            if is_win(sim_board, self.player):
                reward = 1.0
            elif is_win(sim_board, self.opponent):
                reward = 0.0
            else:
                reward = 0.5  # Draw

            while node is not None:
                node.visits += 1
                # If the parent node was the one who made the winning move, reward it
                node.wins += reward if node.player_to_move == self.opponent else (1 - reward)
                node = node.parent

        print(f" -> [MCTS Sampled] Ran {total_rollouts} random future moves across options.")
        # Pick the best child based on highest visit counts
        best_child = max(root.children, key=lambda c: c.visits)
        return best_child.last_move

# ==========================================
# 4. TESTING THE DIFFERENCE
# ==========================================
if __name__ == "__main__":
    # Setup a mid-game board state where:
    # - It's X's turn
    # - Nobody can win on this immediate turn
    # - Nobody needs to block on this immediate turn
    #
    #   X | . | O
    #  -----------
    #   . | O | .
    #  -----------
    #   X | . | .
    mid_game_board = [[1, 0, -1], [0, -1, 0], [1, 0, 0]]

    print("Initial Board Position (X's turn):")
    for row in mid_game_board:
        print(row)
    print("-" * 40)

    # 1. Test Traditional Rule Base
    traditional_agent = TraditionalRuleAgent(player=PLAYER_X)
    print("Executing Traditional Agent...")
    trad_move = traditional_agent.select_move(mid_game_board)
    print(f"Traditional Agent picked move: {trad_move}")

    print("-" * 40)

    # 2. Test MCTS Sampling
    mcts_agent = MCTSAgent(player=PLAYER_X, iterations=100)
    print("Executing MCTS Agent...")
    mcts_move = mcts_agent.select_move(mid_game_board)
    print(f"MCTS Agent picked move: {mcts_move}")