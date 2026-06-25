import math
import random



# 1. THE MDP ENVIRONMENT (A Simple 3x3 Grid)

class GridWorldMDP:

    def __init__(self):
        # 0: Empty, 1: Goal (+10), 2: Trap (-10)
        self.grid = [[0, 0, 1], [0, 0, 2], [0, 0, 0]]
        self.start_state = (0, 0)

    def get_actions(self, state):
        # Up, Down, Left, Right
        return ["U", "D", "L", "R"]

    def is_terminal(self, state):
        r, c = state
        return self.grid[r][c] != 0

    def get_reward(self, state):
        r, c = state
        if self.grid[r][c] == 1:
            return 10.0
        if self.grid[r][c] == 2:
            return -10.0
        return -0.1  # Small step penalty to find the fastest path

    def transition(self, state, action):
        """Stochastic transition: 80% chance of success,

        10% chance of slipping left, 10% right.
        """
        if self.is_terminal(state):
            return state

        r, c = state
        # Map actions to direction vectors
        moves = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}

        # Determine actual action due to stochastic "slipping"
        rand = random.random()
        if rand < 0.8:
            actual_action = action
        elif rand < 0.9:
            # Slip left-ish relative to choice
            actual_action = (
                "L" if action in ["U", "D"] else "U"
            )  # simplified slip
        else:
            actual_action = "R" if action in ["U", "D"] else "D"

        dr, dc = moves[actual_action]
        new_r, new_c = max(0, min(2, r + dr)), max(0, min(2, c + dc))
        return (new_r, new_c)


# 2. CHANG ET AL. ALGORITHM (Depth-First, No Cache)
class ChangEtAlSolver:

    def __init__(self, mdp, depth_limit=4, num_samples=15):
        self.mdp = mdp
        self.depth_limit = depth_limit
        self.num_samples = num_samples
        self.call_count = 0  # To track computational effort

    def evaluate_state(self, state, depth):
        self.call_count += 1
        if self.mdp.is_terminal(state) or depth >= self.depth_limit:
            return self.mdp.get_reward(state)

        actions = self.mdp.get_actions(state)
        action_values = []

        for action in actions:
            # Sample this action multiple times to get a robust average
            sample_rewards = []
            for _ in range(self.num_samples):
                next_state = self.mdp.transition(state, action)
                reward = self.mdp.get_reward(next_state)
                # Recursively look deeper downstream
                future_value = self.evaluate_state(next_state, depth + 1)
                sample_rewards.append(reward + future_value)

            # Average the sampled outcomes up
            action_values.append(sum(sample_rewards) / len(sample_rewards))

        # Chang et al. found that blending max/averages worked,
        # but for direct move selection we return the best action's value
        return max(action_values)

    def select_best_move(self, state):
        actions = self.mdp.get_actions(state)
        best_action = None
        best_value = float("-inf")

        for action in actions:
            # Evaluate each branch recursively from scratch
            sample_rewards = []
            for _ in range(self.num_samples):
                next_state = self.mdp.transition(state, action)
                val = self.mdp.get_reward(next_state) + self.evaluate_state(
                    next_state, 1
                )
                sample_rewards.append(val)
            avg_val = sum(sample_rewards) / len(sample_rewards)

            if avg_val > best_value:
                best_value = avg_val
                best_action = action

        return best_action



# 3. UCT ALGORITHM (Persistent Search Tree)

class UCTNode:

    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = {}  # action -> UCTNode
        self.visit_count = 0
        self.total_value = 0.0


class UCTSolver:

    def __init__(self, mdp, exploration_constant=2.0):
        self.mdp = mdp
        self.c = exploration_constant
        self.root = None
        self.call_count = 0  # To track computational effort

    def run_simulations(self, initial_state, iterations=150):
        # We explicitly preserve the root and its growing tree across iterations
        self.root = UCTNode(initial_state)

        for _ in range(iterations):
            node = self.root
            state = initial_state
            visited_nodes = [node]

            # --- 1. SELECTION (Using saved tree data & UCB1) ---
            while node.children and not self.mdp.is_terminal(state):
                self.call_count += 1
                actions = self.mdp.get_actions(state)

                # Check for unexpanded actions first
                unexpanded = [a for a in actions if a not in node.children]
                if unexpanded:
                    action = random.choice(unexpanded)
                    state = self.mdp.transition(state, action)
                    # --- 2. EXPANSION ---
                    new_node = UCTNode(state, parent=node)
                    node.children[action] = new_node
                    node = new_node
                    visited_nodes.append(node)
                    break
                else:
                    # All actions explored at least once; use UCB1 formula
                    best_uct = float("-inf")
                    best_action = None
                    for action, child in node.children.items():
                        # UCB1 Formula
                        exploitation = child.total_value / child.visit_count
                        exploration = self.c * math.sqrt(
                            math.log(node.visit_count) / child.visit_count
                        )
                        uct_value = exploitation + exploration

                        if uct_value > best_uct:
                            best_uct = uct_value
                            best_action = action
                    state = self.mdp.transition(state, best_action)
                    node = node.children[best_action]
                    visited_nodes.append(node)

            # --- 3. ROLLOUT (Simulation without storing structural nodes) ---
            depth = 0
            total_reward = self.mdp.get_reward(state)
            while not self.mdp.is_terminal(state) and depth < 4:
                action = random.choice(self.mdp.get_actions(state))
                state = self.mdp.transition(state, action)
                total_reward += self.mdp.get_reward(state)
                depth += 1

            # --- 4. BACKPROPAGATION (Averages propagated upwards) ---
            for n in visited_nodes:
                n.visit_count += 1
                n.total_value += total_reward

    def select_best_move(self):
        # Move with highest visit count or highest average score wins
        best_action = None
        best_avg = float("-inf")
        for action, child in self.root.children.items():
            if child.visit_count > 0:
                avg_val = child.total_value / child.visit_count
                if avg_val > best_avg:
                    best_avg = avg_val
                    best_action = action
        return best_action


# ==========================================
# 4. EXECUTION AND COMPARISON
# ==========================================
if __name__ == "__main__":
    # Initialize our environment
    env = GridWorldMDP()
    start = env.start_state

    print("--- Running Chang et al. (2005) Approach ---")
    chang_solver = ChangEtAlSolver(env, depth_limit=3, num_samples=10)
    chang_move = chang_solver.select_best_move(start)
    print(f"Selected Move: {chang_move}")
    print(f"Total recursive state evaluations: {chang_solver.call_count}")
    print(
        "Note: Every time it re-encountered a state, it completely recalculated it."
    )

    print("\n" + "=" * 50 + "\n")

    print("--- Running UCT (2006) Approach ---")
    uct_solver = UCTSolver(env)
    # Give UCT 120 total iterations to search
    uct_solver.run_simulations(start, iterations=120)
    uct_move = uct_solver.select_best_move()
    print(f"Selected Move: {uct_move}")
    print(f"Total tree node lookups/visits: {uct_solver.call_count}")
    print(
        "Note: It remembered shared board structures and recycled information across simulations."
    )
