import numpy as np
import matplotlib.pyplot as plt

class MultiArmedBandit:
    def __init__(self, k=10):
        self.k = k
        
        self.true_values = np.random.normal(0, 1, self.k)
        self.best_action = np.argmax(self.true_values)
        
    def step(self, action):
        reward = self.true_values[action] + np.random.normal(0, 1)
        regret = self.true_values[self.best_action] - self.true_values[action]
        return reward, regret

class EpsilonGreedyAgent:
    def __init__(self, k=10, epsilon=0.1):
        self.k = k
        self.epsilon = epsilon
        self.q_estimates = np.zeros(k)
        self.action_counts = np.zeros(k)
        
    def select_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.k)
        else:
            # Exploit: choose the best estimated action (break ties randomly)
            return np.random.choice(np.flatnonzero(self.q_estimates == self.q_estimates.max()))
            
    def update(self, action, reward):
        self.action_counts[action] += 1
        # Incremental update rule: Q(a) <- Q(a) + 1/N(a) * [R - Q(a)]
        step_size = 1.0 / self.action_counts[action]
        self.q_estimates[action] += step_size * (reward - self.q_estimates[action])

# Simulation Parameters
num_arms = 10
steps = 1000
runs = 500  # Average over 500 independent runs to smooth out noise
epsilons = [0.0, 0.01, 0.1, 0.5]
results = {eps: np.zeros(steps) for eps in epsilons}

# Run Experiment
for eps in epsilons:
    cumulative_regret = np.zeros(steps)
    
    for run in range(runs):
        bandit = MultiArmedBandit(k=num_arms)
        agent = EpsilonGreedyAgent(k=num_arms, epsilon=eps)
        run_regret = 0
        
        for step in range(steps):
            action = agent.select_action()
            reward, regret = bandit.step(action)
            agent.update(action, reward)
            
            run_regret += regret
            cumulative_regret[step] += run_regret
            
    # Average the cumulative regret across all runs
    results[eps] = cumulative_regret / runs
#Plotting the Regret Over Time
plt.figure(figsize=(10, 6))
for eps, regret_history in results.items():
    plt.plot(regret_history, label=f'$\epsilon$ = {eps}')

plt.title('Cumulative Regret Over Time in a Multi-Armed Bandit')
plt.xlabel('Steps')
plt.ylabel('Total Cumulative Regret')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
