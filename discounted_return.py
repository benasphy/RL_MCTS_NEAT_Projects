import numpy as np

def calculate_discounted_return(rewards, gamma):
    g_0 = 0
    
    for t, reward in enumerate(rewards):
        discounted_reward = (gamma ** t) * reward
        g_0 += discounted_reward
    return g_0

# --- Warehouse Robot Scenario ---
# Timeline: 
# Second 1: Safe travel (+1)
# Second 2: Safe travel (+1)
# Second 3: Delivered package! (+10)
robot_rewards = [1, 1, 10]

# Evaluate under different discount factors
gammas = [0.1, 0.9, 1.0]

print("--- Warehouse Robot Return Evaluation ---")
print(f"Sequence of rewards received: {robot_rewards}\n")

for g in gammas:
    total_return = calculate_discounted_return(robot_rewards, gamma=g)
    
    if g == 0.1:
        description = "Short-sighted (Values immediate safety, barely cares about final delivery)"
    elif g == 0.9:
        description = "Balanced/Efficient (Values delivery, but prefers doing it quickly)"
    else:
        description = "Far-sighted (Values total accumulation, indifferent to time delays)"
        
    print(f"Gamma (γ) = {g} -> Total Return (G_0): {total_return:.2f}")
    print(f"  Character: {description}\n")