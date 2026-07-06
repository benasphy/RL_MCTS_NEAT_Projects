import numpy as np

class ContinuousLineEnv:
    """A truly continuous 1D environment."""
    def __init__(self):
        self.state = 0.0
        self.goal = 1.0
    
    def reset(self):
        self.state = 0.0
        return self.state
    
    def step(self, action):
        # Action is a forward push with a bit of noise
        step_size = np.random.uniform(0.05, 0.15)
        self.state = min(self.goal, self.state + step_size)
        
        reward = -1  # Every second spent costs time
        done = (self.state >= self.goal)
        return self.state, reward, done

class RBFFeatureExtractor:
    """Converts a single continuous float state into an RBF feature array."""
    def __init__(self):
        # Place 3 RBF anchors across the 1D track
        self.centers = np.array([0.0, 0.5, 1.0])
        self.sigma = 0.25 # Width of the Gaussian curves
    
    def get_features(self, state):
        # Calculate Gaussian activation for each anchor center
        features = np.exp(-((state - self.centers) ** 2) / (2 * self.sigma ** 2))
        return features

# --- Run Semi-Gradient TD(0) ---
env = ContinuousLineEnv()
extractor = RBFFeatureExtractor()

# Linear weights initialized to 0 (one weight per RBF center)
weights = np.zeros(3)
alpha = 0.05
gamma = 1.0
episodes = 500

print("--- Training Linear Semi-Gradient TD(0) Agent ---")