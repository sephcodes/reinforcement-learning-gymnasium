import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import time

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    
env = gym.make("CartPole-v1")
# Get number of actions from gym action space
n_actions = env.action_space.n
# Get the number of state observations
state, info = env.reset()
n_observations = len(state)

policy_net = DQN(n_observations, n_actions).to(device)

policy_net.load_state_dict(torch.load("cartpole_dqn.pth", map_location=device))
policy_net.eval()

episodes = 5
sleep = 0.02

env = gym.make("CartPole-v1", render_mode="human")

for episode in range(episodes):
    state, _ = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

    done = False
    total_reward = 0

    while not done:
        with torch.no_grad():
            action = policy_net(state).max(1).indices.item()

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward

        if not done:
            state = torch.tensor(
                next_state, dtype=torch.float32, device=device
            ).unsqueeze(0)

        time.sleep(sleep)  # slows down rendering for visibility

    print(f"Episode {episode + 1}: total reward = {total_reward}")

env.close()