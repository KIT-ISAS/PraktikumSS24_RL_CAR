"""
SAC Agent for RC Car Navigation with Pygame Visualization

This script implements a Soft Actor-Critic (SAC) agent to navigate an RC car in a custom
environment. It includes:
  - Replay buffer for experience storage
  - Stochastic actor network outputting mean and standard deviation for actions
  - Twin Q-function critic networks with target updates
  - Automatic entropy adjustment via alpha parameter
  - Main training loop with action selection, learning updates, and periodic model saving
  - Pygame-based visualization of the car, goal, and trajectory
"""

# === Imports ===
import pygame                # For real-time visualization
import numpy as np           # Numerical computations
import torch                 # Deep learning framework
import torch.nn as nn        # Neural network modules
import torch.nn.functional as F  # Functional API for losses and activations
import torch.optim as optim  # Optimizers
import random                # Random sampling for replay buffer
from collections import deque  # Double-ended queue for experience storage
from envs.rc_car_env import RCCarNavigationEnv  # Custom RC car environment

# === KIT Color Palette ===
KIT_COLORS = {
    'white':   (255, 255, 255),
    'black':   (0, 0, 0),
    'green':   (0, 150, 130),
    'blue':    (70, 100, 170),
    'maygreen':(140, 182, 60),
    'yellow':  (252, 229, 0),
    'orange':  (223, 155, 27),
    'brown':   (167, 130, 46),
    'red':     (162, 34, 35),
    'purple':  (163, 16, 124),
    'cyan':    (35, 161, 224)
}

# === Hyperparameters ===
GAMMA        = 0.99      # Discount factor for future rewards
TAU          = 0.005     # Soft update rate for target networks
BUFFER_SIZE  = int(1e5)  # Max size for replay buffer
BATCH_SIZE   = 128       # Mini-batch size for training updates
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_STEPS    = 100000    # Total training steps
LOG_STD_MIN  = -20       # Minimum log standard deviation
LOG_STD_MAX  = 2         # Maximum log standard deviation

# === Replay Buffer ===
class ReplayBuffer:
    """
    Circular buffer storing experience tuples for sampling.
    Each entry: (state, action, reward, next_state, done_flag)
    """
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, transition):
        """Add a new experience to the buffer."""
        self.buffer.append(transition)

    def sample(self, batch_size):
        """
        Sample a random batch of experiences.
        Returns tensors ready for training.
        """
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (
            torch.tensor(state, dtype=torch.float32).to(DEVICE),
            torch.tensor(action, dtype=torch.float32).to(DEVICE),
            torch.tensor(reward, dtype=torch.float32).unsqueeze(1).to(DEVICE),
            torch.tensor(next_state, dtype=torch.float32).to(DEVICE),
            torch.tensor(done, dtype=torch.float32).unsqueeze(1).to(DEVICE)
        )

    def __len__(self):
        """Return current buffer size."""
        return len(self.buffer)

# === Actor Network ===
class Actor(nn.Module):
    """
    Stochastic policy network outputting action distribution parameters.
    Architecture: two hidden layers of size 256, outputs mean and log_std.
    """
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.mean = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)

    def forward(self, state):
        """
        Forward pass returning Gaussian distribution parameters.
        Args:
            state: tensor of shape (batch_size, state_dim)
        Returns:
            mean: tensor (batch_size, action_dim)
            std:  tensor (batch_size, action_dim)
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x).clamp(LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        return mean, std

    def sample(self, state):
        """
        Sample action via the reparameterization trick and compute log-probability.
        Args:
            state: tensor (batch_size, state_dim)
        Returns:
            action: tensor (batch_size, action_dim)
            log_prob: tensor (batch_size, 1)
        """
        mean, std = self.forward(state)
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # Reparameterized sample
        y_t = torch.tanh(x_t)   # Squash to [-1,1]
        log_prob = normal.log_prob(x_t) - torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return y_t, log_prob

# === Critic Network ===
class Critic(nn.Module):
    """
    Twin Q-value networks estimating expected return.
    Each has two hidden layers of size 256, outputs a single Q-value.
    """
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, action):
        """
        Forward pass returning both Q-values.
        Args:
            state: tensor (batch_size, state_dim)
            action: tensor (batch_size, action_dim)
        Returns:
            q1, q2: tensors each of shape (batch_size, 1)
        """
        x = torch.cat([state, action], dim=-1)
        return self.q1(x), self.q2(x)

# === SAC Agent ===
class SACAgent:
    """
    Encapsulates actor, critic, target networks, and training logic.
    Includes automatic entropy tuning.
    """
    def __init__(self, state_dim, action_dim):
        # Initialize networks
        self.actor = Actor(state_dim, action_dim).to(DEVICE)
        self.critic = Critic(state_dim, action_dim).to(DEVICE)
        self.critic_target = Critic(state_dim, action_dim).to(DEVICE)
        self.critic_target.load_state_dict(self.critic.state_dict())
        # Optimizers
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=3e-4)
        # Entropy coefficient alpha and optimizer
        self.log_alpha = torch.tensor([0.0], requires_grad=True, device=DEVICE)
        self.alpha_optim = optim.Adam([self.log_alpha], lr=3e-4)
        self.target_entropy = -action_dim

    def select_action(self, state, evaluate=False):
        """
        Choose action for given state.
        Args:
            state: numpy array of shape (state_dim,)
            evaluate: bool, if True choose deterministic mean action
        Returns:
            action: numpy array of shape (action_dim,)
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        if evaluate:
            mean, _ = self.actor(state)
            return torch.tanh(mean).cpu().numpy()[0]
        else:
            action, _ = self.actor.sample(state)
            return action.cpu().detach().numpy()[0]

    def train(self, replay_buffer):
        """
        Perform one gradient update step for critic, actor, and alpha.
        Args:
            replay_buffer: instance of ReplayBuffer with stored experiences
        """
        if len(replay_buffer) < BATCH_SIZE:
            return
        state, action, reward, next_state, done = replay_buffer.sample(BATCH_SIZE)
        alpha = self.log_alpha.exp().detach()
        # Compute target Q-value
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_state)
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2) - alpha * next_log_prob
            target_value = reward + (1 - done) * GAMMA * target_q
        # Critic loss
        current_q1, current_q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_q1, target_value) + F.mse_loss(current_q2, target_value)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()
        # Actor loss
        new_action, log_prob = self.actor.sample(state)
        q1_pi, q2_pi = self.critic(state, new_action)
        q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (alpha * log_prob - q_pi).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        # Alpha loss for entropy tuning
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()
        # Soft update target critic networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

# === Pygame Setup ===
WIDTH, HEIGHT = 800, 800  # Window dimensions
SCALE = 200               # World-to-screen scaling factor

def world_to_screen(pos):
    """Convert world (x,y) to screen pixels."""
    return int(WIDTH/2 + pos[0]*SCALE), int(HEIGHT/2 - pos[1]*SCALE)

def draw_vehicle(screen, pos, theta):
    """
    Draw triangular representation of car at position and orientation.
    pos: (x,y), theta: heading angle in radians.
    """
    x, y = world_to_screen(pos)
    l, w = 0.2*SCALE, 0.1*SCALE
    front = (int(x + l*np.cos(theta)), int(y - l*np.sin(theta)))
    back_left = (int(x - l*0.7*np.cos(theta) + w*np.sin(theta)), int(y + l*0.7*np.sin(theta) + w*np.cos(theta)))
    back_right = (int(x - l*0.7*np.cos(theta) - w*np.sin(theta)), int(y + l*0.7*np.sin(theta) - w*np.cos(theta)))
    pygame.draw.polygon(screen, KIT_COLORS['green'], [front, back_left, back_right])

def draw_text(screen, font, text, x, y, color):
    """Render text on screen at pixel (x,y)."""
    screen.blit(font.render(text, True, color), (x, y))

# === Main Training Loop ===
def main():
    """
    Initialize environment, agent, replay buffer, and run SAC training loop
    with Pygame visualization.
    """
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("SAC Car Navigation Viewer")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)

    env = RCCarNavigationEnv()
    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = SACAgent(state_dim, action_dim)
    buffer = ReplayBuffer(BUFFER_SIZE)

    try:
        agent.actor.load_state_dict(torch.load("sac_actor.pth"))
        print("Model loaded.")
    except FileNotFoundError:
        print("No saved model found. Starting from scratch.")

    obs, _ = env.reset()
    episode_reward = 0.0
    path           = []
    episode_len    = 0
    current_reward = 0.0
    goals_in_a_row = 0
    total_goals    = 0

    for step in range(1, MAX_STEPS + 1):
        # Handle exit events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        # Select action and step environment
        action = agent.select_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        buffer.push((obs, action, reward, next_obs, float(done)))
        obs = next_obs
        current_reward = reward
        path.append(env.position.copy())
        episode_reward += reward
        episode_len += 1

        # Agent learning update
        agent.train(buffer)

        # Handle goal and reset logic
        if info.get("goal_reached", False):
            print(f"[Goal reached] Reward: {episode_reward:.2f}, Steps: {episode_len}")
            goals_in_a_row += 1
            total_goals += 1
        if done or (episode_len >= 1500 and not info.get("goal_reached", False)):
            print(f"[Reset] Reward: {episode_reward:.2f}, Steps: {episode_len}")
            obs, _ = env.reset(set_new_goal=terminated)
            episode_reward, path, episode_len = 0.0, [], 0
            goals_in_a_row = 0

        # Periodically save actor model
        if step % 10000 == 0:
            torch.save(agent.actor.state_dict(), "sac_actor.pth")
            print(f"Model saved at step {step}")

        # Visualization
        screen.fill(KIT_COLORS['white'])
        for i in range(0, WIDTH, SCALE):
            pygame.draw.line(screen, KIT_COLORS['blue'], (i, 0), (i, HEIGHT), 1)
        for j in range(0, HEIGHT, SCALE):
            pygame.draw.line(screen, KIT_COLORS['blue'], (0, j), (WIDTH, j), 1)
        pygame.draw.circle(screen, KIT_COLORS['red'], world_to_screen(env.goal), 8)
        if len(path) > 1:
            for k in range(1, len(path)):
                pygame.draw.line(screen, KIT_COLORS['black'], world_to_screen(path[k-1]), world_to_screen(path[k]), 2)
        draw_vehicle(screen, env.position, env.theta)
        draw_text(screen, font, f"Speed: {env.v:.2f} m/s", 10, 10, KIT_COLORS['black'])
        draw_text(screen, font, f"Steering: {env.last_steering:.2f} rad", 10, 35, KIT_COLORS['black'])
        draw_text(screen, font, f"Goal: ({env.goal[0]:.2f}, {env.goal[1]:.2f})", 10, 60, KIT_COLORS['black'])
        draw_text(screen, font, f"Position: ({env.position[0]:.2f}, {env.position[1]:.2f})", 10, 85, KIT_COLORS['black'])
        reward_color = KIT_COLORS['green'] if current_reward >= 0 else KIT_COLORS['red']
        draw_text(screen, font, f"Reward: {current_reward:.2f}", 10, 110, reward_color)
        draw_text(screen, font, f"Goals in a row: {goals_in_a_row}", 10, 135, KIT_COLORS['green'])
        draw_text(screen, font, f"Total goals: {total_goals}", 10, 160, KIT_COLORS['green'])
        pygame.display.flip()
        clock.tick(30)

if __name__ == "__main__":
    main()
