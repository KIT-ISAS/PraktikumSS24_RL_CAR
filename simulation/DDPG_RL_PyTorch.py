"""
DDPG Agent for RC Car Navigation with Pygame Visualization

This script implements a Deep Deterministic Policy Gradient (DDPG) agent to navigate
an RC car in a custom environment. It includes:
  - Actor and Critic neural networks
  - Replay buffer for experience storage
  - Main training loop with action selection, learning updates,
    and periodic model saving
  - Pygame-based visualization of the car, goal, and trajectory
"""

# === Imports ===
import pygame              # For real-time visualization
import numpy as np         # Numerical computations
import torch               # Deep learning framework
import torch.nn as nn      # Neural network modules
import torch.optim as optim# Optimizers
import random              # Random sampling for replay buffer
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
LR_ACTOR   = 1e-3   # Learning rate for actor network
LR_CRITIC  = 1e-3   # Learning rate for critic network
GAMMA      = 0.99   # Discount factor for future rewards
TAU        = 0.005  # Soft update rate for target networks
BUFFER_SIZE= int(1e5)   # Max size for replay buffer
BATCH_SIZE = 128     # Mini-batch size for training updates
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_STEPS  = 100000  # Total training steps

# === Neural Network Definitions ===
class Actor(nn.Module):
    """
    Policy network mapping state vectors to continuous actions.
    Architecture: [state_dim] -> 256 -> 256 -> [action_dim] with Tanh activation.
    """
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )

    def forward(self, x):
        """
        Forward pass through actor network.
        Args:
            x: tensor of shape (batch_size, state_dim)
        Returns:
            actions: tensor of shape (batch_size, action_dim) in [-1,1]
        """
        return self.net(x)

class Critic(nn.Module):
    """
    Q-value network estimating expected return for state-action pairs.
    Architecture: [state_dim+action_dim] -> 256 -> 256 -> 1.
    """
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, action):
        """
        Forward pass through critic network.
        Args:
            state: tensor (batch_size, state_dim)
            action: tensor (batch_size, action_dim)
        Returns:
            q_value: tensor (batch_size, 1)
        """
        return self.net(torch.cat([state, action], dim=1))

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

# === Pygame Visualization Helpers ===
WIDTH, HEIGHT = 800, 800           # Window dimensions
SCALE = 200                         # World-to-screen scaling factor
WHITE, BLUE, RED, BLACK, GREY = KIT_COLORS['white'], KIT_COLORS['blue'], KIT_COLORS['red'], KIT_COLORS['black'], (200,200,200)

def world_to_screen(pos):
    """Convert world (x,y) to screen pixels."""
    return int(WIDTH/2 + pos[0]*SCALE), int(HEIGHT/2 - pos[1]*SCALE)

def draw_vehicle(screen, pos, theta):
    """
    Draw triangular representation of car at position and orientation.
    pos: (x,y), theta: heading angle in radians.
    """
    x, y = world_to_screen(pos)
    length, width_ = 0.2*SCALE, 0.1*SCALE
    # Compute vertices
    front = (int(x + length*np.cos(theta)), int(y - length*np.sin(theta)))
    back_left = (int(x - length*0.7*np.cos(theta) + width_*np.sin(theta)), int(y + length*0.7*np.sin(theta) + width_*np.cos(theta)))
    back_right= (int(x - length*0.7*np.cos(theta) - width_*np.sin(theta)), int(y + length*0.7*np.sin(theta) - width_*np.cos(theta)))
    pygame.draw.polygon(screen, KIT_COLORS['green'], [front, back_left, back_right])

def draw_text(screen, font, text, x, y, color):
    """Render text on screen at pixel (x,y)."""
    screen.blit(font.render(text, True, color), (x, y))

# === Main Training and Visualization Loop ===
def main():
    """
    Initialize environment, networks, optimizers, and run DDPG training loop
    with Pygame visualization.
    """
    # Initialize Pygame
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("RL Car Navigation Viewer")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)

    # Create environment and get dims
    env = RCCarNavigationEnv()
    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # Instantiate networks and target copies
    actor = Actor(state_dim, action_dim).to(DEVICE)
    actor_target = Actor(state_dim, action_dim).to(DEVICE)
    actor_target.load_state_dict(actor.state_dict())

    critic = Critic(state_dim, action_dim).to(DEVICE)
    critic_target = Critic(state_dim, action_dim).to(DEVICE)
    critic_target.load_state_dict(critic.state_dict())

    # Set up optimizers and replay buffer
    actor_optim  = optim.Adam(actor.parameters(), lr=LR_ACTOR)
    critic_optim = optim.Adam(critic.parameters(), lr=LR_CRITIC)
    buffer = ReplayBuffer(BUFFER_SIZE)

    # Attempt to load pretrained actor weights
    try:
        actor.load_state_dict(torch.load("actor_200000.pth"))
        print("Model loaded.")
    except FileNotFoundError:
        print("No saved model found. Starting from scratch.")

    # Reset environment and initialize statistics
    obs, _ = env.reset()
    episode_reward = 0.0
    path           = []
    episode_len    = 0
    current_reward = 0.0
    goals_in_a_row = 0
    total_goals    = 0

    # Main loop over steps
    for step in range(1, MAX_STEPS+1):
        # Handle exit events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        # Select action according to current policy with exploration noise
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            action = actor(obs_tensor).cpu().numpy()[0]
        action += np.random.normal(0, 0.1, size=action_dim)
        action = np.clip(action, -1.0, 1.0)

        # Step environment
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Store transition in replay buffer
        buffer.push((obs, action, reward, next_obs, float(done)))
        obs = next_obs
        path.append(env.position.copy())
        episode_reward += reward
        episode_len += 1
        current_reward = reward

        # Check goal event
        if info.get("goal_reached", False):
            print(f"[Goal reached] Reward: {episode_reward:.2f}, Steps: {episode_len}")
            episode_reward = 0.0
            episode_len = 0
            path = []
            goals_in_a_row += 1
            total_goals += 1

        # Reset environment if episode done or too long
        if done or episode_len >= 1500:
            print(f"[Reset] Reward: {episode_reward:.2f}, Steps: {episode_len}")
            obs, _ = env.reset(set_new_goal=terminated)
            episode_reward = 0.0
            episode_len = 0
            path = []
            goals_in_a_row = 0  # reset consecutive success counter

        # Learning updates
        if len(buffer) > BATCH_SIZE:
            s, a, r, s2, d = buffer.sample(BATCH_SIZE)
            # Critic update
            with torch.no_grad():
                next_action = actor_target(s2)
                q_target = r + GAMMA * (1 - d) * critic_target(s2, next_action)
            q_val = critic(s, a)
            critic_loss = nn.MSELoss()(q_val, q_target)
            critic_optim.zero_grad()
            critic_loss.backward()
            critic_optim.step()
            # Actor update
            actor_loss = -critic(s, actor(s)).mean()
            actor_optim.zero_grad()
            actor_loss.backward()
            actor_optim.step()
            # Soft update target networks
            for tp, p in zip(actor_target.parameters(), actor.parameters()):
                tp.data.copy_(TAU * p.data + (1 - TAU) * tp.data)
            for tp, p in zip(critic_target.parameters(), critic.parameters()):
                tp.data.copy_(TAU * p.data + (1 - TAU) * tp.data)

        # Periodically save actor model
        if step % 10000 == 0:
            torch.save(actor.state_dict(), f"actor_{step}.pth")
            print(f"Model saved at step {step}")

        # Visualization: draw grid, goal, trajectory, vehicle, and stats
        screen.fill(KIT_COLORS['white'])
        # Draw world grid
        for i in range(0, WIDTH, SCALE):
            pygame.draw.line(screen, KIT_COLORS['blue'], (i, 0), (i, HEIGHT), 1)
        for j in range(0, HEIGHT, SCALE):
            pygame.draw.line(screen, KIT_COLORS['blue'], (0, j), (WIDTH, j), 1)
        # Draw goal
        pygame.draw.circle(screen, KIT_COLORS['red'], world_to_screen(env.goal), 8)
        # Draw path
        if len(path) > 1:
            for k in range(1, len(path)):
                pygame.draw.line(screen, KIT_COLORS['black'],
                                 world_to_screen(path[k-1]), world_to_screen(path[k]), 2)
        # Draw vehicle
        draw_vehicle(screen, env.position, env.theta)
        # Draw telemetry text
        draw_text(screen, font, f"Speed: {env.v:.2f} m/s", 10, 10, KIT_COLORS['black'])
        draw_text(screen, font, f"Steering: {env.last_steering:.2f} rad", 10, 35, KIT_COLORS['black'])
        draw_text(screen, font, f"Goal: ({env.goal[0]:.2f}, {env.goal[1]:.2f})", 10, 60, KIT_COLORS['black'])
        draw_text(screen, font, f"Position: ({env.position[0]:.2f}, {env.position[1]:.2f})", 10, 85, KIT_COLORS['black'])
        # Reward and success counters
        reward_color = KIT_COLORS['green'] if current_reward >= 0 else KIT_COLORS['red']
        draw_text(screen, font, f"Reward: {current_reward:.2f}", 10, 110, reward_color)
        draw_text(screen, font, f"Goals in a row: {goals_in_a_row}", 10, 135, KIT_COLORS['green'])
        draw_text(screen, font, f"Total goals: {total_goals}", 10, 160, KIT_COLORS['green'])

        # Update display and control frame rate
        pygame.display.flip()
        clock.tick(30)

# Entry point
if __name__ == "__main__":
    main()