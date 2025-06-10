import pygame
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from envs.rc_car_env import RCCarNavigationEnv

# === KIT Color Palette ===
KIT_COLORS = {
    'white': (255, 255, 255),
    'black': (0, 0, 0),
    'green': (0, 150, 130),
    'blue': (70, 100, 170),
    'maygreen': (140, 182, 60),
    'yellow': (252, 229, 0),
    'orange': (223, 155, 27),
    'brown': (167, 130, 46),
    'red': (162, 34, 35),
    'purple': (163, 16, 124),
    'cyan': (35, 161, 224)
}

# === Hyperparameters ===
LR_ACTOR = 1e-3
LR_CRITIC = 1e-3
GAMMA = 0.99
TAU = 0.005
BUFFER_SIZE = int(1e5)
BATCH_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_STEPS = 100000

# === Neural Networks ===
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, action_dim), nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=1))

# === Replay Buffer ===
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (torch.tensor(state, dtype=torch.float32).to(DEVICE),
                torch.tensor(action, dtype=torch.float32).to(DEVICE),
                torch.tensor(reward, dtype=torch.float32).unsqueeze(1).to(DEVICE),
                torch.tensor(next_state, dtype=torch.float32).to(DEVICE),
                torch.tensor(done, dtype=torch.float32).unsqueeze(1).to(DEVICE))

    def __len__(self):
        return len(self.buffer)

# === Pygame Setup ===
WIDTH, HEIGHT = 800, 800
SCALE = 200
WHITE, BLUE, RED, BLACK, GREY = (255,)*3, (0,120,255), (255,50,50), (0,0,0), (200,)*3

def world_to_screen(pos):
    return int(WIDTH / 2 + pos[0] * SCALE), int(HEIGHT / 2 - pos[1] * SCALE)

def draw_vehicle(screen, pos, theta):
    x, y = world_to_screen(pos)
    l, w = 0.2 * SCALE, 0.1 * SCALE

    front = (
        int(x + l * np.cos(theta)),
        int(y - l * np.sin(theta))
    )
    back_left = (
        int(x - l * 0.7 * np.cos(theta) + w * np.sin(theta)),
        int(y + l * 0.7 * np.sin(theta) + w * np.cos(theta))
    )
    back_right = (
        int(x - l * 0.7 * np.cos(theta) - w * np.sin(theta)),
        int(y + l * 0.7 * np.sin(theta) - w * np.cos(theta))
    )

    pygame.draw.polygon(screen, KIT_COLORS['green'], [front, back_left, back_right])

def draw_text(screen, font, text, x, y, color):
    screen.blit(font.render(text, True, color), (x, y))

# === Training and Visualization ===
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("RL Car Navigation Viewer")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)

    env = RCCarNavigationEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    actor = Actor(state_dim, action_dim).to(DEVICE)
    actor_target = Actor(state_dim, action_dim).to(DEVICE)
    actor_target.load_state_dict(actor.state_dict())

    critic = Critic(state_dim, action_dim).to(DEVICE)
    critic_target = Critic(state_dim, action_dim).to(DEVICE)
    critic_target.load_state_dict(critic.state_dict())

    actor_optim = optim.Adam(actor.parameters(), lr=LR_ACTOR)
    critic_optim = optim.Adam(critic.parameters(), lr=LR_CRITIC)
    buffer = ReplayBuffer(BUFFER_SIZE)

    # === Laden gespeicherter Modelle ===
    try:
        actor.load_state_dict(torch.load("actor_100000.pth"))
        print("Model loaded.")
    except FileNotFoundError:
        print("No saved model found. Starting from scratch.")

    obs, _ = env.reset()
    episode_reward, path, episode_len = 0.0, [], 0
    current_reward = 0.0
    goals_in_a_row = 0
    total_goals = 0

    for step in range(1, MAX_STEPS + 1):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            action = actor(obs_tensor).cpu().numpy()[0]
        action += np.random.normal(0, 0.1, size=action_dim)
        action = np.clip(action, -1.0, 1.0)

        next_obs, reward, terminated, truncated, info = env.step(action)
        current_reward = reward
        done = terminated or truncated

        buffer.push((obs, action, reward, next_obs, float(done)))
        obs = next_obs
        path.append(env.position.copy())
        episode_reward += reward
        episode_len += 1

        if info.get("goal_reached", False):
            print(f"[Goal reached] Reward: {episode_reward:.2f}, Steps: {episode_len}")
            episode_reward = 0.0
            episode_len = 0
            path = []
            goals_in_a_row += 1
            total_goals += 1

        if done or episode_len >= 1500:
            print(f"[Reset] Reward: {episode_reward:.2f}, Steps: {episode_len}")
            obs, _ = env.reset(set_new_goal=terminated)
            episode_reward, path, episode_len = 0.0, [], 0
            goals_in_a_row = 0  # reset only the consecutive success counter

        if len(buffer) > BATCH_SIZE:
            s, a, r, s2, d = buffer.sample(BATCH_SIZE)
            with torch.no_grad():
                next_action = actor_target(s2)
                q_target = r + GAMMA * (1 - d) * critic_target(s2, next_action)
            q_val = critic(s, a)
            critic_loss = nn.MSELoss()(q_val, q_target)
            critic_optim.zero_grad()
            critic_loss.backward()
            critic_optim.step()

            actor_loss = -critic(s, actor(s)).mean()
            actor_optim.zero_grad()
            actor_loss.backward()
            actor_optim.step()

            for tp, p in zip(actor_target.parameters(), actor.parameters()):
                tp.data.copy_(TAU * p.data + (1 - TAU) * tp.data)
            for tp, p in zip(critic_target.parameters(), critic.parameters()):
                tp.data.copy_(TAU * p.data + (1 - TAU) * tp.data)

        if step % 10000 == 0:
            torch.save(actor.state_dict(), f"actor_{step}.pth")
            print(f"Model saved at step {step}")

        # === Visualization ===
        screen.fill(KIT_COLORS['white'])

        for i in range(0, WIDTH, SCALE):
            pygame.draw.line(screen, KIT_COLORS['blue'], (i, 0), (i, HEIGHT), 1)
        for j in range(0, HEIGHT, SCALE):
            pygame.draw.line(screen, KIT_COLORS['blue'], (0, j), (WIDTH, j), 1)

        pygame.draw.circle(screen, KIT_COLORS['red'], world_to_screen(env.goal), 8)

        if len(path) > 1:
            for i in range(1, len(path)):
                pygame.draw.line(screen, KIT_COLORS['black'], world_to_screen(path[i-1]), world_to_screen(path[i]), 2)

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
