import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
from collections import deque
import csv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env.vec_env import VECEnv

TRAIN_EPISODES = 50
EVAL_EPISODES = 50
MAX_STEPS_PER_EPISODE = 300
TAU = 0.005

def log_episode(method_name, episode, total_reward):
    os.makedirs("../results", exist_ok=True)
    file_path = f"../results/{method_name}.csv"
    write_header = not os.path.exists(file_path)
    with open(file_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["episode", "reward"])
        writer.writerow([episode, total_reward])

def log_advanced_metrics(method_name, energy, controlled_energy, total_makespan, max_makespan):
    file_path = f"../results/{method_name}.csv"
    write_header = not os.path.exists(file_path)
    with open(file_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["energy", "controlled_energy", "total_makespan", "max_makespan"])
        writer.writerow([energy, controlled_energy, total_makespan, max_makespan])

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    def forward(self, x):
        return self.fc(x)

def run_ddqn(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = VECEnv("../sumo/osm.sumocfg", "../sumo/rsus.json")
    state_dim, action_dim = 11, 6

    online_net = QNetwork(state_dim, action_dim)
    target_net = QNetwork(state_dim, action_dim)
    target_net.load_state_dict(online_net.state_dict())
    optimizer = optim.Adam(online_net.parameters(), lr=0.001)
    buffer = deque(maxlen=10000)

    log_train = f"ddqn_train_seed{seed}"
    log_eval = f"ddqn_log_seed{seed}"
    log_metrics = f"ddqn_metrics_seed{seed}"
    for name in [log_train, log_eval, log_metrics]:
        if os.path.exists(f"../results/{name}.csv"):
            os.remove(f"../results/{name}.csv")

    print(f"Phase 1: Training DDQN (seed={seed}, {TRAIN_EPISODES} episodes) -- ALL 40 vehicles controlled")
    for episode in range(TRAIN_EPISODES):
        states = env.reset(render=False)
        total_reward = 0
        epsilon = max(0.01, 0.5 - (episode / (TRAIN_EPISODES * 0.8)))

        for step in range(MAX_STEPS_PER_EPISODE):
            actions = []
            for s in states:
                state_t = torch.FloatTensor(s)
                if random.random() < epsilon:
                    action = random.randint(0, action_dim - 1)
                else:
                    action = online_net(state_t).argmax().item()
                actions.append(action)

            next_states, reward, done, _ = env.step(actions)
            scaled_reward = reward / 1000.0
            for s, a, ns in zip(states, actions, next_states):
                buffer.append((s, a, scaled_reward, ns, done))
            total_reward += reward
            states = next_states

            if len(buffer) > 64:
                batch = random.sample(buffer, 64)
                b_s, b_a, b_r, b_ns, b_d = zip(*batch)
                b_s = torch.FloatTensor(np.array(b_s))
                b_a = torch.LongTensor(b_a).unsqueeze(1)
                b_r = torch.FloatTensor(b_r)
                b_ns = torch.FloatTensor(np.array(b_ns))
                b_d = torch.FloatTensor(b_d)

                next_actions = online_net(b_ns).argmax(dim=1, keepdim=True)
                next_q = target_net(b_ns).gather(1, next_actions).squeeze(1)
                target_q = b_r + 0.95 * next_q * (1 - b_d)
                current_q = online_net(b_s).gather(1, b_a).squeeze(1)

                loss = F.mse_loss(current_q, target_q.detach())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                for target_param, local_param in zip(target_net.parameters(), online_net.parameters()):
                    target_param.data.copy_(TAU * local_param.data + (1.0 - TAU) * target_param.data)

            if done:
                break
        print(f"Train Ep {episode+1}/{TRAIN_EPISODES} | Reward: {total_reward:.2f} | Epsilon: {epsilon:.2f}")
        log_episode(log_train, episode + 1, total_reward)
        env.close()

    torch.save(online_net.state_dict(), f"ddqn_model_seed{seed}.pth")
    print(f"Saved ddqn_model_seed{seed}.pth")

    print(f"\\nPhase 2: Evaluating DDQN Deterministically (seed={seed}, {EVAL_EPISODES} episodes)...")
    for episode in range(EVAL_EPISODES):
        states = env.reset(render=False)
        total_reward = 0

        for step in range(MAX_STEPS_PER_EPISODE):
            actions = []
            for s in states:
                state_t = torch.FloatTensor(s)
                with torch.no_grad():
                    action = online_net(state_t).argmax().item()
                actions.append(action)

            next_states, reward, done, _ = env.step(actions)
            total_reward += reward
            states = next_states
            if done:
                break

        print(f"Eval Ep {episode+1}/{EVAL_EPISODES} | Reward: {total_reward:.2f}")
        log_episode(log_eval, episode + 1, total_reward)

        total_makespan = sum(env.episode_makespans) if env.episode_makespans else 0
        max_makespan = max(env.episode_makespans) if env.episode_makespans else 0
        log_advanced_metrics(log_metrics, env.episode_energy, env.episode_controlled_energy,
                              total_makespan, max_makespan)
        env.close()

if __name__ == "__main__":
    seed_arg = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    run_ddqn(seed=seed_arg)