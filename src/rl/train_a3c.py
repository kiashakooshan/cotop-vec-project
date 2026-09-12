import sys
import os
import torch
import torch.multiprocessing as mp
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env.vec_env import VECEnv
from rl.a3c_agent import ActorCritic
class SharedAdam(optim.Adam):
    """Adam optimizer whose internal state lives in shared memory, so multiple
    worker processes can safely update the same global network (Hogwild!-style)."""
    def __init__(self, params, lr=1e-4):
        super().__init__(params, lr=lr)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = torch.zeros(1)
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()
                state['step'].share_memory_()
def worker(rank, global_agent, optimizer, num_episodes):
    random.seed(rank)
    np.random.seed(rank)
    torch.manual_seed(rank)
    env = VECEnv("../sumo/osm.sumocfg", "../sumo/rsus.json", sumo_port=8813 + rank)
    local_agent = ActorCritic(11, 6)
    for episode in range(num_episodes):
        local_agent.load_state_dict(global_agent.state_dict())   # sync with the global brain
        states = env.reset(render=False)
        total_reward = 0
        epsilon = max(0.01, 0.5 - (episode / (num_episodes * 0.8)))
        for step in range(300):
            actions, log_probs, values = [], [], []
            for s in states:
                state_t = torch.FloatTensor(s)
                action_probs, state_value = local_agent(state_t)
                if random.random() < epsilon:
                    action = random.randint(0, 5)
                else:
                    action = torch.argmax(action_probs).item()
                actions.append(action)
                log_probs.append(torch.log(action_probs[action] + 1e-10))
                values.append(state_value)
            next_states, reward, done, _ = env.step(actions)
            total_reward += reward
            scaled_reward = reward / 1000.0
            target = torch.tensor(scaled_reward, dtype=torch.float32)
            loss = 0.0
            for log_prob, value in zip(log_probs, values):
                advantage = scaled_reward - value.item()
                loss = loss + (-log_prob * advantage) + F.mse_loss(value.squeeze(), target)
            loss = loss / len(log_probs)
            optimizer.zero_grad()
            loss.backward()
            # push this worker's local gradients into the SHARED global network
            for local_param, global_param in zip(local_agent.parameters(), global_agent.parameters()):
                global_param._grad = local_param.grad
            optimizer.step()
            states = next_states
            if done:
                break
        print(f"[Worker {rank}] Ep {episode+1}/{num_episodes} | Reward: {total_reward:.2f}")
        env.close()
def train_a3c(num_workers=4, episodes_per_worker=50):
    global_agent = ActorCritic(11, 6)
    global_agent.share_memory()   # <-- makes the network's weights visible/writable across processes
    optimizer = SharedAdam(global_agent.parameters(), lr=0.0002)
    processes = []
    for rank in range(num_workers):
        p = mp.Process(target=worker, args=(rank, global_agent, optimizer, episodes_per_worker))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    torch.save(global_agent.state_dict(), "cotop_model_a3c.pth")
    print("Saved true multi-worker A3C model as 'cotop_model_a3c.pth'")
if __name__ == "__main__":
    mp.set_start_method("spawn")
    train_a3c(num_workers=4, episodes_per_worker=50)
