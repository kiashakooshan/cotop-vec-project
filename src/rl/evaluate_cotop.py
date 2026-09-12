import sys
import os
import csv
import torch
import random
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env.vec_env import VECEnv
from rl.a3c_agent import ActorCritic

MAX_EPISODES = 50
MAX_STEPS_PER_EPISODE = 300

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

def log_mobility_predictions(env):
    mob_path = "../results/mobility_predictions.csv"
    with open(mob_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["veh_id", "pred_x", "pred_y", "actual_x", "actual_y"])
        for veh_id, pairs in env.mobility_predictions.items():
            for pred, actual in pairs:
                writer.writerow([veh_id, pred[0], pred[1], actual[0], actual[1]])

def evaluate_trained_model(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    print(f"Evaluating trained CoTOP (seed={seed}, {MAX_EPISODES} episodes) -- ALL 40 vehicles controlled")
    env = VECEnv("../sumo/osm.sumocfg", "../sumo/rsus.json")
    agent = ActorCritic(11, 6)

    model_path = f"cotop_model_seed{seed}.pth"
    if os.path.exists(model_path):
        agent.load_state_dict(torch.load(model_path))
    else:
        print(f"Warning: '{model_path}' not found! Evaluating with random weights.")
    agent.eval()

    log_eval = f"cotop_log_seed{seed}"
    log_metrics = f"cotop_metrics_seed{seed}"
    for name in [log_eval, log_metrics]:
        if os.path.exists(f"../results/{name}.csv"):
            os.remove(f"../results/{name}.csv")

    for episode in range(MAX_EPISODES):
        states = env.reset(render=False)
        episode_reward = 0

        for step in range(MAX_STEPS_PER_EPISODE):
            actions = []
            for s in states:
                state_t = torch.FloatTensor(s)
                with torch.no_grad():
                    action_probs, _ = agent.forward(state_t)
                    action = torch.argmax(action_probs).item()
                actions.append(action)

            next_states, reward, done, _ = env.step(actions)
            episode_reward += reward
            states = next_states
            if done:
                break

        print(f"CoTOP (Eval) Episode {episode+1}/{MAX_EPISODES} | Reward: {episode_reward:.2f}")
        log_episode(log_eval, episode + 1, episode_reward)

        total_makespan = sum(env.episode_makespans) if env.episode_makespans else 0
        max_makespan = max(env.episode_makespans) if env.episode_makespans else 0
        log_advanced_metrics(log_metrics, env.episode_energy, env.episode_controlled_energy,
                              total_makespan, max_makespan)

        if episode == MAX_EPISODES - 1:
            log_mobility_predictions(env)

        env.close()

if __name__ == "__main__":
    seed_arg = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    evaluate_trained_model(seed=seed_arg)