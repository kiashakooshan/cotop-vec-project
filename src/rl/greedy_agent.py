import sys
import os
import csv
import random
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env.vec_env import VECEnv

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

def evaluate_greedy(seed=0):
    random.seed(seed)
    np.random.seed(seed)

    env = VECEnv("../sumo/osm.sumocfg", "../sumo/rsus.json")
    log_eval = f"greedy_log_seed{seed}"
    log_metrics = f"greedy_metrics_seed{seed}"
    for name in [log_eval, log_metrics]:
        if os.path.exists(f"../results/{name}.csv"):
            os.remove(f"../results/{name}.csv")

    print(f"Starting Greedy Baseline Evaluation (seed={seed}, {MAX_EPISODES} episodes) -- ALL 40 vehicles controlled")

    for episode in range(MAX_EPISODES):
        states = env.reset(render=False)
        episode_reward = 0

        for step in range(MAX_STEPS_PER_EPISODE):
            # every vehicle independently picks its own least-loaded RSU
            actions = []
            for _ in states:
                best_action, min_load = 0, float('inf')
                for i, node in enumerate(env.rsu_nodes):
                    avg_free = sum(p.free_at for p in node.processors) / len(node.processors)
                    if avg_free < min_load:
                        min_load = avg_free
                        best_action = i
                actions.append(best_action)

            next_states, reward, done, _ = env.step(actions)
            episode_reward += reward
            states = next_states
            if done:
                break

        print(f"Greedy Episode {episode + 1}/{MAX_EPISODES} | Total Reward: {episode_reward:.2f}")
        log_episode(log_eval, episode + 1, episode_reward)

        total_makespan = sum(env.episode_makespans) if env.episode_makespans else 0
        max_makespan = max(env.episode_makespans) if env.episode_makespans else 0
        log_advanced_metrics(log_metrics, env.episode_energy, env.episode_controlled_energy,
                              total_makespan, max_makespan)
        env.close()

    print("Greedy Baseline Finished!")

if __name__ == "__main__":
    seed_arg = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    evaluate_greedy(seed=seed_arg)