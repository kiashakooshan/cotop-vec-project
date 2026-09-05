import sys
import os
import csv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env.vec_env import VECEnv

MAX_EPISODES = 50
MAX_STEPS_PER_EPISODE = 300

def log_episode(method_name, episode, total_reward):
    os.makedirs("../results", exist_ok=True)
    file_path = f"../results/{method_name}_log.csv"
    write_header = not os.path.exists(file_path)
    with open(file_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["episode", "reward"])
        writer.writerow([episode, total_reward])

def log_advanced_metrics(method_name, energy, avg_makespan, max_makespan):
    file_path = f"../results/{method_name}_metrics.csv"
    write_header = not os.path.exists(file_path)
    with open(file_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["energy", "avg_makespan", "max_makespan"])
        writer.writerow([energy, avg_makespan, max_makespan])

def evaluate_local():
    env = VECEnv("../sumo/osm.sumocfg", "../sumo/rsus.json", use_collaboration=False)
    
    if os.path.exists("../results/local_log.csv"):
        os.remove("../results/local_log.csv")
    if os.path.exists("../results/local_metrics.csv"):
        os.remove("../results/local_metrics.csv")
        
    print(f"🚀 Starting Local Baseline Evaluation ({MAX_EPISODES} Episodes)...")
    
    for episode in range(MAX_EPISODES):
        state = env.reset(render=False)
        episode_reward = 0
        
        for step in range(MAX_STEPS_PER_EPISODE):
            action = 0 
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            state = next_state
            if done: break
                
        print(f"✅ Local Agent - Episode {episode + 1}/{MAX_EPISODES} | Total Reward: {episode_reward:.2f}")
        log_episode("local", episode + 1, episode_reward)
        
        total_makespan = sum(env.episode_makespans) if env.episode_makespans else 0
        max_makespan = max(env.episode_makespans) if env.episode_makespans else 0
        log_advanced_metrics("local", env.episode_energy, total_makespan, max_makespan)
        
        env.close()
        
    print("🏁 Local Baseline Finished!")

if __name__ == "__main__":
    evaluate_local()