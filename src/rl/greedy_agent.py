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

# --- اضافه شدن ثبت گزارش‌های پیشرفته (Energy و Makespan) ---
def log_advanced_metrics(method_name, energy, avg_makespan, max_makespan):
    file_path = f"../results/{method_name}_metrics.csv"
    write_header = not os.path.exists(file_path)
    with open(file_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["energy", "avg_makespan", "max_makespan"])
        writer.writerow([energy, avg_makespan, max_makespan])

def evaluate_greedy():
    env = VECEnv("../sumo/osm.sumocfg", "../sumo/rsus.json")
    
    # پاک کردن لاگ‌های قبلی
    if os.path.exists("../results/greedy_log.csv"):
        os.remove("../results/greedy_log.csv")
    if os.path.exists("../results/greedy_metrics.csv"):
        os.remove("../results/greedy_metrics.csv")
        
    print(f"🚀 Starting Greedy Baseline Evaluation ({MAX_EPISODES} Episodes)...")
    
    for episode in range(MAX_EPISODES):
        state = env.reset(render=False)
        episode_reward = 0
        
        for step in range(MAX_STEPS_PER_EPISODE):
            # --- اصلاح منطق حریصانه برای همگام‌سازی با پردازنده‌های A7/A12 ---
            best_action = 0
            min_load = float('inf')
            
            # جستجو برای پیدا کردن RSU با کمترین میانگین زمان درگیری پردازنده‌ها
            for i, node in enumerate(env.rsu_nodes):
                avg_free = sum(p.free_at for p in node.processors) / len(node.processors)
                if avg_free < min_load:
                    min_load = avg_free
                    best_action = i
                    
            next_state, reward, done, _ = env.step(best_action)
            episode_reward += reward
            state = next_state
            
            if done: break
                
        print(f"✅ Greedy Agent - Episode {episode + 1}/{MAX_EPISODES} | Total Reward: {episode_reward:.2f}")
        log_episode("greedy", episode + 1, episode_reward)
        
        # استخراج و ثبت متریک‌های پیشرفته برای این اپیزود
        total_makespan = sum(env.episode_makespans) if env.episode_makespans else 0
        max_makespan = max(env.episode_makespans) if env.episode_makespans else 0
        log_advanced_metrics("greedy", env.episode_energy, total_makespan, max_makespan)
        
        env.close()
        
    print("🏁 Greedy Baseline Finished!")

if __name__ == "__main__":
    evaluate_greedy()