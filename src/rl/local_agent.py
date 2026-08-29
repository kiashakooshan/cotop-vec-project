import sys
import os
import csv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env.vec_env import VECEnv

MAX_EPISODES = 50
MAX_STEPS_PER_EPISODE = 300

def log_episode(method_name, episode, total_reward):
    """ذخیره نتایج در فایل CSV برای رسم نمودار (اصلاح ۶)"""
    os.makedirs("../results", exist_ok=True)
    file_path = f"../results/{method_name}_log.csv"
    write_header = not os.path.exists(file_path)
    
    with open(file_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["episode", "reward"])
        writer.writerow([episode, total_reward])

def evaluate_local():
    config_file = "../sumo/osm.sumocfg" 
    rsus_file = "../sumo/rsus.json"
    
    # در روش محلی، ویژگی همکاری (Collaboration) خاموش است
    env = VECEnv(config_file, rsus_file, use_collaboration=False)
    
    print(f"🚀 Starting Local Baseline Evaluation ({MAX_EPISODES} Episodes)...")
    
    for episode in range(MAX_EPISODES):
        state = env.reset(render=False)
        episode_reward = 0
        
        for step in range(MAX_STEPS_PER_EPISODE):
            # اکشن همیشه 0 است (پردازش در همان RSU اول)
            action = 0 
            
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            state = next_state
            
            if done:
                break
                
        print(f"✅ Local Agent - Episode {episode + 1}/{MAX_EPISODES} | Total Reward: {episode_reward:.2f}")
        log_episode("local", episode + 1, episode_reward)
        env.close()
        
    print("🏁 Local Baseline Finished! Results saved to 'results/local_log.csv'")

if __name__ == "__main__":
    evaluate_local()