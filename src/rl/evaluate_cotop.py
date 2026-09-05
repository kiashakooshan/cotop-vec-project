import sys
import os
import csv
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env.vec_env import VECEnv
from rl.a3c_agent import ActorCritic

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

# --- توابع جدید برای ثبت گزارش‌های انرژی، Makespan و دقت GAT-GRU ---
def log_advanced_metrics(method_name, energy, avg_makespan, max_makespan):
    file_path = f"../results/{method_name}_metrics.csv"
    write_header = not os.path.exists(file_path)
    with open(file_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["energy", "avg_makespan", "max_makespan"])
        writer.writerow([energy, avg_makespan, max_makespan])

def log_mobility_predictions(env):
    mob_path = "../results/mobility_predictions.csv"
    with open(mob_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["veh_id", "pred_x", "pred_y", "actual_x", "actual_y"])
        for veh_id, pairs in env.mobility_predictions.items():
            for pred, actual in pairs:
                writer.writerow([veh_id, pred[0], pred[1], actual[0], actual[1]])
# -----------------------------------------------------------------

def evaluate_trained_model():
    print(f"🚀 Evaluating FULLY TRAINED CoTOP ({MAX_EPISODES} Episodes)...")
    env = VECEnv("../sumo/osm.sumocfg", "../sumo/rsus.json")
    
    agent = ActorCritic(8, 6)
    # اصلاح نام فایل به مدل نهایی
    if os.path.exists("cotop_model_final.pth"):
        agent.load_state_dict(torch.load("cotop_model_final.pth"))
    else:
        print("⚠️ Warning: 'cotop_model_final.pth' not found! Evaluating with random weights.")
    agent.eval()
    
    if os.path.exists("../results/cotop_log.csv"):
        os.remove("../results/cotop_log.csv")
    if os.path.exists("../results/cotop_metrics.csv"):
        os.remove("../results/cotop_metrics.csv")
        
    for episode in range(MAX_EPISODES):
        state = env.reset(render=False)
        episode_reward = 0
        
        for step in range(MAX_STEPS_PER_EPISODE):
            state_tensor = torch.FloatTensor(state)
            with torch.no_grad():
                action_probs, _ = agent.forward(state_tensor)
                action = torch.argmax(action_probs).item() 
                
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            state = next_state
            if done: break
            
        print(f"✅ CoTOP (Eval) - Episode {episode+1} | Reward: {episode_reward:.2f}")
        log_episode("cotop", episode + 1, episode_reward)
        
        # استخراج و ثبت متریک‌های پیشرفته برای این اپیزود
        total_makespan = sum(env.episode_makespans) if env.episode_makespans else 0
        max_makespan = max(env.episode_makespans) if env.episode_makespans else 0
        log_advanced_metrics("cotop", env.episode_energy, total_makespan, max_makespan)
        
        # در اپیزود آخر، خطای مسیر را برای گزارش GAT-GRU ذخیره کن
        if episode == MAX_EPISODES - 1:
            log_mobility_predictions(env)
            
        env.close()

if __name__ == "__main__":
    evaluate_trained_model()