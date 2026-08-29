import sys
import os
import csv
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env.vec_env import VECEnv
from rl.a3c_agent import ActorCritic

MAX_EPISODES = 10 # برای تست سریع‌تر روی ۱۰ اپیزود تنظیم شده است
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

def evaluate_trained_model():
    print(f"🚀 Evaluating FULLY TRAINED CoTOP ({MAX_EPISODES} Episodes)...")
    env = VECEnv("../sumo/osm.sumocfg", "../sumo/rsus.json")
    
    agent = ActorCritic(10, 6)
    if os.path.exists("cotop_trained_model.pth"):
        agent.load_state_dict(torch.load("cotop_trained_model.pth"))
    agent.eval()
    
    # فایل لاگ قبلی پاک می‌شود تا با دیتای دوران آموزش قاطی نشود
    if os.path.exists("../results/cotop_log.csv"):
        os.remove("../results/cotop_log.csv")
        
    for episode in range(MAX_EPISODES):
        state = env.reset(render=False)
        episode_reward = 0
        
        for step in range(MAX_STEPS_PER_EPISODE):
            state_tensor = torch.FloatTensor(state)
            with torch.no_grad():
                action_probs, _ = agent.forward(state_tensor)
                # انتخاب قطعی بهترین اکشن (بدون Epsilon-Greedy یا تصادف)
                action = torch.argmax(action_probs).item() 
                
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            state = next_state
            if done: break
            
        print(f"✅ CoTOP (Eval) - Episode {episode+1} | Reward: {episode_reward:.2f}")
        log_episode("cotop", episode + 1, episode_reward)
        
        env.close()

if __name__ == "__main__":
    evaluate_trained_model()