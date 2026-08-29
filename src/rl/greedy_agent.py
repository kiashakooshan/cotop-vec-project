import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env.vec_env import VECEnv

MAX_EPISODES = 50
MAX_STEPS_PER_EPISODE = 300

def evaluate_greedy():
    config_file = "../sumo/osm.sumocfg" 
    rsus_file = "../sumo/rsus.json"
    env = VECEnv(config_file, rsus_file)
    
    print(f"🚀 Starting Greedy Baseline Evaluation ({MAX_EPISODES} Episodes)...")
    total_rewards = []
    
    for episode in range(MAX_EPISODES):
        state = env.reset(render=False)
        episode_reward = 0
        
        for step in range(MAX_STEPS_PER_EPISODE):
            queues = env.queues
            
            # اصلاح ۳: مقایسه درست بر اساس طول صف (تعداد وظایف)
            best_rsu_id = min(queues, key=lambda k: len(queues[k]))
            
            # تبدیل شناسه RSU به اکشن عددی داینامیک (بین 0 تا 5)
            action = [r["id"] for r in env.rsus].index(best_rsu_id)
            
            next_state, reward, done, step_data = env.step(action)
            episode_reward += reward
            state = next_state
            
            if done:
                break
                
        total_rewards.append(episode_reward)
        print(f"✅ Greedy Agent - Episode {episode + 1}/{MAX_EPISODES} | Total Reward: {episode_reward:.2f}")
        env.close()
        
    avg_reward = sum(total_rewards) / len(total_rewards)
    print(f"🏁 Greedy Baseline Finished! Average Reward: {avg_reward:.2f}")

if __name__ == "__main__":
    evaluate_greedy()