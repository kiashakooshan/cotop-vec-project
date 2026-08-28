import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env.vec_env import VECEnv

MAX_EPISODES = 5
MAX_STEPS_PER_EPISODE = 20

def evaluate_local():
    test_rsus = [
        {"id": "RSU1", "x": 100, "y": 100, "range": 400},
        {"id": "RSU2", "x": 300, "y": 200, "range": 400}
    ]
    config_file = "../sumo/osm.sumocfg" 
    env = VECEnv(config_file, test_rsus)
    
    print("🚀 Starting Local Baseline Evaluation...")
    
    total_rewards = []
    
    for episode in range(MAX_EPISODES):
        state = env.reset(render=False)
        episode_reward = 0
        
        for step in range(MAX_STEPS_PER_EPISODE):
            # در روش Local، ماشین‌ها وظایف را به جای دیگری نمی‌فرستند
            # بنابراین اکشن همیشه 0 (پردازش در همان RSU) است
            action = 0 
            
            next_state, reward, done, step_data = env.step(action)
            episode_reward += reward
            state = next_state
            
            if done:
                break
                
        total_rewards.append(episode_reward)
        print(f"✅ Local Agent - Episode {episode + 1} | Total Reward: {episode_reward:.2f}")
        
        env.close()
        
    avg_reward = sum(total_rewards) / len(total_rewards)
    print(f"🏁 Local Baseline Finished! Average Reward: {avg_reward:.2f}")

if __name__ == "__main__":
    evaluate_local()