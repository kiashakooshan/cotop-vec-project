import sys
import os
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env.vec_env import VECEnv
from rl.a3c_agent import ActorCritic 

def demonstrate_model():
    print("🎓 Loading Pre-trained CoTOP Model for Presentation...")
    
    config_file = "../sumo/osm.sumocfg" 
    rsus_file = "../sumo/rsus.json"
    env = VECEnv(config_file, rsus_file)
    
    # اصلاح ۱: فضای حالت به ۸ تغییر یافت (۲ ویژگی مکانی + ۶ وضعیت لود سرورها)
    state_dim = 8 
    action_dim = 6 
    
    agent = ActorCritic(state_dim, action_dim)
    
    # اصلاح ۲: لود کردن مغز نهایی و اصلی
    model_path = "cotop_model_final.pth"
    if os.path.exists(model_path):
        agent.load_state_dict(torch.load(model_path))
        print(f"✅ Model '{model_path}' loaded successfully!")
    else:
        print(f"⚠️ Warning: '{model_path}' not found! Please train the model first.")
        
    agent.eval() 
    
    state = env.reset(render=True)
    total_reward = 0
    
    print("🚗 Simulation is ready! Press 'Play' in SUMO to start...")
    
    MAX_STEPS = 300 
    
    for step in range(MAX_STEPS):
        state_tensor = torch.FloatTensor(state)
        
        with torch.no_grad(): 
            # اصلاح ۳: انتخاب قطعی و هوشمندانه بهترین اکشن برای اجرای بی‌نقص در ارائه
            action_probs, _ = agent.forward(state_tensor)
            action = torch.argmax(action_probs).item()
            
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        state = next_state
        
        if done:
            break
            
    print(f"🏁 Demonstration Finished! Total Reward achieved: {total_reward:.2f}")
    env.close()

if __name__ == "__main__":
    demonstrate_model()