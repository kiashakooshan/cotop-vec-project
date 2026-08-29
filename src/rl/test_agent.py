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
    
    state_dim = 10 # ۴ ویژگی ماشین + صف ۶ دستگاه RSU
    action_dim = 6 # ۶ دستگاه RSU
    
    # ۱. ساختن اسکلت شبکه
    agent = ActorCritic(state_dim, action_dim)
    
    # ۲. لود کردن وزن‌های آموزش‌دیده
    agent.load_state_dict(torch.load("cotop_trained_model.pth"))
    agent.eval() # قرار دادن شبکه در حالت استنتاج (بسیار مهم: خاموش کردن Dropout و غیره)
    
    # ۳. باز کردن گرافیک برای ارائه
    state = env.reset(render=True)
    total_reward = 0
    
    print("🚗 Simulation is ready! Press 'Play' in SUMO to start...")
    
    MAX_STEPS = 300 
    
    for step in range(MAX_STEPS):
        state_tensor = torch.FloatTensor(state)
        
        # بدون نیاز به محاسبه گرادیان (سرعت اجرای بسیار بالا)
        with torch.no_grad(): 
            action, _ = agent.get_action(state_tensor)
            
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        state = next_state
        
        if done:
            break
            
    print(f"🏁 Demonstration Finished! Total Reward achieved: {total_reward:.2f}")
    env.close()

if __name__ == "__main__":
    demonstrate_model()