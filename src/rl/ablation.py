import sys
import os
import csv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env.vec_env import VECEnv

MAX_STEPS = 100

def log_ablation_result(test_name, total_reward):
    """
    اصلاح 6: ذخیره نتایج واقعی در فایل CSV
    """
    os.makedirs("../results", exist_ok=True)
    file_path = "../results/ablation_log.csv"
    write_header = not os.path.exists(file_path)
    
    with open(file_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["Test_Condition", "Total_Reward"])
        writer.writerow([test_name, total_reward])

def run_single_test(test_name, use_md, use_tp, use_co):
    print(f"\n🔬 Running: {test_name}")
    config_file = "../sumo/osm.sumocfg" 
    rsus_file = "../sumo/rsus.json"
    
    # ساخت محیط با کلیدهای تنظیم شده
    env = VECEnv(config_file, rsus_file, 
                 use_mobility_detector=use_md, 
                 use_priority=use_tp, 
                 use_collaboration=use_co)
    
    env.reset(render=False)
    total_reward = 0
    
    for step in range(MAX_STEPS):
        # برای تست Ablation یک اکشن ساده اعمال می‌کنیم تا صرفاً تاثیر محیط را بسنجیم
        action = 0 
        _, reward, done, _ = env.step(action)
        total_reward += reward
        if done: break
        
    env.close()
    print(f"📊 Result for {test_name}: {total_reward:.2f}")
    log_ablation_result(test_name, total_reward)

def run_ablation_studies():
    print("🚀 Starting Real Ablation Studies...")
    
    # فایل قبلی را پاک می‌کنیم تا لاگ جدید ثبت شود
    if os.path.exists("../results/ablation_log.csv"):
        os.remove("../results/ablation_log.csv")
        
    # تست 1: سیستم کامل (CoTOP)
    run_single_test("Complete_CoTOP", use_md=True, use_tp=True, use_co=True)
    
    # تست 2: حذف ماژول تحرک (w/o MD)
    run_single_test("w/o_MD", use_md=False, use_tp=True, use_co=True)
    
    # تست 3: حذف اولویت‌بندی (w/o TP)
    run_single_test("w/o_TP", use_md=True, use_tp=False, use_co=True)
    
    # تست 4: حذف همکاری (w/o CO)
    run_single_test("w/o_CO", use_md=True, use_tp=True, use_co=False)
    
    print("\n✅ Ablation tests completed! Results saved to 'results/ablation_log.csv'")

if __name__ == "__main__":
    run_ablation_studies()