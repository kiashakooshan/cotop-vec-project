import os
import sys
import traci

# بررسی وجود مسیر SUMO در متغیرهای سیستم
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Error: Please declare environment variable 'SUMO_HOME'")

class VECEnv:
    def __init__(self, sumocfg_path, rsus):
        self.sumocfg = sumocfg_path
        self.rsus = rsus  # لیستی از دیکشنری‌های مشخصات RSU
        
        # ساخت صف خالی برای هر RSU جهت مدیریت وظایف در آینده
        self.queues = {r["id"]: [] for r in rsus}
        
    def reset(self):
        # راه‌اندازی شبیه‌ساز در پس‌زمینه با رابط TraCI
        traci.start(["sumo", "-c", self.sumocfg])
        print("✅ SUMO Started Successfully via TraCI!")
        return self._get_state()
        
    def step(self):
        # جلو بردن شبیه‌ساز به اندازه یک گام زمانی
        traci.simulationStep()
        
        # استخراج آیدی تمام ماشین‌های فعال در این لحظه
        vehicle_ids = traci.vehicle.getIDList()
        vehicles_info = []
        
        for v_id in vehicle_ids:
            # دریافت موقعیت مکانی و سرعت هر ماشین
            pos = traci.vehicle.getPosition(v_id)
            speed = traci.vehicle.getSpeed(v_id)
            vehicles_info.append({"id": v_id, "position": pos, "speed": speed})
            
        return vehicles_info
        
    def _get_state(self):
        # فعلاً یک حالت خالی برمی‌گردانیم تا در فازهای بعدی ساختار RL تکمیل شود
        return {}

    def close(self):
        # بستن امن ارتباط با شبیه‌ساز
        traci.close()
        print("🛑 SUMO Closed.")

# ---------------- بلوک تست کد ----------------
if __name__ == "__main__":
    # تعریف دو RSU فرضی برای تست اولیه
    test_rsus = [
        {"id": "RSU1", "x": 100, "y": 100, "range": 400},
        {"id": "RSU2", "x": 300, "y": 200, "range": 400}
    ]
    
    # آدرس فایل کانفیگ نسبت به پوشه src
    config_file = "../sumo/osm.sumocfg" 
    
    # ساخت محیط
    env = VECEnv(config_file, test_rsus)
    env.reset()
    
    # شبیه‌ساز را برای 15 ثانیه (قدم) اجرا می‌کنیم تا حرکت ماشین‌ها را ببینیم
    for i in range(15):
        print(f"\n--- Time Step {i} ---")
        info = env.step()
        print(f"Active Vehicles: {len(info)}")
        
        if len(info) > 0:
            # چاپ اطلاعات اولین ماشین برای نمونه
            print(f"Sample Data -> ID: {info[0]['id']}, Pos: {info[0]['position']}, Speed: {info[0]['speed']:.2f} m/s")
            
    env.close()