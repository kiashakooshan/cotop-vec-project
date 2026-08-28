import os
import sys
import traci
import math
import numpy as np
from env.task_generator import generate_random_task
from models import models

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Error: Please declare environment variable 'SUMO_HOME'")

class VECEnv:
    def __init__(self, sumocfg_path, rsus):
        self.sumocfg = sumocfg_path
        self.rsus = rsus
        # مقدار صف هر RSU در ابتدا صفر است
        self.queues = {r["id"]: 0 for r in rsus}
        self.current_time = 0
        
    def reset(self, render=False):
        """
        با قرار دادن render=True محیط گرافیکی باز می‌شود و منتظر فرمان شما می‌ماند!
        """
        if render:
            # در حالت گرافیکی، دستورات استارت و خروج خودکار را برمی‌داریم
            traci.start(["sumo-gui", "-c", self.sumocfg])
        else:
            # در حالت آموزش پس‌زمینه، با نهایت سرعت و خودکار اجرا می‌شود
            traci.start(["sumo", "-c", self.sumocfg, "--start", "--quit-on-end"])
            
        self.current_time = 0
        self.queues = {r["id"]: 0 for r in self.rsus}
        
        return self._get_state()
        
    def step(self, action):
        """
        اعمال اکشن (انتخاب RSU) و محاسبه پاداش واقعی
        """
        traci.simulationStep()
        self.current_time += 1
        
        vehicle_ids = traci.vehicle.getIDList()
        done = traci.simulation.getMinExpectedNumber() <= 0
        
        step_data = []
        reward = 0
        
        # برای سادگی در این فاز، ماشین اول را به عنوان نماینده بررسی می‌کنیم
        if len(vehicle_ids) > 0:
            v_id = vehicle_ids[0]
            pos = traci.vehicle.getPosition(v_id)
            
            # 1. تولید وظیفه
            task = generate_random_task(v_id, self.current_time)
            
            # 2. محاسبه تاخیر و انرژی بر اساس مدل‌های ریاضی
            # فرض می‌کنیم RSU انتخاب شده ظرفیت پردازش 2 Gcycles/s دارد
            F_rsu = 2.0 
            delay = models.processing_delay(task["phi"], F_rsu)
            energy = delay * 5.0 # فرض توان مصرفی 5 وات
            
            total_delay = delay + 0.1 # به علاوه تاخیر شبکه
            
            # 3. محاسبه تابع پاداش (Reward)
            # اگر زمان انجام از مهلت (Deadline) بیشتر شود، جریمه سنگین (-100) می‌گیرد
            if total_delay > task["d"]:
                reward = -100.0
            else:
                # پاداش = منفی (تاخیر + انرژی)
                reward = -(total_delay + energy)
                
            # آپدیت صف RSU
            selected_rsu_id = self.rsus[0]["id"]
            self.queues[selected_rsu_id] += task["phi"]
            
        next_state = self._get_state(vehicle_ids)
        
        return next_state, reward, done, step_data
        
    def _get_state(self, vehicle_ids=None):
        """
        تبدیل داده‌های محیط به یک آرایه عددی (Tensor) برای شبکه عصبی
        آرایه شامل: [X_ماشین, Y_ماشین, حجم_وظیفه, مهلت_وظیفه, صف_RSU1, صف_RSU2]
        """
        if vehicle_ids is None:
            try:
                vehicle_ids = traci.vehicle.getIDList()
            except:
                vehicle_ids = []
                
        # مقادیر پیش‌فرض در صورت نبود ماشین
        state = [0.0, 0.0, 0.0, 0.0] 
        
        if len(vehicle_ids) > 0:
            v_id = vehicle_ids[0]
            pos = traci.vehicle.getPosition(v_id)
            task = generate_random_task(v_id, self.current_time)
            state = [pos[0], pos[1], task["rho"], task["d"]]
            
        # اضافه کردن وضعیت صف RSUها به State
        for rsu in self.rsus:
            state.append(self.queues[rsu["id"]])
            
        # تبدیل به آرایه Numpy و سپس Tensor
        return np.array(state, dtype=np.float32)

    def close(self):
        traci.close()