import os
import sys
import json
import math
import traci
import torch
import numpy as np

# تنظیم مسیر برای دسترسی به ماژول‌های پروژه
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env.task_generator import generate_random_task
from models import models
from models.priority import sort_tasks_by_priority
from mobility.gat_gru import MobilityDetector

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Error: Please declare environment variable 'SUMO_HOME'")

class VECEnv:
    def __init__(self, sumocfg_path, rsus_json_path="../../sumo/rsus.json"):
        self.sumocfg = sumocfg_path
        
        # 1. خواندن هوشمند مختصات RSUها از فایل JSON
        with open(rsus_json_path, 'r') as f:
            self.rsus = json.load(f)
            
        # مقدار صف هر RSU در ابتدا خالی است (برای اولویت‌بندی به لیست نیاز داریم)
        self.queues = {r["id"]: [] for r in self.rsus}
        self.current_time = 0
        
        # 2. راه‌اندازی مدل تشخیص تحرک (GAT + GRU)
        # ورودی را روی 2 تنظیم می‌کنیم (فقط مختصات X و Y)
        self.mobility_model = MobilityDetector(in_dim=2, hidden=32, gru_hidden=64)
        self.mobility_model.eval() # مدل در حالت استنتاج (بدون آموزش موقت)
        
    def reset(self, render=False):
        if render:
            traci.start(["sumo-gui", "-c", self.sumocfg])
        else:
            traci.start(["sumo", "-c", self.sumocfg, "--start", "--quit-on-end"])
            
        self.current_time = 0
        self.queues = {r["id"]: [] for r in self.rsus}
        
        return self._get_state()
        
    def _build_graph(self, vehicle_ids, max_distance=50.0):
        """
        ساخت ماتریس مجاورت (Edge Index) برای شبکه GAT 
        ارتباط بین ماشین‌هایی که کمتر از 50 متر با هم فاصله دارند
        """
        edges = []
        positions = {}
        
        for v_id in vehicle_ids:
            positions[v_id] = traci.vehicle.getPosition(v_id)
            
        for i, v1 in enumerate(vehicle_ids):
            for j, v2 in enumerate(vehicle_ids):
                if i != j:
                    dist = math.dist(positions[v1], positions[v2])
                    if dist < max_distance:
                        edges.append([i, j])
                        
        if not edges:
            # گراف خالی (بدون همسایه)
            return torch.empty((2, 0), dtype=torch.long)
            
        return torch.tensor(edges, dtype=torch.long).t().contiguous()

    def step(self, action):
        traci.simulationStep()
        self.current_time += 1
        
        vehicle_ids = traci.vehicle.getIDList()
        done = traci.simulation.getMinExpectedNumber() <= 0
        step_data = []
        reward = 0
        
        if len(vehicle_ids) > 0:
            v_id = vehicle_ids[0]
            pos = traci.vehicle.getPosition(v_id)
            
            # 3. استفاده از گراف و GAT+GRU برای تخمین T_stay
            edge_index = self._build_graph(vehicle_ids)
            
            # یک ورودی تنسور فرضی از موقعیت فعلی (در واقعیت باید توالی گذشته باشد)
            x_seq = torch.tensor([[[pos[0], pos[1]]]], dtype=torch.float32)
            edge_index_seq = [edge_index] 
            
            with torch.no_grad():
                # پیش‌بینی مسیر آینده توسط مدل GAT+GRU
                predicted_future = self.mobility_model(x_seq, edge_index_seq, future_steps=1)
            
            # فرضا بر اساس پیش‌بینی، ماشین 15 ثانیه در پوشش می‌ماند
            predicted_t_stay = 15.0 
            
            # 4. تولید وظیفه و اضافه کردن به صف RSU
            task = generate_random_task(v_id, self.current_time)
            task["T_stay"] = predicted_t_stay 
            
            selected_rsu_id = self.rsus[0]["id"]
            self.queues[selected_rsu_id].append(task)
            
            # 5. اعمال اولویت‌بندی وظایف در صف
            self.queues[selected_rsu_id] = sort_tasks_by_priority(self.queues[selected_rsu_id])
            
            # پردازش مهم‌ترین وظیفه (ایندکس 0 بعد از سورت شدن)
            top_task = self.queues[selected_rsu_id].pop(0)
            
            F_rsu = 2.0 
            delay = models.processing_delay(top_task["phi"], F_rsu)
            energy = delay * 5.0 
            total_delay = delay + 0.1 
            
            if total_delay > top_task["d"]:
                reward = -100.0
            else:
                reward = -(total_delay + energy)
                
        next_state = self._get_state(vehicle_ids)
        
        return next_state, reward, done, step_data
        
    def _get_state(self, vehicle_ids=None):
        if vehicle_ids is None:
            try:
                vehicle_ids = traci.vehicle.getIDList()
            except:
                vehicle_ids = []
                
        state = [0.0, 0.0, 0.0, 0.0] 
        
        if len(vehicle_ids) > 0:
            v_id = vehicle_ids[0]
            pos = traci.vehicle.getPosition(v_id)
            task = generate_random_task(v_id, self.current_time)
            state = [pos[0], pos[1], task["rho"], task["d"]]
            
        for rsu in self.rsus:
            # محاسبه بار پردازشی کل صف
            total_load = sum(t["phi"] for t in self.queues[rsu["id"]])
            state.append(total_load)
            
        return np.array(state, dtype=np.float32)

    def close(self):
        traci.close()