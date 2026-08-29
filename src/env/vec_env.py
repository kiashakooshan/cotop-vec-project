import os
import sys
import json
import math
import traci
import torch
import numpy as np

# Setup paths to access other modules
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
    def __init__(self, sumocfg_path, rsus_json_path="../sumo/rsus.json"):
        self.sumocfg = sumocfg_path
        
        # 1. Dynamically load RSUs from your JSON file (Works for 6 RSUs seamlessly)
        with open(rsus_json_path, 'r') as f:
            self.rsus = json.load(f)
            
        # Initialize an empty queue list for each RSU
        self.queues = {r["id"]: [] for r in self.rsus}
        self.current_time = 0
        
        # 2. Initialize the Mobility Detector (GAT + GRU)
        self.mobility_model = MobilityDetector(in_dim=2, hidden=32, gru_hidden=64)
        self.mobility_model.eval() 
        
    def reset(self, render=False):
        if render:
            traci.start(["sumo-gui", "-c", self.sumocfg])
            
            # --- کدهای جدید برای رسم RSUها روی نقشه ---
            for rsu in self.rsus:
                try:
                    # رسم مرکز RSU (نقطه قرمز)
                    traci.poi.add(poiID=rsu["id"], x=rsu["x"], y=rsu["y"], color=(255, 0, 0, 255), poiType="RSU", layer=100)
                    
                    # رسم دایره شعاع آنتن‌دهی (چندضلعی آبی شفاف)
                    shape = []
                    for i in range(36):
                        angle = i * 10 * math.pi / 180
                        cx = rsu["x"] + rsu["range"] * math.cos(angle)
                        cy = rsu["y"] + rsu["range"] * math.sin(angle)
                        shape.append((cx, cy))  
                    traci.polygon.add(polygonID=rsu["id"]+"_cov", shape=shape, color=(0, 150, 255, 255), fill=False, lineWidth=2, layer=50)
                except:
                    pass # در صورتی که دایره از قبل رسم شده باشد، خطا ندهد
            # ----------------------------------------
            
        else:
            traci.start(["sumo", "-c", self.sumocfg, "--start", "--quit-on-end"])
            
        self.current_time = 0
        self.queues = {r["id"]: [] for r in self.rsus}
        return self._get_state()
        
    def _build_graph(self, vehicle_ids, max_distance=100.0):
        """
        Builds the Adjacency Matrix (edge_index) for the Graph Neural Network.
        Connects vehicles that are within 100 meters of each other.
        """
        edges = []
        positions = {v_id: traci.vehicle.getPosition(v_id) for v_id in vehicle_ids}
            
        for i, v1 in enumerate(vehicle_ids):
            for j, v2 in enumerate(vehicle_ids):
                if i != j:
                    if math.dist(positions[v1], positions[v2]) < max_distance:
                        edges.append([i, j])
                        
        if not edges:
            return torch.empty((2, 0), dtype=torch.long)
        return torch.tensor(edges, dtype=torch.long).t().contiguous()

    def step(self, action):
        """
        Executes one step in the simulator, applies the RL action, and calculates the reward.
        """
        traci.simulationStep()
        self.current_time += 1
        
        vehicle_ids = traci.vehicle.getIDList()
        done = traci.simulation.getMinExpectedNumber() <= 0
        reward = 0
        
        if len(vehicle_ids) > 0:
            v_id = vehicle_ids[0]
            
            # ۱. استخراج موقعیت تمام ماشین‌های فعال برای ساخت گراف
            all_positions = [traci.vehicle.getPosition(v) for v in vehicle_ids]
            edge_index = self._build_graph(vehicle_ids)
            
            # تبدیل به تنسور با ابعاد: [تعداد ماشین‌ها, طول توالی=1, ویژگی‌ها=2]
            x_seq = torch.tensor(all_positions, dtype=torch.float32).unsqueeze(1)
            
            # ۲. ارسال گراف کامل به مدل تحرک GAT+GRU
            with torch.no_grad():
                predicted_future = self.mobility_model(x_seq, [edge_index], future_steps=1)
            
            predicted_t_stay = 20.0 # فرض زمان خروج برای ادامه محاسبات
            
            # ۳. بقیه منطق تولید وظیفه برای ماشین نماینده
            task = generate_random_task(v_id, self.current_time)
            task["T_stay"] = predicted_t_stay 
            
            selected_rsu_idx = action if action < len(self.rsus) else 0
            selected_rsu_id = self.rsus[selected_rsu_idx]["id"]
            
            self.queues[selected_rsu_id].append(task)
            self.queues[selected_rsu_id] = sort_tasks_by_priority(self.queues[selected_rsu_id])
            top_task = self.queues[selected_rsu_id].pop(0)
            
            delay = models.processing_delay(top_task["phi"], 2.0)
            energy = delay * 5.0 
            total_delay = delay + 0.1 
            
            if total_delay > top_task["d"]:
                reward = -100.0
            else:
                reward = -(total_delay + energy)
                
        next_state = self._get_state(vehicle_ids)
        return next_state, reward, done, []
        
    def _get_state(self, vehicle_ids=None):
        """
        Converts the environment into a Tensor array.
        Dynamically scales: [X, Y, Task_Size, Deadline, Queue_RSU1, Queue_RSU2, ... Queue_RSU6]
        """
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
            
        # Dynamically append the load of EVERY RSU in the JSON file
        for rsu in self.rsus:
            total_load = sum(t["phi"] for t in self.queues[rsu["id"]])
            state.append(total_load)
            
        return np.array(state, dtype=np.float32)

    def close(self):
        traci.close()