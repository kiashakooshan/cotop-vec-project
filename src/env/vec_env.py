import os
import sys
import json
import math
import traci
import torch
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env.task_generator import generate_random_task
from models import models
from models.models import sort_tasks_by_priority
from mobility.gat_gru import MobilityDetector

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Error: Please declare environment variable 'SUMO_HOME'")

class VECEnv:
    def __init__(self, sumocfg_path, rsus_json_path="../sumo/rsus.json", 
                 use_mobility_detector=True, use_priority=True, use_collaboration=True):
        self.sumocfg = sumocfg_path
        self.use_mobility_detector = use_mobility_detector
        self.use_priority = use_priority
        self.use_collaboration = use_collaboration
        
        with open(rsus_json_path, 'r') as f:
            self.rsus = json.load(f)
            
        self.queues = {r["id"]: [] for r in self.rsus}
        self.current_time = 0
        
        if self.use_mobility_detector:
            self.mobility_model = MobilityDetector(in_dim=2, hidden=32, gru_hidden=64)
            # --- تغییر جدید: لود کردن وزن‌های آموزش‌دیده ---
            model_path = os.path.join(os.path.dirname(__file__), '../mobility/mobility_trained.pth')
            if os.path.exists(model_path):
                self.mobility_model.load_state_dict(torch.load(model_path))
            # ---------------------------------------------
            self.mobility_model.eval()
            
    def reset(self, render=False):
        if render:
            traci.start(["sumo-gui", "-c", self.sumocfg])
            for rsu in self.rsus:
                try:
                    traci.poi.add(poiID=rsu["id"], x=rsu["x"], y=rsu["y"], color=(255, 0, 0, 255), poiType="RSU", layer=100)
                    shape = []
                    for i in range(36):
                        angle = i * 10 * math.pi / 180
                        cx = rsu["x"] + rsu["range"] * math.cos(angle)
                        cy = rsu["y"] + rsu["range"] * math.sin(angle)
                        shape.append((cx, cy))  
                    traci.polygon.add(polygonID=rsu["id"]+"_cov", shape=shape, color=(0, 150, 255, 255), fill=False, lineWidth=2, layer=50)
                except:
                    pass 
        else:
            traci.start(["sumo", "-c", self.sumocfg, "--start", "--quit-on-end"])
            
        self.current_time = 0
        self.queues = {r["id"]: [] for r in self.rsus}
        return self._get_state()
        
    def _build_graph(self, vehicle_ids, max_distance=100.0):
        edges = []
        positions = {v_id: traci.vehicle.getPosition(v_id) for v_id in vehicle_ids}
        for i, v1 in enumerate(vehicle_ids):
            for j, v2 in enumerate(vehicle_ids):
                if i != j and math.dist(positions[v1], positions[v2]) < max_distance:
                    edges.append([i, j])
        if not edges:
            return torch.empty((2, 0), dtype=torch.long)
        return torch.tensor(edges, dtype=torch.long).t().contiguous()

    def _get_nearest_rsu(self, pos):
        nearest = self.rsus[0]
        min_dist = float('inf')
        for rsu in self.rsus:
            d = math.dist(pos, (rsu["x"], rsu["y"]))
            if d < min_dist:
                min_dist = d
                nearest = rsu
        return nearest

    def step(self, action):
        traci.simulationStep()
        self.current_time += 1
        
        vehicle_ids = traci.vehicle.getIDList()
        done = traci.simulation.getMinExpectedNumber() <= 0
        reward = 0
        
        if len(vehicle_ids) > 0:
            # --- اولویت 2-ب: تولید وظایف موازی (Parallel Tasks) برای شلوغ کردن صف‌ها ---
            # حداکثر 20 ماشین را بررسی می‌کنیم تا بار پردازشی گراف منطقی بماند
            active_vehicles = vehicle_ids[:20]
            
            # 1. ساخت گراف و استخراج مختصات برای همه
            all_positions = [traci.vehicle.getPosition(v) for v in active_vehicles]
            edge_index = self._build_graph(active_vehicles)
            x_seq = torch.tensor(all_positions, dtype=torch.float32).unsqueeze(1)
            
            if self.use_mobility_detector:
                with torch.no_grad():
                    predicted_futures = self.mobility_model(x_seq, [edge_index], future_steps=5)
            
            # 2. تولید ترافیک پس‌زمینه (ماشین‌های 1 تا 19)
            for idx in range(1, len(active_vehicles)):
                bg_vid = active_vehicles[idx]
                bg_pos = all_positions[idx]
                bg_task = generate_random_task(bg_vid, self.current_time)
                bg_task["T_stay"] = 10.0 # عدد ثابت برای پس‌زمینه
                nearest_rsu = self._get_nearest_rsu(bg_pos)
                self.queues[nearest_rsu["id"]].append(bg_task)
                
            # 3. پردازش ماشین اصلی (نماینده هوش مصنوعی)
            v_id = active_vehicles[0]
            pos = all_positions[0]
            
            selected_rsu_idx = action if action < len(self.rsus) else 0
            if not self.use_collaboration:
                selected_rsu_idx = 0 
            current_rsu = self.rsus[selected_rsu_idx]
            
            # محاسبه T_stay ماشین اصلی
            predicted_t_stay = 15.0
            if self.use_mobility_detector:
                future_positions = predicted_futures[0]
                for step_idx, future_pos in enumerate(future_positions):
                    dist_to_rsu = math.dist((future_pos[0].item(), future_pos[1].item()), (current_rsu["x"], current_rsu["y"]))
                    if dist_to_rsu > current_rsu["range"]:
                        predicted_t_stay = float(step_idx + 1)
                        break
                        
            task = generate_random_task(v_id, self.current_time)
            task["T_stay"] = predicted_t_stay 
            
            self.queues[current_rsu["id"]].append(task)
            
            if self.use_priority:
                self.queues[current_rsu["id"]] = sort_tasks_by_priority(self.queues[current_rsu["id"]])
                
            top_task = self.queues[current_rsu["id"]].pop(0)
            
            # --- اولویت 2-الف: پردازش مشارکتی بین دو RSU (Case 2) ---
            distance = math.dist(pos, (current_rsu["x"], current_rsu["y"]))
            rate = models.v2r_rate(B=10, P_v=0.5, K=1e-3, omega=1e-9, distance=max(distance, 1.0), sigma=2)
            t_up = models.upload_delay(top_task["rho"], rate)
            F_rsu = current_rsu.get("capacity", 2.0)
            
            queue_load = sum(t["phi"] for t in self.queues[current_rsu["id"]])
            t_wait = models.waiting_delay(queue_load, F_rsu)
            
            phi_total = top_task["phi"]
            
            if self.use_collaboration:
                # محاسبه مقدار پردازش قابل انجام در زمان باقی‌مانده محلی
                phi_done_locally = min(phi_total, predicted_t_stay * F_rsu)
                phi_rest = phi_total - phi_done_locally
                
                if phi_rest > 0:
                    # خودرو در حال خروج است! بخشی از وظیفه به RSU بعدی منتقل می‌شود
                    next_rsu_idx = (selected_rsu_idx + 1) % len(self.rsus)
                    next_rsu = self.rsus[next_rsu_idx]
                    
                    r2r_rate = 100.0 # فرض نرخ ثابت بین آنتن‌ها
                    t2 = models.upload_delay(phi_rest, r2r_rate)
                    F_next_rsu = next_rsu.get("capacity", 2.0)
                    t3 = models.processing_delay(phi_rest, F_next_rsu)
                    
                    t1 = predicted_t_stay
                    t_pro = max(t1, t2 + t3) # فرمول موازی‌سازی مقاله
                else:
                    t_pro = models.processing_delay(phi_total, F_rsu)
            else:
                t_pro = models.processing_delay(phi_total, F_rsu)
            # --------------------------------------------------------
            
            total_delay = t_up + t_pro + t_wait
            E_V2R = 0.1 
            E_RSU = 0.05
            energy = t_up * E_V2R + t_pro * E_RSU
            
            if total_delay > top_task["d"]:
                reward = -100.0
            else:
                sigma_w = 0.5
                reward = -(sigma_w * total_delay + (1 - sigma_w) * energy)
                
        next_state = self._get_state(vehicle_ids)
        return next_state, reward, done, []
        
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
            total_load = sum(t["phi"] for t in self.queues[rsu["id"]])
            state.append(total_load)
            
        MAP_W, MAP_H = 2000.0, 1000.0 
        state[0] /= MAP_W
        state[1] /= MAP_H
        state[2] /= 5.0  
        state[3] /= 30.0 
        
        for i in range(4, len(state)):
            state[i] /= 50.0 
            
        return np.array(state, dtype=np.float32)

    def close(self):
        traci.close()