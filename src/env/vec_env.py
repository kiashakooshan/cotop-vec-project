import os
import sys
import json
import math
import traci
import torch
import numpy as np
import csv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env.task_generator import VehicleTaskScheduler, generate_task_dag
from env.fleet_manager import FleetManager
from models.models import RSU_Node
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
        
        # 1. لود کردن آنتن‌ها و ساخت سخت‌افزار ناهمگون (A7/A12)
        with open(rsus_json_path, 'r') as f:
            rsus_data = json.load(f)
            
        self.rsus = rsus_data
        self.rsu_nodes = [
            RSU_Node(r["id"], r["cpu_type"], r["num_processors"], r["speed_factor"], r["power_draw"]) 
            for r in self.rsus
        ]
        
        self.current_time = 0
        self.fleet_manager = FleetManager(num_vehicles=40)
        self.schedulers = {} # تایمرهای تولید وظیفه برای هر ماشین
        
        # متغیرهای گزارش‌گیری
        self.episode_energy = 0.0
        self.episode_makespans = []
        self.mobility_predictions = {}
        
        if self.use_mobility_detector:
            self.mobility_model = MobilityDetector(in_dim=2, hidden=32, gru_hidden=64)
            model_path = os.path.join(os.path.dirname(__file__), '../mobility/mobility_trained.pth')
            if os.path.exists(model_path):
                self.mobility_model.load_state_dict(torch.load(model_path))
            self.mobility_model.eval()

    def reset(self, render=False):
        if render:
            traci.start(["sumo-gui", "-c", self.sumocfg])
        else:
            traci.start(["sumo", "-c", self.sumocfg, "--start", "--quit-on-end"])
            
        self.current_time = 0
        self.fleet_manager.spawn_fixed_fleet()
        
        # مقداردهی زمان‌بندها برای 40 ماشین
        self.schedulers = {f"car_{i}": VehicleTaskScheduler(f"car_{i}") for i in range(40)}
        
        self.episode_energy = 0.0
        self.episode_makespans = []
        self.mobility_predictions = {f"car_{i}": [] for i in range(40)}
        
        # ریست کردن وضعیت پردازنده‌ها
        for node in self.rsu_nodes:
            for p in node.processors:
                p.free_at = 0.0
                
        return self._get_state()
        
    def _get_nearest_rsu_node(self, pos):
        nearest_idx = 0
        min_dist = float('inf')
        for i, rsu in enumerate(self.rsus):
            d = math.dist(pos, (rsu["x"], rsu["y"]))
            if d < min_dist:
                min_dist = d
                nearest_idx = i
        return self.rsu_nodes[nearest_idx]

    def _schedule_dag_on_rsu(self, dag, rsu_node):
        """زمان‌بندی زیروظایف DAG روی پردازنده‌های فیزیکی RSU"""
        task_results = {}
        dag_energy = 0.0
        BASE_CAPACITY = 2.0
        
        # پردازش وظایف بر اساس لایه‌ها تا قید والد-فرزندی رعایت شود
        sorted_tasks = sorted(dag["tasks"].values(), key=lambda x: x["layer"])
        
        for task in sorted_tasks:
            proc = rsu_node.pick_free_processor()
            parent_ready_time = dag["release_time"]
            
            # قید وابستگی: وظیفه باید منتظر تمام والدینش بماند
            for p_id in task["parents"]:
                p_finish = task_results[p_id]["finish_time"]
                parent_ready_time = max(parent_ready_time, p_finish)
                
            start_time = max(proc.free_at, parent_ready_time)
            processing_time = task["phi"] / (BASE_CAPACITY * proc.speed_factor)
            finish_time = start_time + processing_time
            
            energy = processing_time * proc.power_draw
            dag_energy += energy
            proc.free_at = finish_time
            
            task_results[task["id"]] = {"finish_time": finish_time}
            
        # محاسبه Makespan کل DAG
        makespan = max([task_results[t]["finish_time"] for t in dag["exit_tasks"]]) - dag["release_time"]
        return makespan, dag_energy

    def step(self, action):
        traci.simulationStep()
        self.current_time += 1
        self.fleet_manager.keep_fleet_closed()
        
        vehicle_ids = traci.vehicle.getIDList()
        done = traci.simulation.getMinExpectedNumber() <= 0
        reward = 0
        
        if len(vehicle_ids) > 0:
            active_vehicles = vehicle_ids[:40] 
            all_positions = [traci.vehicle.getPosition(v) for v in active_vehicles]
            
            # --- گزارش‌گیری تحرک (ADE/FDE) ---
            if self.use_mobility_detector:
                x_seq = torch.tensor(all_positions, dtype=torch.float32).unsqueeze(1)
                edges = torch.empty((2, 0), dtype=torch.long) # ساده‌سازی گراف برای سرعت
                with torch.no_grad():
                    predicted_futures = self.mobility_model(x_seq, [edges], future_steps=1)
                
                for idx, v_id in enumerate(active_vehicles):
                    pred_pos = (predicted_futures[idx][0][0].item(), predicted_futures[idx][0][1].item())
                    actual_pos = all_positions[idx]
                    self.mobility_predictions[v_id].append((pred_pos, actual_pos))
            # -------------------------------

            # انتخاب RSU توسط عامل RL برای ماشین اصلی
            selected_rsu_idx = action if action < len(self.rsu_nodes) else 0
            main_rsu_node = self.rsu_nodes[selected_rsu_idx]
            
            step_makespans = []
            step_energies = []

            # تولید و پردازش گراف‌ها (DAG)
            for idx in range(len(active_vehicles)):
                v_id = active_vehicles[idx]
                pos = all_positions[idx]
                scheduler = self.schedulers.get(v_id)
                
                # بررسی اینکه آیا زمان تولید DAG جدید برای این ماشین رسیده است؟
                if scheduler and scheduler.should_generate(self.current_time):
                    new_dag = generate_task_dag(v_id, self.current_time)
                    scheduler.schedule_next(self.current_time)
                    
                    if idx == 0:
                        # ماشین اصلی: اعمال اکشن شبکه RL
                        target_rsu = main_rsu_node
                    else:
                        # ماشین‌های پس‌زمینه: نزدیک‌ترین RSU
                        target_rsu = self._get_nearest_rsu_node(pos)
                        
                    # اجرای DAG روی سخت‌افزار
                    makespan, energy = self._schedule_dag_on_rsu(new_dag, target_rsu)
                    step_makespans.append(makespan)
                    step_energies.append(energy)

            # محاسبه پاداش شبکه RL بر اساس Makespan و Energy
            if len(step_makespans) > 0:
                avg_makespan = sum(step_makespans) / len(step_makespans)
                avg_energy = sum(step_energies) / len(step_energies)
                
                self.episode_energy += sum(step_energies)
                self.episode_makespans.extend(step_makespans)
                
                sigma_w = 0.6
                reward = -(sigma_w * avg_makespan + (1 - sigma_w) * avg_energy)
            else:
                reward = 0  # اگر در این ثانیه DAG تولید نشد، پاداش صفر است
                
        next_state = self._get_state(vehicle_ids)
        return next_state, reward, done, []

    def _get_state(self, vehicle_ids=None):
        if vehicle_ids is None:
            vehicle_ids = traci.vehicle.getIDList()
            
        state = [0.0, 0.0] 
        if len(vehicle_ids) > 0:
            v_id = vehicle_ids[0]
            pos = traci.vehicle.getPosition(v_id)
            state = [pos[0] / 2000.0, pos[1] / 1000.0]
            
        # به جای صف تخت، میانگین زمان آزاد شدن پردازنده‌های هر RSU را به عامل می‌دهیم
        for node in self.rsu_nodes:
            avg_free_time = sum(p.free_at for p in node.processors) / len(node.processors)
            load = max(0, avg_free_time - self.current_time)
            state.append(load / 50.0)
            
        return np.array(state, dtype=np.float32)

    def close(self):
        traci.close()