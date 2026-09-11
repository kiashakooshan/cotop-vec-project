import os
import sys
import json
import math
import traci
import torch
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env.task_generator import VehicleTaskScheduler, generate_task_dag
from env.fleet_manager import FleetManager
from models.models import RSU_Node, sort_tasks_by_priority, v2r_rate, r2r_rate
from mobility.gat_gru import MobilityDetector
from mobility.train_mobility import build_graph

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
            rsus_data = json.load(f)
            
        self.rsus = rsus_data
        self.rsu_nodes = [
            RSU_Node(r["id"], r["cpu_type"], r["num_processors"], r["speed_factor"], r["power_draw"]) 
            for r in self.rsus
        ]
        
        self.current_time = 0
        self.fleet_manager = FleetManager(num_vehicles=40)
        self.schedulers = {} 
        
        self.episode_energy = 0.0
        self.episode_makespans = []
        self.mobility_predictions = {}
        
        self.last_dag_features = {}
        self.pending_predictions = {}
        
        if self.use_mobility_detector:
            self.mobility_model = MobilityDetector(in_dim=2, hidden=32, gru_hidden=64)
            model_path = os.path.join(os.path.dirname(__file__), '../mobility/mobility_trained.pth')
            if os.path.exists(model_path):
                self.mobility_model.load_state_dict(torch.load(model_path))
            self.mobility_model.eval()

    def reset(self, render=False):
        if render:
            traci.start(["sumo-gui", "-c", self.sumocfg, "--no-warnings"])
        else:
            traci.start(["sumo", "-c", self.sumocfg, "--start", "--quit-on-end", "--no-warnings"])
            
        self.current_time = 0
        self.fleet_manager.spawn_fixed_fleet()
        
        self.schedulers = {f"car_{i}": VehicleTaskScheduler(f"car_{i}") for i in range(40)}
        
        self.episode_energy = 0.0
        self.episode_controlled_energy = 0.0
        self.episode_makespans = []
        self.mobility_predictions = {f"car_{i}": [] for i in range(40)}
        
        self.last_dag_features = {f"car_{i}": [0.0, 0.0, 0.0] for i in range(40)}
        self.pending_predictions = {}
        
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

    def _get_next_rsu_on_route(self, veh_id, current_rsu):
        current_idx = next(i for i, rsu in enumerate(self.rsu_nodes) if rsu.id == current_rsu.id)
        next_idx = (current_idx + 1) % len(self.rsu_nodes)
        return self.rsu_nodes[next_idx]

    def _estimate_t_stay(self, current_pos, predicted_pos, rsu_pos, rsu_range):
        """Estimate dwell time using the REAL mobility-model prediction (paper's 
    t1)."""
        cx, cy = current_pos
        px, py = predicted_pos
        rx, ry = rsu_pos
        dx_to_rsu, dy_to_rsu = cx - rx, cy - ry
        dist_to_center = math.hypot(dx_to_rsu, dy_to_rsu)
        if dist_to_center < 1e-6:
            return 50.0  # vehicle is exactly at the RSU; treat as long dwell time
        # unit vector pointing AWAY from the RSU
        away_x, away_y = dx_to_rsu / dist_to_center, dy_to_rsu / dist_to_center
        # velocity implied by the model's one-step-ahead prediction
        vx, vy = px - cx, py - cy
        # how fast the vehicle is moving away from the RSU )radial speed(
        radial_speed = vx * away_x + vy * away_y
        remaining_dist = max(0.0, rsu_range - dist_to_center)
        if radial_speed <= 0.01:
            return 50.0  # not moving away meaningfully -> effectively staying
        return remaining_dist / radial_speed

    def _schedule_dag_on_rsu(self, dag, primary_rsu, predicted_t_stay, use_collaboration=True):
        task_results = {}
        total_energy = 0.0
        BASE_CAPACITY = 2.0
        E_V2R = 0.5
        E_R2R = 0.2
        
        rsu_data = next(r for r in self.rsus if r["id"] == primary_rsu.id)
        rsu_pos = (rsu_data["x"], rsu_data["y"])
        vehicle_pos = traci.vehicle.getPosition(dag["veh_id"])
        
        tasks_by_layer = {}
        for t in dag["tasks"].values():
            tasks_by_layer.setdefault(t["layer"], []).append(t)
            
        sorted_tasks = []
        for layer_idx in sorted(tasks_by_layer.keys()):
            layer_tasks = tasks_by_layer[layer_idx]
            if self.use_priority:
                layer_tasks = sort_tasks_by_priority(layer_tasks)
            sorted_tasks.extend(layer_tasks)
            
        for task in sorted_tasks:
            proc = primary_rsu.pick_free_processor()
            parent_ready_time = dag["release_time"]
            
            for p_id in task["parents"]:
                if p_id in task_results:
                    parent_ready_time = max(parent_ready_time, task_results[p_id]["finish_time"])
            
            distance = math.dist(vehicle_pos, rsu_pos)
            rate = v2r_rate(B=10, P_v=0.5, K=1e-3, omega=1e-9, distance=distance, sigma=2)
            t_up = task["rho"] / rate
            trans_e = t_up * E_V2R
            total_energy += trans_e
            
            start_time = max(proc.free_at, parent_ready_time) + t_up
            full_processing_time = task["phi"] / (BASE_CAPACITY * proc.speed_factor)
            
            if use_collaboration and (start_time + full_processing_time - dag["release_time"]) > predicted_t_stay:
                time_available = max(0.0, predicted_t_stay - (start_time - dag["release_time"]))
                phi_done_locally = time_available * BASE_CAPACITY * proc.speed_factor
                phi_rest = max(0.0, task["phi"] - phi_done_locally)
                
                next_rsu = self._get_next_rsu_on_route(dag["veh_id"], primary_rsu)
                next_proc = next_rsu.pick_free_processor()
                next_rsu_data = next(r for r in self.rsus if r["id"] == next_rsu.id)
                
                r2r_distance = math.dist(rsu_pos, (next_rsu_data["x"], next_rsu_data["y"]))
                rate_r2r = r2r_rate(distance=r2r_distance)
                t2 = task["rho"] / rate_r2r 
                t3 = phi_rest / (BASE_CAPACITY * next_proc.speed_factor)
                
                finish_time = start_time + max(predicted_t_stay, t2 + t3)
                next_proc.free_at = finish_time
                
                total_energy += (time_available * proc.power_draw) + (t3 * next_proc.power_draw) + (task["rho"] * E_R2R)
            else:
                finish_time = start_time + full_processing_time
                proc.free_at = finish_time
                total_energy += full_processing_time * proc.power_draw
                
            task_results[task["id"]] = {"finish_time": finish_time}
            
        makespan = max([task_results[t]["finish_time"] for t in dag["exit_tasks"]]) - dag["release_time"]
        return makespan, total_energy

    def step(self, action):
        traci.simulationStep()
        self.current_time += 1
        self.fleet_manager.keep_fleet_closed()
        
        raw_vehicle_ids = traci.vehicle.getIDList()
        vehicle_ids = [v for v in raw_vehicle_ids if v.startswith("car_")]
        
        done = traci.simulation.getMinExpectedNumber() <= 0
        reward = 0
        
        if len(vehicle_ids) > 0:
            active_vehicles = vehicle_ids[:40] 
            all_positions = [traci.vehicle.getPosition(v) for v in active_vehicles]
            
            pos_dict = {v_id: pos for v_id, pos in zip(active_vehicles, all_positions)}
            
            if self.use_mobility_detector:
                x_seq = torch.tensor(all_positions, dtype=torch.float32).unsqueeze(1)
                
                edges = build_graph(active_vehicles, pos_dict, max_distance=100.0) 
                
                with torch.no_grad():
                    predicted_futures = self.mobility_model(x_seq, [edges], future_steps=1)
                
                for v_id in active_vehicles:
                    if v_id in self.pending_predictions:
                        predicted_pos = self.pending_predictions[v_id]
                        actual_pos_now = pos_dict.get(v_id)
                        if actual_pos_now is not None:
                            self.mobility_predictions[v_id].append((predicted_pos, actual_pos_now))
                
                for idx, v_id in enumerate(active_vehicles):
                    pred_pos = (predicted_futures[idx][0][0].item(), predicted_futures[idx][0][1].item())
                    self.pending_predictions[v_id] = pred_pos

            selected_rsu_idx = action if action < len(self.rsu_nodes) else 0
            main_rsu_node = self.rsu_nodes[selected_rsu_idx]
            
            step_makespans = []
            step_energies = []

            controlled_idx = self.current_time % len(active_vehicles)

            for idx in range(len(active_vehicles)):
                v_id = active_vehicles[idx]
                pos = all_positions[idx]
                
                scheduler = self.schedulers.get(v_id)
                
                if scheduler and scheduler.should_generate(self.current_time):
                    new_dag = generate_task_dag(v_id, self.current_time)
                    
                    total_rho = sum(t["rho"] for t in new_dag["tasks"].values())
                    total_phi = sum(t["phi"] for t in new_dag["tasks"].values())
                    max_d = max(t["d"] for t in new_dag["tasks"].values())
                    
                    self.last_dag_features[v_id] = [total_rho / 50.0, total_phi / 50.0, max_d / 20.0]
                    
                    scheduler.schedule_next(self.current_time)
                    
                    if idx == controlled_idx:
                        target_rsu = main_rsu_node
                    else:
                        target_rsu = self._get_nearest_rsu_node(pos)
                        
                    target_rsu_data = next(r for r in self.rsus if r["id"] == target_rsu.id)
                    dist_to_rsu = math.dist(pos, (target_rsu_data["x"], target_rsu_data["y"]))
                    if self.use_mobility_detector and v_id in self.pending_predictions:
                        predicted_pos = self.pending_predictions[v_id]
                        predicted_t_stay = self._estimate_t_stay(
                            current_pos=pos,
                            predicted_pos=predicted_pos,
                            rsu_pos=(target_rsu_data["x"], target_rsu_data["y"]),
                            rsu_range=target_rsu_data.get("range", 400.0)
                        )
                    else:
                        # fallback heuristic — used when mobility detector is OFF (ablation "w/o_MD")
                        predicted_t_stay = max(0.1, (target_rsu_data.get("range", 400.0) - dist_to_rsu) / 15.0)

                    makespan, energy = self._schedule_dag_on_rsu(
                        new_dag, 
                        target_rsu, 
                        predicted_t_stay=predicted_t_stay,
                        use_collaboration=(idx == controlled_idx and self.use_collaboration)
                    )
                    
                    step_makespans.append(makespan)
                    step_energies.append(energy)
                    if idx == controlled_idx:
                        self.episode_controlled_energy += energy

            if len(step_makespans) > 0:
                avg_makespan = sum(step_makespans) / len(step_makespans)
                avg_energy = sum(step_energies) / len(step_energies)
                
                self.episode_energy += sum(step_energies)
                self.episode_makespans.extend(step_makespans)
                
                sigma_w = 0.6
                reward = -(sigma_w * avg_makespan + (1 - sigma_w) * avg_energy)
            else:
                reward = 0  
                
        next_state = self._get_state(vehicle_ids)
        return next_state, reward, done, []

    def _get_state(self, vehicle_ids=None):
        if vehicle_ids is None:
            raw_vehicle_ids = traci.vehicle.getIDList()
            vehicle_ids = [v for v in raw_vehicle_ids if v.startswith("car_")]
            
        state = [0.0, 0.0] 
        task_state = [0.0, 0.0, 0.0] 

        if len(vehicle_ids) > 0:
            actual_v_id = vehicle_ids[0]
            pos = traci.vehicle.getPosition(actual_v_id)
            state = [pos[0] / 2000.0, pos[1] / 1000.0]
            
            if actual_v_id in self.last_dag_features:
                task_state = self.last_dag_features[actual_v_id]
                
        state.extend(task_state)
            
        for node in self.rsu_nodes:
            avg_free_time = sum(p.free_at for p in node.processors) / len(node.processors)
            load = max(0, avg_free_time - self.current_time)
            state.append(load / 50.0)
            
        return np.array(state, dtype=np.float32)

    def close(self):
        traci.close()