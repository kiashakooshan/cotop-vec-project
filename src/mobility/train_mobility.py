import os
import sys
import math
import torch
import torch.nn as nn
import torch.optim as optim
import traci

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from mobility.gat_gru import MobilityDetector
from env.fleet_manager import FleetManager # --- اضافه شدن مدیر ناوگان ---

def build_graph(vehicle_ids, positions, max_distance=100.0):
    edges = []
    for i, v1 in enumerate(vehicle_ids):
        for j, v2 in enumerate(vehicle_ids):
            if i != j and math.dist(positions[v1], positions[v2]) < max_distance:
                edges.append([i, j])
    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(edges, dtype=torch.long).t().contiguous()

def train_gat_gru():
    print("🧠 Starting Mobility Model (GAT+GRU) Training...")
    
    sumocfg = "../sumo/osm.sumocfg"
    traci.start(["sumo", "-c", sumocfg, "--start", "--quit-on-end", "--no-warnings"])
    
    # --- ساخت 40 ماشین در ثانیه صفر ---
    fleet_manager = FleetManager(num_vehicles=40)
    fleet_manager.spawn_fixed_fleet()
    
    model = MobilityDetector(in_dim=2, hidden=32, gru_hidden=64)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss() 
    
    vehicle_history = {}
    epochs = 1000 
    
    for step in range(epochs):
        traci.simulationStep()
        # --- حفظ ماشین‌ها در نقشه ---
        fleet_manager.keep_fleet_closed()
        
        vehicle_ids = traci.vehicle.getIDList()
        
        if len(vehicle_ids) < 2:
            continue
            
        positions = {v_id: traci.vehicle.getPosition(v_id) for v_id in vehicle_ids}
        
        for v_id in vehicle_ids:
            if v_id not in vehicle_history:
                vehicle_history[v_id] = []
            vehicle_history[v_id].append([positions[v_id][0], positions[v_id][1]])
            
            if len(vehicle_history[v_id]) > 6:
                vehicle_history[v_id].pop(0)
                
        valid_vids = [v_id for v_id in vehicle_ids if len(vehicle_history[v_id]) == 6]
        
        if len(valid_vids) > 0:
            x_list = [vehicle_history[v][:5] for v in valid_vids]
            y_list = [[vehicle_history[v][5]] for v in valid_vids] 
            
            x_seq = torch.tensor(x_list, dtype=torch.float32)
            y_true = torch.tensor(y_list, dtype=torch.float32)
            
            edge_index = build_graph(valid_vids, positions)
            
            optimizer.zero_grad()
            y_pred = model(x_seq, [edge_index]*5, future_steps=1)
            
            loss = criterion(y_pred, y_true)
            loss.backward()
            optimizer.step()
                
        if step % 100 == 0:
            print(f"Step {step}/{epochs} | MSE Loss: {loss.item() if 'loss' in locals() else 0.0:.4f}")

    traci.close()
    
    torch.save(model.state_dict(), "mobility_trained.pth")
    print("✅ Mobility Model trained and saved as 'mobility_trained.pth'")

if __name__ == "__main__":
    train_gat_gru()