import os
import sys
import traci
import math
from task_generator import generate_random_task

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Error: Please declare environment variable 'SUMO_HOME'")

class VECEnv:
    def __init__(self, sumocfg_path, rsus):
        self.sumocfg = sumocfg_path
        self.rsus = rsus
        self.queues = {r["id"]: [] for r in rsus}
        self.current_time = 0
        
    def reset(self):
        traci.start(["sumo", "-c", self.sumocfg])
        self.current_time = 0
        return self._get_state()
        
    def step(self):
        traci.simulationStep()
        self.current_time += 1
        
        vehicle_ids = traci.vehicle.getIDList()
        step_data = []
        
        for v_id in vehicle_ids:
            pos = traci.vehicle.getPosition(v_id)
            speed = traci.vehicle.getSpeed(v_id)
            
            # 1. پیدا کردن RSUهایی که این ماشین در برد 400 متری آنهاست
            connected_rsus = []
            for rsu in self.rsus:
                distance = math.dist(pos, (rsu["x"], rsu["y"]))
                if distance <= rsu["range"]:
                    connected_rsus.append({"rsu_id": rsu["id"], "distance": distance})
            
            # 2. تولید یک وظیفه جدید برای این ماشین در این لحظه
            new_task = generate_random_task(v_id, self.current_time)
            
            step_data.append({
                "id": v_id, 
                "position": pos, 
                "speed": speed,
                "connected_rsus": connected_rsus,
                "task": new_task
            })
            
        return step_data
        
    def _get_state(self):
        return {}

    def close(self):
        traci.close()

# ---------------- بلوک تست کد ----------------
if __name__ == "__main__":
    # تعریف دو RSU با برد 400 متر طبق مقاله
    test_rsus = [
        {"id": "RSU1", "x": 100, "y": 100, "range": 400},
        {"id": "RSU2", "x": 300, "y": 200, "range": 400}
    ]
    
    config_file = "../sumo/osm.sumocfg" 
    
    env = VECEnv(config_file, test_rsus)
    env.reset()
    
    # اجرای 5 قدم برای تست
    for i in range(5):
        print(f"\n--- Time Step {i} ---")
        info = env.step()
        
        if len(info) > 0:
            veh = info[0]
            print(f"Vehicle: {veh['id']}, Speed: {veh['speed']:.2f}")
            print(f"   -> Connected RSUs: {veh['connected_rsus']}")
            print(f"   -> Generated Task: Size={veh['task']['rho']:.2f}MB, Deadline={veh['task']['d']:.2f}s")
            
    env.close()