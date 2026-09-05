import traci
import random

REROUTE_THRESHOLD = 2 

class FleetManager:
    def __init__(self, num_vehicles=40):
        self.num_vehicles = num_vehicles
        self.all_edges = []

    def spawn_fixed_fleet(self):
        """تولید ماشین‌ها فقط در مسیرهای دارای ارتباط فیزیکی"""
        self.all_edges = [edge for edge in traci.edge.getIDList() if not edge.startswith(':')]
        
        spawned = 0
        attempts = 0
        max_attempts = self.num_vehicles * 15 
        
        while spawned < self.num_vehicles and attempts < max_attempts:
            attempts += 1
            start_edge = random.choice(self.all_edges)
            dest_edge = random.choice(self.all_edges)
            
            route = traci.simulation.findRoute(start_edge, dest_edge)
            
            if route and len(route.edges) > 1:
                veh_id = f"car_{spawned}"
                route_id = f"route_{spawned}_{attempts}"
                traci.route.add(route_id, route.edges)
                
                try:
                    traci.vehicle.add(veh_id, route_id, typeID="DEFAULT_VEHTYPE")
                    spawned += 1
                except traci.exceptions.TraCIException:
                    pass
                    
        if spawned < self.num_vehicles:
            print(f"⚠️ Warning: Could only spawn {spawned}/{self.num_vehicles} cars due to map dead-ends.")

    def keep_fleet_closed(self):
        """تمدید مسیر با جستجوی هوشمندانه مقاصد در دسترس"""
        if not self.all_edges:
            return
            
        for veh_id in traci.vehicle.getIDList():
            # --- جادوی جدید: ماشین‌های متفرقه را کاملاً نادیده بگیر ---
            if not veh_id.startswith("car_"):
                continue
            # -----------------------------------------------------------
                
            try:
                route = traci.vehicle.getRoute(veh_id)
                route_idx = traci.vehicle.getRouteIndex(veh_id)
                remaining = len(route) - route_idx
                
                if remaining <= REROUTE_THRESHOLD:
                    current_edge = traci.vehicle.getRoadID(veh_id)
                    
                    if current_edge.startswith(':'):
                        continue
                        
                    for _ in range(5):
                        new_dest = random.choice(self.all_edges)
                        new_route = traci.simulation.findRoute(current_edge, new_dest)
                        
                        if new_route and len(new_route.edges) > 1:
                            traci.vehicle.setRoute(veh_id, new_route.edges)
                            break
            except traci.exceptions.TraCIException:
                pass