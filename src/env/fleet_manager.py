import traci
import random

# آستانه را بالا بردیم تا ماشین‌ها قبل از رسیدن به ته خط مسیرشان عوض شود
REROUTE_THRESHOLD = 5 

class FleetManager:
    def __init__(self, num_vehicles=40):
        self.num_vehicles = num_vehicles
        self.all_edges = []
        # شمارنده برای ساختن ID های جدید و گول زدن SUMO
        self.respawn_counts = {f"car_{i}": 0 for i in range(num_vehicles)}

    def spawn_fixed_fleet(self):
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
                
                try:
                    traci.route.add(route_id, route.edges)
                    traci.vehicle.add(veh_id, route_id, typeID="DEFAULT_VEHTYPE")
                    spawned += 1
                except traci.exceptions.TraCIException:
                    pass
                    
        if spawned < self.num_vehicles:
            print(f"⚠️ Warning: Could only spawn {spawned}/{self.num_vehicles} cars.")

    def keep_fleet_closed(self):
        if not self.all_edges:
            return
            
        active_base_cars = set()
        
        for veh_id in traci.vehicle.getIDList():
            if not veh_id.startswith("car_"):
                continue
                
            # قیچی کردن پسوندها: "car_30_2" تبدیل می‌شود به "car_30"
            base_id = "_".join(veh_id.split('_')[:2])
            active_base_cars.add(base_id)
            
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

        # --- سیستم احیای خودکار با ID های جدید ---
        expected_cars = {f"car_{i}" for i in range(self.num_vehicles)}
        missing_cars = expected_cars - active_base_cars
        
        for base_id in missing_cars:
            self.respawn_counts[base_id] += 1
            new_veh_id = f"{base_id}_{self.respawn_counts[base_id]}"
            
            for _ in range(15):
                start_edge = random.choice(self.all_edges)
                dest_edge = random.choice(self.all_edges)
                route = traci.simulation.findRoute(start_edge, dest_edge)
                
                if route and len(route.edges) > 1:
                    route_id = f"route_{new_veh_id}_{random.randint(10000, 99999)}"
                    try:
                        # هر دو دستور داخل Try قرار گرفتند تا برنامه کرش نکند
                        traci.route.add(route_id, route.edges)
                        traci.vehicle.add(new_veh_id, route_id, typeID="DEFAULT_VEHTYPE")
                        break
                    except traci.exceptions.TraCIException:
                        pass