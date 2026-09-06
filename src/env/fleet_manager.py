import traci
import random

# آستانه را بالا بردیم: ماشین خیلی قبل‌تر از رسیدن به انتها مسیرش عوض می‌شود
REROUTE_THRESHOLD = 15 

class FleetManager:
    def __init__(self, num_vehicles=40):
        self.num_vehicles = num_vehicles
        self.all_edges = []

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
            
            # فقط مسیرهای نسبتاً طولانی را برای شروع انتخاب می‌کنیم
            if route and len(route.edges) > 10:
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
            
        # گام 11: تضمین اینکه دقیقاً 40 ماشین وارد شبیه‌سازی شوند
        assert spawned == self.num_vehicles, f"CRITICAL ERROR: Failed to spawn exact number of vehicles ({self.num_vehicles})."

    def keep_fleet_closed(self):
        """ایده جدید: تغییر مسیر به مقاصد دور، بدون هیچ‌گونه Respawn"""
        if not self.all_edges:
            return
            
        for veh_id in traci.vehicle.getIDList():
            if not veh_id.startswith("car_"):
                continue
                
            try:
                route = traci.vehicle.getRoute(veh_id)
                route_idx = traci.vehicle.getRouteIndex(veh_id)
                remaining = len(route) - route_idx
                
                # اگر به 15 قدمی انتهای مسیر رسید
                if remaining <= REROUTE_THRESHOLD:
                    current_edge = traci.vehicle.getRoadID(veh_id)
                    if current_edge.startswith(':'):
                        continue
                        
                    # تلاش برای پیدا کردن یک مقصد دور و دراز
                    for _ in range(10):
                        new_dest = random.choice(self.all_edges)
                        new_route = traci.simulation.findRoute(current_edge, new_dest)
                        
                        # حتماً مقصدی را انتخاب کن که حداقل 10 یال فاصله داشته باشد
                        if new_route and len(new_route.edges) > 10:
                            traci.vehicle.setRoute(veh_id, new_route.edges)
                            break
            except traci.exceptions.TraCIException:
                pass