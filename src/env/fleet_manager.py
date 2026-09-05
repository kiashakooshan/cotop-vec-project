import traci
import random

# وقتی ماشین به این تعداد یال (Edge) از انتهای مسیرش رسید، مسیرش تمدید می‌شود
REROUTE_THRESHOLD = 2 

class FleetManager:
    def __init__(self, num_vehicles=40):
        self.num_vehicles = num_vehicles
        self.all_edges = []

    def spawn_fixed_fleet(self):
        """در ثانیه صفر اجرا می‌شود تا دقیقاً ۴۰ ماشین را روی نقشه ظاهر کند."""
        # دریافت تمام خیابان‌های نقشه (خیابان‌های داخلی که با ':' شروع می‌شوند را نادیده می‌گیریم)
        self.all_edges = [edge for edge in traci.edge.getIDList() if not edge.startswith(':')]
        
        for i in range(self.num_vehicles):
            veh_id = f"car_{i}"
            start_edge = random.choice(self.all_edges)
            dest_edge = random.choice(self.all_edges)
            
            route_id = f"route_{i}"
            
            try:
                # یک مسیر موقت فقط با نقطه شروع می‌سازیم
                traci.route.add(route_id, [start_edge])
                # ماشین را به نقشه اضافه می‌کنیم
                traci.vehicle.add(veh_id, route_id, typeID="DEFAULT_VEHTYPE")
                # از SUMO می‌خواهیم خودش بهترین مسیر را تا مقصد جدید پیدا کند
                traci.vehicle.changeTarget(veh_id, dest_edge)
            except traci.exceptions.TraCIException as e:
                print(f"⚠️ Warning: Could not spawn {veh_id} on edge {start_edge}. Retrying next step.")

    def keep_fleet_closed(self):
        """در هر گام از شبیه‌سازی اجرا می‌شود تا نگذارد هیچ ماشینی از نقشه خارج شود."""
        if not self.all_edges:
            return
            
        for veh_id in traci.vehicle.getIDList():
            try:
                route = traci.vehicle.getRoute(veh_id)
                route_idx = traci.vehicle.getRouteIndex(veh_id)
                remaining = len(route) - route_idx
                
                # اگر ماشین به آخر خط نزدیک شده بود، یک مقصد تصادفی جدید به آن می‌دهیم
                if remaining <= REROUTE_THRESHOLD:
                    new_dest = random.choice(self.all_edges)
                    traci.vehicle.changeTarget(veh_id, new_dest)
            except traci.exceptions.TraCIException:
                # اگر ماشین در حال تلپورت شدن باشد یا خطای مسیریابی بدهد، نادیده می‌گیریم
                pass