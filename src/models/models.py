import math

class Processor:
    """نماینده یک هسته پردازشی فیزیکی در سرور"""
    def __init__(self, proc_id, speed_factor, power_draw):
        self.proc_id = proc_id
        self.speed_factor = speed_factor
        self.power_draw = power_draw
        self.free_at = 0.0  # زمان شبیه‌سازی که این پردازنده آزاد می‌شود

class RSU_Node:
    """نماینده یک سرور Edge با چندین هسته پردازشی ناهمگون"""
    def __init__(self, rsu_id, cpu_type, num_processors, speed_factor, power_draw):
        self.id = rsu_id
        self.cpu_type = cpu_type
        self.processors = [
            Processor(f"{rsu_id}_p{i}", speed_factor, power_draw)
            for i in range(num_processors)
        ]

    def pick_free_processor(self):
        """انتخاب پردازنده‌ای که زودتر از بقیه آزاد می‌شود (List Scheduling)"""
        return min(self.processors, key=lambda p: p.free_at)

def v2r_rate(B, P_v, K, omega, distance, sigma):
    """محاسبه نرخ انتقال V2R بر اساس قضیه شانون"""
    if distance <= 0.01:
        distance = 0.01
    return B * math.log2(1 + (P_v * K) / (omega * distance**sigma))

def r2r_rate(distance=500.0):
    """نرخ انتقال بین دو RSU برای پاس‌کاری وظایف (ارتباطات فیبر/بک‌هاول)"""
    return 100.0  # فرض یک نرخ ثابت و بالا (مثلاً 100 Mbps) برای سادگی

def upload_delay(task_size, rate):
    return task_size / rate

def transfer_delay(data_size, rate):
    """تأخیر انتقال داده بین دو RSU مختلف برای زیروظایف وابسته (DAG)"""
    if data_size == 0.0:
        return 0.0
    return data_size / rate

def processing_delay(cpu_cycles, base_capacity, speed_factor=1.0):
    """تأخیر پردازش با در نظر گرفتن ضریب سرعت پردازنده ناهمگون"""
    return cpu_cycles / (base_capacity * speed_factor)

def waiting_delay(queue_load, F_rsu):
    """(تابع قدیمی برای سازگاری عقب‌رو)"""
    return queue_load / F_rsu

def calculate_task_priority(alpha, beta, T_stay, rho, d):
    """محاسبه امتیاز اولویت مقاله"""
    if T_stay <= 0.01:
        T_stay = 0.01 
    return alpha * math.exp(-1 / T_stay) + beta * (rho / d)

def sort_tasks_by_priority(tasks_queue, alpha=0.3, beta=0.7):
    for task in tasks_queue:
        task["priority"] = calculate_task_priority(
            alpha, 
            beta, 
            task.get("T_stay", 10.0), 
            task["rho"], 
            task["d"]
        )
    return sorted(tasks_queue, key=lambda t: t["priority"], reverse=True)