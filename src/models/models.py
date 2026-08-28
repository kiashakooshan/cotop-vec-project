import math

def v2r_rate(B, P_v, K, omega, distance, sigma):
    """فرمول 1: محاسبه نرخ انتقال V2R بر اساس قضیه شانون"""
    # برای جلوگیری از خطای تقسیم بر صفر در صورت صفر بودن فاصله
    if distance == 0:
        distance = 0.1
    return B * math.log2(1 + (P_v * K) / (omega * distance**sigma))

def upload_delay(task_size, rate):
    """فرمول 3: تأخیر آپلود وظیفه"""
    return task_size / rate

def processing_delay(cpu_cycles, F_rsu):
    """فرمول 4: تأخیر پردازش در RSU"""
    return cpu_cycles / F_rsu

def waiting_delay(queue_load, F_rsu):
    """فرمول 5: تأخیر انتظار در صف RSU"""
    return queue_load / F_rsu

def task_priority(alpha, beta, T_stay, rho, d):
    """فرمول 23: محاسبه اولویت وظیفه بر اساس زمان ماندگاری، حجم و مهلت"""
    if T_stay <= 0:
        T_stay = 0.1 # جلوگیری از خطای تقسیم بر صفر
    return alpha * math.exp(-1/T_stay) + beta * (rho / d)