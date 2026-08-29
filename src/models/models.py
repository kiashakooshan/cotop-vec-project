import math

def v2r_rate(B, P_v, K, omega, distance, sigma):
    """فرمول 1: محاسبه نرخ انتقال V2R بر اساس قضیه شانون"""
    if distance <= 0.01:
        distance = 0.01
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

def calculate_task_priority(alpha, beta, T_stay, rho, d):
    """
    فرمول 23: محاسبه امتیاز اولویت مقاله:
    P = alpha * exp(-1/T_stay) + beta * (rho / d)
    """
    # جلوگیری از خطای ریاضی وقتی خودرو در حال خروج فوری است
    if T_stay <= 0.01:
        T_stay = 0.01 
        
    return alpha * math.exp(-1 / T_stay) + beta * (rho / d)

def sort_tasks_by_priority(tasks_queue, alpha=0.3, beta=0.7):
    """
    دریافت صف وظایف یک RSU و مرتب‌سازی نزولی آن‌ها بر اساس اولویت
    """
    for task in tasks_queue:
        task["priority"] = calculate_task_priority(
            alpha, 
            beta, 
            task.get("T_stay", 10.0),  # زمان توقف پیش‌فرض در صورت نبود دیتا
            task["rho"],             # حجم داده
            task["d"]                # مهلت مجاز
        )
        
    # مرتب‌سازی وظایف به صورتی که بالاترین اولویت در ابتدای لیست قرار بگیرد
    return sorted(tasks_queue, key=lambda t: t["priority"], reverse=True)