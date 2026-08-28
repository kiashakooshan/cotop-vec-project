import random

def generate_random_task(vehicle_id, current_time):
    """
    تولید یک وظیفه محاسباتی تصادفی برای خودرو بر اساس پارامترهای مقاله
    """
    task = {
        "veh_id": vehicle_id,
        "time": current_time,
        "rho": random.uniform(2, 5),      # حجم داده بین 2 تا 5 مگابایت
        "d": random.uniform(20, 30),      # مهلت مجاز بین 20 تا 30 ثانیه
        "phi": random.uniform(1, 10)      # منابع پردازشی مورد نیاز (Mcycles)
    }
    return task