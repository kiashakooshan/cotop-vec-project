import math

def calculate_task_priority(alpha, beta, T_stay, rho, d):
    """
    محاسبه امتیاز اولویت بر اساس فرمول 23 مقاله:
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
            task.get("T_stay", 10),  # زمان توقف پیش‌فرض در صورت نبود دیتا
            task["rho"],             # حجم داده
            task["d"]                # مهلت مجاز
        )
        
    # مرتب‌سازی وظایف به صورتی که بالاترین اولویت در ابتدای لیست قرار بگیرد
    return sorted(tasks_queue, key=lambda t: t["priority"], reverse=True)