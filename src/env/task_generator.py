import random

class VehicleTaskScheduler:
    """
    مدیریت زمان‌بندی تولید وظایف برای هر ماشین به صورت کاملاً مستقل و دوره‌ای.
    """
    def __init__(self, veh_id, base_period=15.0, jitter=5.0):
        self.veh_id = veh_id
        self.base_period = base_period
        self.jitter = jitter
        # برای اینکه همه ماشین‌ها در ثانیه صفر با هم درخواست ندهند، زمان شروع را پخش می‌کنیم
        self.next_gen_time = random.uniform(0, base_period)

    def should_generate(self, current_time):
        """آیا زمان تولید یک DAG جدید برای این ماشین فرا رسیده است؟"""
        return current_time >= self.next_gen_time

    def schedule_next(self, current_time):
        """تنظیم تایمر برای وظیفه بعدی با کمی نویز تصادفی (Jitter)"""
        self.next_gen_time = current_time + random.uniform(
            self.base_period - self.jitter, self.base_period + self.jitter)


def generate_task_dag(veh_id, current_time, min_tasks=4, max_tasks=10):
    """
    تولید یک گراف جهت‌دار بدون دور (DAG) شامل چندین زیروظیفه وابسته.
    الگوریتم لایه‌ای تضمین می‌کند که هیچ حلقه (Cycle) در گراف به وجود نمی‌آید.
    """
    n_tasks = random.randint(min_tasks, max_tasks)
    n_layers = random.randint(2, max(2, n_tasks // 2))
    
    # پخش کردن وظایف در لایه‌های مختلف (حداقل 1 وظیفه در هر لایه)
    layer_sizes = [1] * n_layers
    for _ in range(n_tasks - n_layers):
        layer_sizes[random.randint(0, n_layers - 1)] += 1
        
    tasks = {}
    layers = []
    tid = 0
    
    # 1. ساخت گره‌ها (زیروظایف) در لایه‌ها
    for layer_idx, size in enumerate(layer_sizes):
        layer_tasks = []
        for _ in range(size):
            task_id = f"{veh_id}_t{tid}"
            tasks[task_id] = {
                "id": task_id,
                "veh_id": veh_id,
                "layer": layer_idx,
                "rho": random.uniform(1.0, 4.0),  # حجم داده (MB)
                "phi": random.uniform(0.5, 5.0),  # توان پردازشی مورد نیاز
                "d": random.uniform(5.0, 15.0),   # مهلت مجاز برای همین زیروظیفه
                "parents": [],
                "children": [],
                "in_edges": {} # حجم داده‌ای که باید از والد به این گره منتقل شود
            }
            layer_tasks.append(task_id)
            tid += 1
        layers.append(layer_tasks)
        
    # 2. اتصال لایه‌ها به یکدیگر (ایجاد یال‌ها و وابستگی‌ها)
    # هر گره در لایه‌های بعدی باید حداقل به یک گره در لایه قبلی متصل شود
    for i in range(len(layers) - 1):
        for child in layers[i + 1]:
            # انتخاب تصادفی 1 یا 2 والد از لایه قبلی
            n_parents = random.randint(1, min(2, len(layers[i])))
            parents = random.sample(layers[i], n_parents)
            
            for p in parents:
                tasks[p]["children"].append(child)
                tasks[child]["parents"].append(p)
                # حجم داده‌ای که پس از پردازش والد، باید به فرزند منتقل شود
                tasks[child]["in_edges"][p] = random.uniform(0.2, 1.0)
                
    return {
        "dag_id": f"{veh_id}_dag_{int(current_time)}",
        "veh_id": veh_id,
        "release_time": current_time,
        "tasks": tasks,
        "entry_tasks": layers[0],    # وظایف شروع (بدون والد)
        "exit_tasks": layers[-1],    # وظایف پایان (بدون فرزند)
    }