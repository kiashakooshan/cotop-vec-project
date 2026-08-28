import matplotlib.pyplot as plt
import numpy as np

def draw_evaluation_charts():
    # داده‌هایی که از اجرای سه الگوریتم به دست آوردیم
    # توجه: پاداش‌ها منفی هستند (چون مجموع جریمه تاخیر و انرژی است)، 
    # برای نمایش بهتر در نمودار، ما قدر مطلق (مقدار مثبت) آن‌ها را به عنوان "هزینه کل" نشان می‌دهیم.
    # هرچه این ستون کوتاه‌تر باشد، یعنی الگوریتم بهتر کار کرده است.
    
    algorithms = ['CoTOP (A3C)', 'Local (No Collab)', 'Greedy']
    # تبدیل پاداش‌های منفی به هزینه (Cost)
    costs = [264.24, 342.32, 358.43] 
    
    # تنظیمات استایل نمودار
    plt.figure(figsize=(8, 6))
    plt.style.use('ggplot') # استفاده از استایل زیبا
    
    # رسم نمودار میله‌ای
    bars = plt.bar(algorithms, costs, color=['#2ca02c', '#1f77b4', '#d62728'], width=0.5)
    
    # اضافه کردن جزئیات متنی
    plt.title('Total System Cost Comparison (Delay + Energy)', fontsize=14, fontweight='bold')
    plt.ylabel('Total Cost (Lower is Better)', fontsize=12)
    plt.xlabel('Offloading Strategy', fontsize=12)
    
    # نوشتن اعداد دقیق روی هر ستون
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 5, f'{yval}', ha='center', va='bottom', fontweight='bold')
        
    # نمایش نمودار
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("🚀 Generating Evaluation Plots...")
    draw_evaluation_charts()