import subprocess
import sys

# فقط Seed 0 را اجرا می‌کنیم تا سرعت ۳ برابر شود
SEEDS = [0]
SCRIPTS = ["rl/train.py", "rl/evaluate_cotop.py", "rl/ddqn_agent.py",
           "rl/qrmp_dqn_agent.py", "rl/greedy_agent.py", "rl/local_agent.py"]

for seed in SEEDS:
    print(f"\n{'='*20} SEED {seed} {'='*20}")
    for script in SCRIPTS:
        print(f"--- Running {script} (seed={seed}) ---")
        subprocess.run([sys.executable, script, str(seed)], check=True)

print("\nAll seeds finished. Run evaluate.py now to see aggregated results.")