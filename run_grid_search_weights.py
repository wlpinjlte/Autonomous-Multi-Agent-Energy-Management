import os
import re
import subprocess
import argparse
import random
import numpy as np

CWD = r"f:\projekty\Autonomous-Multi-Agent-Energy-Management"

def run_docker(service, env_vars):
    cmd = ["docker", "compose", "-f", f"{service}/docker-compose.yml", "up", "--build", "--abort-on-container-exit"]
    print(f"Running {' '.join(cmd)}...")
    
    # Create an isolated environment ONLY for this process (doesn't override global env!)
    isolated_env = os.environ.copy()
    isolated_env.update(env_vars)
    
    # Protection against waiting for keyboard input (DEVNULL)
    subprocess.run(cmd, cwd=CWD, check=True, env=isolated_env, stdin=subprocess.DEVNULL)

def read_metrics():
    metrics_file = os.path.join(CWD, "data", "rl_vs_baseline_metrics.txt")
    with open(metrics_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    energy_match = re.search(r'Gain \(SAC vs Occupancy\):\s*(-?\d+\.\d+)%', content)
    energy_saving = energy_match.group(1) if energy_match else "N/A"
    
    energy_abs_match = re.search(r'Cumulative energy cost.*?\n\s*SAC Agent:\s*(\d+\.\d+)', content, re.DOTALL)
    energy_abs = energy_abs_match.group(1) if energy_abs_match else "N/A"
    
    comfort_abs_match = re.search(r'Cumulative thermal discomfort penalty.*?\n\s*SAC Agent:\s*(\d+\.\d+)', content, re.DOTALL)
    comfort_abs = comfort_abs_match.group(1) if comfort_abs_match else "N/A"
    
    # Znalezienie procentowego zysku dla komfortu
    comfort_pct_match = re.search(r'Cumulative thermal discomfort penalty.*?\n.*?\n.*?\n.*?\n\s*Gain \(SAC vs Occupancy\):\s*(-?\d+\.\d+)%', content, re.DOTALL)
    comfort_pct = comfort_pct_match.group(1) if comfort_pct_match else "N/A"
    
    return energy_saving, energy_abs, comfort_abs, comfort_pct, content

def run_evaluations(candidates, mode_name, file_prefix):
    out_file_detailed = os.path.join(CWD, "data", f"grid_search_{file_prefix}_results_detailed.txt")
    out_file_short = os.path.join(CWD, "data", f"grid_search_{file_prefix}_results_short.txt")
    
    # Use append mode (a) so interruptions don't delete previous sessions
    with open(out_file_detailed, "a", encoding="utf-8") as f:
        f.write(f"\n--- {mode_name} Results (Detailed) ---\n")
    with open(out_file_short, "a", encoding="utf-8") as f:
        f.write(f"\n--- {mode_name} Results (Short) ---\n")
        
    print(f"--- STARTING {mode_name.upper()} ---")
    for w_eng, w_comf_now, w_fut in candidates:
        print(f"\n[Test] Weights -> Energy: {w_eng:.2f}, Comfort_Now: {w_comf_now:.2f}, Comfort_Future: {w_fut:.2f}")
        
        env_vars = {
            "WEIGHT_ENERGY": f"{w_eng:.2f}",
            "WEIGHT_COMFORT_NOW": f"{w_comf_now:.2f}",
            "WEIGHT_COMFORT_FUTURE": f"{w_fut:.2f}"
        }
        
        print("Running multi-agent training...")
        run_docker("decision_agent", env_vars)
        
        print("Running evaluation...")
        run_docker("evaluate_agent", env_vars)
        
        energy_saving, energy_abs, comfort_abs, comfort_pct, raw_content = read_metrics()
        
        header = f"- Weights ({w_eng:.2f}, {w_comf_now:.2f}, {w_fut:.2f}) -> Energy: {energy_abs} units (Gain: {energy_saving}%), Comfort: {comfort_abs} units ({comfort_pct}%)"
        indented_content = "\n".join([f"    {line}" for line in raw_content.strip().split("\n")])
        result_detailed = f"{header}\n{indented_content}\n"
        
        print(f"Result for test:\n{header}")
        
        # Save detailed version
        with open(out_file_detailed, "a", encoding="utf-8") as f:
            f.write(result_detailed + "\n")
            
        # Save short version
        with open(out_file_short, "a", encoding="utf-8") as f:
            f.write(header + "\n")
    
    print(f"\n--- {mode_name.upper()} FINISHED. Results saved in data/grid_search_{file_prefix}_results_*.txt ---")

def main():
    parser = argparse.ArgumentParser(description="Agent hyperparameters optimization (Grid Search / Random Search)")
    subparsers = parser.add_subparsers(dest="mode", required=True)
    
    # Coarse mode
    parser_coarse = subparsers.add_parser("coarse", help="Coarse grid search with specific steps for each weight")
    parser_coarse.add_argument("--min-eng", type=float, default=0.0, help="Minimum energy weight")
    parser_coarse.add_argument("--max-eng", type=float, default=1.0, help="Maximum energy weight")
    parser_coarse.add_argument("--step-eng", type=float, default=0.1, help="Search step for energy weight (e.g. 0.1)")
    parser_coarse.add_argument("--min-comf", type=float, default=0.0, help="Minimum current comfort weight")
    parser_coarse.add_argument("--max-comf", type=float, default=1.0, help="Maximum current comfort weight")
    parser_coarse.add_argument("--step-comf", type=float, default=0.1, help="Search step for current comfort weight (e.g. 0.1)")
    
    # Random mode
    parser_random = subparsers.add_parser("random", help="Random search in a specific range of the optimal window")
    parser_random.add_argument("--min-eng", type=float, required=True, help="Minimum energy weight")
    parser_random.add_argument("--max-eng", type=float, required=True, help="Maximum energy weight")
    parser_random.add_argument("--min-comf", type=float, required=True, help="Minimum current comfort weight")
    parser_random.add_argument("--max-comf", type=float, required=True, help="Maximum current comfort weight")
    parser_random.add_argument("--trials", type=int, default=10, help="Number of trials")
    
    args = parser.parse_args()
    
    candidates = []
    
    if args.mode == "coarse":
        steps_eng = np.arange(args.min_eng, args.max_eng + 1e-5, args.step_eng)
        steps_comf = np.arange(args.min_comf, args.max_comf + 1e-5, args.step_comf)
        for w_eng in steps_eng:
            for w_comf in steps_comf:
                if w_eng + w_comf <= 1.0:
                    w_fut = round(1.0 - w_eng - w_comf, 2)
                    candidates.append((round(w_eng, 2), round(w_comf, 2), w_fut))
        print(f"Generated {len(candidates)} configurations for Coarse Grid Search (eng step: {args.step_eng}, comf step: {args.step_comf}).")
        run_evaluations(candidates, "Coarse Grid Search", "coarse")
        
    elif args.mode == "random":
        while len(candidates) < args.trials:
            w_eng = round(random.uniform(args.min_eng, args.max_eng), 2)
            w_comf = round(random.uniform(args.min_comf, args.max_comf), 2)
            w_fut = round(1.0 - w_eng - w_comf, 2)
            if w_fut > 0.0:
                candidates.append((w_eng, w_comf, w_fut))
        
        print(f"Sampled {args.trials} configurations in the specified range.")
        run_evaluations(candidates, "Random Search (Fine tuning)", "random")

if __name__ == "__main__":
    main()
