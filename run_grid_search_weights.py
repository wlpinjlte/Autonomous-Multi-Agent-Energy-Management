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
    
    # Tworzymy izolowane środowisko TYLKO dla tego procesu (nie nadpisuje globalnego env!)
    isolated_env = os.environ.copy()
    isolated_env.update(env_vars)
    
    # Zabezpieczenie przed oczekiwaniem na input z klawiatury (DEVNULL)
    subprocess.run(cmd, cwd=CWD, check=True, env=isolated_env, stdin=subprocess.DEVNULL)

def read_metrics():
    metrics_file = os.path.join(CWD, "data", "rl_vs_baseline_metrics.txt")
    with open(metrics_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    energy_match = re.search(r'Zysk \(SAC vs Occupancy\):\s*(-?\d+\.\d+)%', content)
    energy_saving = energy_match.group(1) if energy_match else "N/A"
    
    energy_abs_match = re.search(r'Skumulowany koszt energii.*?\n\s*SAC Agent:\s*(\d+\.\d+)', content, re.DOTALL)
    energy_abs = energy_abs_match.group(1) if energy_abs_match else "N/A"
    
    comfort_abs_match = re.search(r'Skumulowana kara za dyskomfort.*?\n\s*SAC Agent:\s*(\d+\.\d+)', content, re.DOTALL)
    comfort_abs = comfort_abs_match.group(1) if comfort_abs_match else "N/A"
    
    # Znalezienie procentowego "Zysk (SAC vs Occupancy)" dla komfortu
    comfort_pct_match = re.search(r'Skumulowana kara za dyskomfort.*?\n.*?\n.*?\n.*?\n\s*Zysk \(SAC vs Occupancy\):\s*(-?\d+\.\d+)%', content, re.DOTALL)
    comfort_pct = comfort_pct_match.group(1) if comfort_pct_match else "N/A"
    
    return energy_saving, energy_abs, comfort_abs, comfort_pct, content

def run_evaluations(candidates, mode_name):
    out_file_detailed = os.path.join(CWD, "data", "grid_search_results_detailed.txt")
    out_file_short = os.path.join(CWD, "data", "grid_search_results_short.txt")
    
    # Używamy dopisywania (a), by ewentualne przerwania nie kasowały poprzednich sesji
    with open(out_file_detailed, "a", encoding="utf-8") as f:
        f.write(f"\n--- Wyniki {mode_name} (Szczegółowe) ---\n")
    with open(out_file_short, "a", encoding="utf-8") as f:
        f.write(f"\n--- Wyniki {mode_name} (Skrócone) ---\n")
        
    print(f"--- ROZPOCZĘCIE {mode_name.upper()} ---")
    for w_eng, w_comf_now, w_fut in candidates:
        print(f"\n[Test] Wagi -> Energia: {w_eng:.2f}, Komfort_Now: {w_comf_now:.2f}, Komfort_Future: {w_fut:.2f}")
        
        env_vars = {
            "WEIGHT_ENERGY": f"{w_eng:.2f}",
            "WEIGHT_COMFORT_NOW": f"{w_comf_now:.2f}",
            "WEIGHT_COMFORT_FUTURE": f"{w_fut:.2f}"
        }
        
        print("Trwa trening wielowątkowy...")
        run_docker("decision_agent", env_vars)
        
        print("Trwa ewaluacja...")
        run_docker("evaluate_agent", env_vars)
        
        energy_saving, energy_abs, comfort_abs, comfort_pct, raw_content = read_metrics()
        
        header = f"- Wagi ({w_eng:.2f}, {w_comf_now:.2f}, {w_fut:.2f}) -> Energia: {energy_abs} jedn. (Zysk: {energy_saving}%), Komfort: {comfort_abs} jedn. ({comfort_pct}%)"
        indented_content = "\n".join([f"    {line}" for line in raw_content.strip().split("\n")])
        result_detailed = f"{header}\n{indented_content}\n"
        
        print(f"Wynik dla testu:\n{header}")
        
        # Zapisz wersję szczegółową
        with open(out_file_detailed, "a", encoding="utf-8") as f:
            f.write(result_detailed + "\n")
            
        # Zapisz wersję skróconą
        with open(out_file_short, "a", encoding="utf-8") as f:
            f.write(header + "\n")
    
    print(f"\n--- ZAKOŃCZONO {mode_name.upper()}. Wyniki zapisane w data/grid_search_results_*.txt ---")

def main():
    parser = argparse.ArgumentParser(description="Optymalizacja hiperparametrów agenta (Grid Search / Random Search)")
    subparsers = parser.add_subparsers(dest="mode", required=True)
    
    # Tryb Coarse
    parser_coarse = subparsers.add_parser("coarse", help="Gruboziarnisty grid search z wybranymi krokami dla poszczególnych wag")
    parser_coarse.add_argument("--step-eng", type=float, default=0.1, help="Krok poszukiwań dla wagi energii (np. 0.1)")
    parser_coarse.add_argument("--step-comf", type=float, default=0.1, help="Krok poszukiwań dla wagi komfortu bieżącego (np. 0.1)")
    
    # Tryb Random
    parser_random = subparsers.add_parser("random", help="Losowe przeszukiwanie w konkretnym zakresie okna optymalnego")
    parser_random.add_argument("--min-eng", type=float, required=True, help="Minimalna waga energii")
    parser_random.add_argument("--max-eng", type=float, required=True, help="Maksymalna waga energii")
    parser_random.add_argument("--min-comf", type=float, required=True, help="Minimalna waga bieżącego komfortu")
    parser_random.add_argument("--max-comf", type=float, required=True, help="Maksymalna waga bieżącego komfortu")
    parser_random.add_argument("--trials", type=int, default=10, help="Liczba losowań")
    
    args = parser.parse_args()
    
    candidates = []
    
    if args.mode == "coarse":
        steps_eng = np.arange(0.0, 1.01, args.step_eng)
        steps_comf = np.arange(0.0, 1.01, args.step_comf)
        for w_eng in steps_eng:
            for w_comf in steps_comf:
                if w_eng + w_comf <= 1.0:
                    w_fut = round(1.0 - w_eng - w_comf, 2)
                    candidates.append((round(w_eng, 2), round(w_comf, 2), w_fut))
        print(f"Wygenerowano {len(candidates)} konfiguracji dla Coarse Grid Search (krok eng: {args.step_eng}, krok comf: {args.step_comf}).")
        run_evaluations(candidates, "Coarse Grid Search")
        
    elif args.mode == "random":
        for _ in range(args.trials):
            w_eng = round(random.uniform(args.min_eng, args.max_eng), 2)
            w_comf = round(random.uniform(args.min_comf, args.max_comf), 2)
            w_fut = round(1.0 - w_eng - w_comf, 2)
            if w_fut >= 0:
                candidates.append((w_eng, w_comf, w_fut))
            else:
                # Bezpiecznik jeśli w_eng i w_comf przekroczą 1.0
                w_comf_adjusted = round(1.0 - w_eng, 2)
                candidates.append((w_eng, w_comf_adjusted, 0.0))
        
        print(f"Wylosowano {args.trials} konfiguracji w zadanym zakresie.")
        run_evaluations(candidates, "Random Search (Fine tuning)")

if __name__ == "__main__":
    main()
