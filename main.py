import time, os, psutil, math
from pathlib import Path
from typing import Tuple, Dict, List
import statistics as stats
import tsplib95
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.library import QFT
from qiskit_aer import AerSimulator
import warnings

N_RUNS = 64 # 반복 횟수

warnings.filterwarnings("ignore", category=DeprecationWarning)
os.system('cls')

TSP_FILE = "gr17.tsp"
OPTIMAL_LENGTH = 2085

def current_rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024*1024)


# Held-Karp: 동적 프로그래밍

def held_karp(problem):
    nodes = sorted(problem.get_nodes())
    n = len(nodes)
    start = nodes[0]

    INF = 10**18
    DP = [[INF] * n for _ in range(1 << n)]
    PREV = [[-1] * n for _ in range(1 << n)]
    start_i = 0
    DP[1 << start_i][start_i] = 0

    for mask in range(1 << n):
        if not (mask & (1 << start_i)):
            continue
        for j in range(n):
            if not (mask & (1 << j)):
                continue
            cj = DP[mask][j]
            if cj >= INF:
                continue
            jnode = nodes[j]
            for k in range(n):
                if mask & (1 << k):
                    continue
                knode = nodes[k]
                new_mask = mask | (1 << k)
                new_cost = cj + problem.get_weight(jnode, knode)
                if new_cost < DP[new_mask][k]:
                    DP[new_mask][k] = new_cost
                    PREV[new_mask][k] = j

    full = (1 << n) - 1
    best_cost = INF
    last = -1
    for j in range(n):
        if DP[full][j] >= INF:
            continue
        tour_cost = DP[full][j] + problem.get_weight(nodes[j], start)
        if tour_cost < best_cost:
            best_cost = tour_cost
            last = j

    path_idx = [last]
    mask = full
    while path_idx[-1] != start_i:
        j = path_idx[-1]
        pj = PREV[mask][j]
        mask ^= (1 << j)
        path_idx.append(pj)
    path_idx.reverse()

    tour_nodes = [nodes[i] for i in path_idx] + [start]
    return best_cost, tour_nodes


def run_heldkarp_once() -> Dict[str, float]:
    if not Path(TSP_FILE).exists():
        raise FileNotFoundError(f"'{TSP_FILE}' 파일이 없습니다.")
    problem = tsplib95.load_problem(TSP_FILE)
    mem_before = current_rss_mb()
    t0 = time.perf_counter()
    best_len, _ = held_karp(problem)
    elapsed = time.perf_counter() - t0
    mem_after = current_rss_mb()
    error_rate = ((best_len - OPTIMAL_LENGTH) / OPTIMAL_LENGTH) * 100
    return {"time": elapsed, "mem": mem_after - mem_before, "err": error_rate}

# Quantum Phase Estimation: 양자 위상 추정

def build_qpe_circuit(theta: float, m: int) -> Tuple[QuantumCircuit, QuantumRegister, ClassicalRegister]:
    q_count = QuantumRegister(m, "count")
    q_target = QuantumRegister(1, "target")
    c_count = ClassicalRegister(m, "c")
    qc = QuantumCircuit(q_count, q_target, c_count)

    qc.x(q_target[0])

    for j in range(m):
        qc.h(q_count[j])

    phi = 2 * math.pi * theta
    for j in range(m):
        power = 1 << (m - 1 - j)
        qc.cp(phi * power, q_count[j], q_target[0])

    iqft = QFT(num_qubits=m, approximation_degree=0, do_swaps=True).inverse()
    qc.append(iqft, q_count[:])

    for j in range(m):
        qc.measure(q_count[j], c_count[m-1-j])

    return qc, q_count, c_count

def phase_from_counts(counts: dict, m: int) -> float:
    bitstring = max(counts.items(), key=lambda kv: kv[1])[0]
    return int(bitstring, 2) / (2 ** m)

def circular_error(true_theta: float, est_theta: float) -> float:
    delta = abs(est_theta - true_theta) % 1.0
    return min(delta, 1.0 - delta)

def run_qpe_once(theta: float = 5/16, m: int = 6, shots: int = 4096) -> Dict[str, float]:
    sim = AerSimulator()
    qc, q_count, c_count = build_qpe_circuit(theta, m)
    mem_before = current_rss_mb()
    t0 = time.perf_counter()
    tqc = transpile(qc, sim)
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()
    elapsed = time.perf_counter() - t0
    mem_after = current_rss_mb()
    est_theta = phase_from_counts(counts, m)
    err_pct = circular_error(theta, est_theta) * 100.0
    return {"time": elapsed, "mem": mem_after - mem_before, "err": err_pct}

# Grover search: 그로버 탐색

def apply_phase_oracle(qc: QuantumCircuit, q, marked: str):
    n = len(marked)
    for i, b in enumerate(marked):
        if b == '0':
            qc.x(q[i])
    qc.h(q[n-1])
    qc.mcx(list(range(n-1)), q[n-1])
    qc.h(q[n-1])
    for i, b in enumerate(marked):
        if b == '0':
            qc.x(q[i])


def apply_diffuser(qc: QuantumCircuit, q):
    n = len(q)
    qc.h(q)
    qc.x(q)
    qc.h(q[n-1])
    qc.mcx(list(range(n-1)), q[n-1])
    qc.h(q[n-1])
    qc.x(q)
    qc.h(q)


def build_grover_circuit(n: int, marked: str, iterations: int) -> QuantumCircuit:
    assert len(marked) == n, "marked 문자열 길이는 n과 같아야 합니다."
    q = QuantumRegister(n, "q")
    c = ClassicalRegister(n, "c")
    qc = QuantumCircuit(q, c)
    qc.h(q)
    for _ in range(iterations):
        apply_phase_oracle(qc, q, marked)
        apply_diffuser(qc, q)
    for i in range(n):
        qc.measure(q[i], c[n-1-i])
    return qc


def recommended_iterations(n: int, k: int = 1) -> int:
    N = 2**n
    return max(1, int(math.floor(math.pi/4 * math.sqrt(N / k))))


def run_grover_once(n: int = 6, marked: str = "101011", shots: int = 2048) -> Dict[str, float]:
    iters = recommended_iterations(n, k=1)
    qc = build_grover_circuit(n, marked, iters)
    sim = AerSimulator()
    mem_before = current_rss_mb()
    t0 = time.perf_counter()
    tqc = transpile(qc, sim)
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()
    elapsed = time.perf_counter() - t0
    mem_after = current_rss_mb()
    success_prob = counts.get(marked, 0) / shots
    error_rate = (1.0 - success_prob) * 100.0
    return {"time": elapsed, "mem": mem_after - mem_before, "err": error_rate}


# 결과 출력

def summarize(name: str, records: List[Dict[str, float]]):
    times = [r["time"] for r in records]
    mems  = [r["mem"]  for r in records]
    # errs  = [r["err"]  for r in records if "err" in r]
    t_avg, t_std = stats.mean(times), (stats.pstdev(times) if len(times) > 1 else 0.0)
    m_avg, m_std = stats.mean(mems),  (stats.pstdev(mems)  if len(mems)  > 1 else 0.0)
    # e_avg, e_std = (stats.mean(errs), (stats.pstdev(errs) if len(errs) > 1 else 0.0)) if errs else (None, None)

    print(f"\n=== {name} (N={len(records)}) ===")
    print(f"Avg Exec Time: {t_avg:.4f} sec (±{t_std:.4f})")
    print(f"Avg Mem Usage: {m_avg:.4f} MB (±{m_std:.4f})")
    # if e_avg is not None:
    #     print(f"Avg Error Rate  : {e_avg:.4f}% (±{e_std:.4f})")

def main_all():
    hk_records = [run_heldkarp_once() for _ in range(N_RUNS)]
    summarize("Held–Karp", hk_records)

    qpe_records = [run_qpe_once(theta=5/16, m=6, shots=2048) for _ in range(N_RUNS)]
    summarize("Quantum Phase Estimation", qpe_records)

    grover_records = [run_grover_once(n=6, marked="101011", shots=2048) for _ in range(N_RUNS)]
    summarize("Grover Search", grover_records)

if __name__ == "__main__":
    main_all()
