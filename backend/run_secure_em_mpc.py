import csv
from datetime import datetime, timezone
import re
import subprocess
import time
import json
import resource
from pathlib import Path
import numpy as np

from app.algorithms.em import e_step
from app.algorithms.secure_em import (
    compute_local_statistics,
    update_global_parameters,
)
from app.algorithms.fixed_point import (
    encode_statistics,
    decode_statistics,
)

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MP_SPDZ = PROJECT_ROOT / "MP-SPDZ"

PROGRAM = "secure_em_aggregate"
N_PARTIES = 3
N_COMPONENTS = 3
N_FEATURES = 4
N_SAMPLES_PER_PARTY = 200

MAX_ITERATIONS = 20
SCALE = 1_000_000

INPUT_FILES = [
    MP_SPDZ / "Player-Data" / f"Input-P{i}-0"
    for i in range(N_PARTIES)
]
MPC_RUN_METRICS = []

# --------------------------------------------------
# 1. GENERATE SYNTHETIC HOSPITAL DATA
# --------------------------------------------------


def generate_hospital_data(total_samples=600):

    if total_samples % 3 != 0:
        raise ValueError("Total samples must be divisible by 3")

    rng = np.random.default_rng(42)

    centers = np.array([
        [-4.0, -4.0,  0.0,  0.0],
        [ 0.0,  4.0,  3.0, -2.0],
        [ 4.0, -2.0, -3.0,  3.0],
    ])

    samples_per_cluster = total_samples // 3

    all_data = []

    for center in centers:
        samples = rng.normal(
            loc=center,
            scale=1.2,
            size=(samples_per_cluster, N_FEATURES)
        )
        all_data.append(samples)

    X = np.vstack(all_data)
    rng.shuffle(X)

    return np.array_split(X, N_PARTIES)



# --------------------------------------------------
# 2. WRITE PRIVATE INPUT FILES
# --------------------------------------------------

def write_private_inputs(encoded_parties):

    for party_id, encoded in enumerate(encoded_parties):

        if len(encoded) != 63:
            raise ValueError(
                f"Party {party_id} has {len(encoded)} values, expected 63"
            )

        INPUT_FILES[party_id].write_text(
            " ".join(str(int(value)) for value in encoded)
        )

    print("Private input files generated successfully.")


# --------------------------------------------------
# 3. EXECUTE THREE MPC PARTIES
# --------------------------------------------------


def run_mpc():

    processes = []
    start_time = time.perf_counter()

    try:
        for party_id in range(N_PARTIES):

            command = [
                "./replicated-ring-party.x",
                str(party_id),
                PROGRAM,
            ]

            process = subprocess.Popen(
                command,
                cwd=MP_SPDZ,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )

            processes.append(process)

        outputs = []

        for party_id, process in enumerate(processes):

            output, _ = process.communicate(timeout=120)

            if process.returncode != 0:
                raise RuntimeError(
                    f"Party {party_id} failed:\n{output}"
                )

            outputs.append(output)

            log_path = (
                PROJECT_ROOT / "results" /
                f"mpc_party_{party_id}.log"
            )
            log_path.write_text(output)

        mpc_runtime = time.perf_counter() - start_time

        # Extract communication from Party 0's log.
        communication_match = re.search(
            r"Global data sent = ([0-9.]+) MB",
            outputs[0]
        )

        communication_mb = (
            float(communication_match.group(1))
            if communication_match else None
        )

        party_time_match = re.search(
            r"Time = ([0-9.]+) seconds",
            outputs[0]
        )

        party_runtime = (
            float(party_time_match.group(1))
            if party_time_match else None
        )

        metrics = {
            "mpc_wall_runtime_seconds": mpc_runtime,
            "party0_runtime_seconds": party_runtime,
            "global_communication_mb": communication_mb,
        }

        MPC_RUN_METRICS.append(metrics)

        print("All three MPC parties completed.")
        print("MPC wall runtime:", mpc_runtime)
        print("Global communication:", communication_mb, "MB")

        return outputs[0]

    except Exception:
        for process in processes:
            if process.poll() is None:
                process.kill()
        raise


    except Exception:
        for process in processes:
            if process.poll() is None:
                process.kill()
        raise


# --------------------------------------------------
# 4. EXTRACT SECURE AGGREGATES
# --------------------------------------------------

def parse_aggregates(output):

    results = {}

    for line in output.splitlines():

        match = re.search(
            r"AGG\[(\d+)\]\s*=\s*(-?\d+)",
            line
        )

        if match:
            index = int(match.group(1))
            value = int(match.group(2))
            results[index] = value

    if len(results) != 63:
        raise RuntimeError(
            f"Expected 63 aggregates, received {len(results)}"
        )

    return np.array(
        [results[i] for i in range(63)],
        dtype=np.int64
    )


# --------------------------------------------------
# 5. SECURE EM ITERATIONS
# --------------------------------------------------

def main(total_samples=600):
    MPC_RUN_METRICS.clear()

    overall_start = time.perf_counter()

    print("\n========== SECURE EM STARTED ==========")

    parties = generate_hospital_data(total_samples)

    print("Hospitals:", len(parties))
    print("Samples per hospital:",
          [len(party) for party in parties])

    # Shared public initialization.
    weights = np.full(N_COMPONENTS, 1.0 / N_COMPONENTS)

    rng = np.random.default_rng(100)

    public_centers = np.array([
        [-4.0, -4.0, 0.0, 0.0],
        [0.0, 4.0, 3.0, -2.0],
        [4.0, -2.0, -3.0, 3.0],
    ])

    means = public_centers + rng.normal(
        0, 0.5, public_centers.shape
    )

    covariances = np.repeat(
        (2.0 * np.eye(N_FEATURES))[None, :, :],
        N_COMPONENTS,
        axis=0
    )

    for iteration in range(1, MAX_ITERATIONS + 1):
        algorithm_start = time.perf_counter()
        print(f"\n--- Secure EM iteration {iteration} ---")

        encoded_parties = []

        # Local E-step and local statistics.
        for party_id, X_local in enumerate(parties):

            responsibilities, local_ll = e_step(
                X_local,
                weights,
                means,
                covariances
            )

            statistics = compute_local_statistics(
                X_local,
                responsibilities
            )

            encoded = encode_statistics(
                *statistics,
                scale=SCALE
            )

            encoded_parties.append(encoded)

        # Fixed-point private input generation.
        write_private_inputs(encoded_parties)

        # Actual three-party MPC execution.
        output = run_mpc()

        # Decode revealed aggregate.
        aggregated = parse_aggregates(output)

        counts_shape = (N_COMPONENTS,)
        sums_shape = (N_COMPONENTS, N_FEATURES)
        second_shape = (
            N_COMPONENTS,
            N_FEATURES,
            N_FEATURES
        )

        counts, sums, second_moments = decode_statistics(
            aggregated,
            counts_shape,
            sums_shape,
            second_shape,
            scale=SCALE
        )

        # Global M-step.
        new_weights, new_means, new_covariances = (
            update_global_parameters(
                counts,
                sums,
                second_moments
            )
        )

        parameter_change = max(
            np.max(np.abs(new_weights - weights)),
            np.max(np.abs(new_means - means)),
            np.max(np.abs(new_covariances - covariances)),
        )

        weights = new_weights
        means = new_means
        covariances = new_covariances

        print("Secure aggregation: SUCCESS")
        print("Parameter change:", parameter_change)
        print("Mixture weights:", weights)

        if parameter_change < 1e-4:
            print("Convergence reached.")
            break

    print("\n========== SECURE EM COMPLETED ==========")
    algorithm_runtime = time.perf_counter() - algorithm_start
    print("Secure EM algorithm runtime:", algorithm_runtime)
    print("Iterations:", iteration)
    print("Final means:\n", means)
    print("Final covariance matrices:\n", covariances)



    np.savez(
        PROJECT_ROOT / "results" / "secure_em_results.npz",
        weights=weights,
        means=means,
        covariances=covariances,
        iterations=iteration,
    )

    print("Results saved successfully.")
    total_runtime = time.perf_counter() - overall_start

    peak_memory_mb = resource.getrusage(
        resource.RUSAGE_SELF
    ).ru_maxrss / 1024

    metrics = {
        "dataset_samples": total_samples,
        "hospitals": N_PARTIES,
        "iterations": iteration,
        "total_runtime_seconds": total_runtime,
        "algorithm_runtime_seconds": algorithm_runtime,
        "peak_memory_mb": peak_memory_mb,
        "mpc_iterations": MPC_RUN_METRICS,
        "total_mpc_wall_runtime_seconds": sum(
            m["mpc_wall_runtime_seconds"]
            for m in MPC_RUN_METRICS
        ),
        "total_global_communication_mb": (
            sum(
                m["global_communication_mb"]
                for m in MPC_RUN_METRICS
                if m["global_communication_mb"] is not None
            )
            if all(
                m["global_communication_mb"] is not None
                for m in MPC_RUN_METRICS
            )
            else None
        ),
    }

    metrics_path = (
        PROJECT_ROOT / "results" / "secure_em_metrics.json"
    )

    metrics_path.write_text(
        json.dumps(metrics, indent=4)
    )


    # --------------------------------------------------
    # SAVE PERMANENT EXPERIMENT HISTORY
    # --------------------------------------------------

    history_path = (
        PROJECT_ROOT / "results" / "experiment_history.csv"
    )

    history_fields = [
        "timestamp",
        "dataset_samples",
        "hospitals",
        "iterations",
        "total_runtime_seconds",
        "peak_memory_mb",
        "total_mpc_wall_runtime_seconds",
        "total_global_communication_mb",
    ]

    history_record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset_samples": metrics["dataset_samples"],
        "hospitals": metrics["hospitals"],
        "iterations": metrics["iterations"],
        "total_runtime_seconds": metrics["total_runtime_seconds"],
        "peak_memory_mb": metrics["peak_memory_mb"],
        "total_mpc_wall_runtime_seconds": metrics[
            "total_mpc_wall_runtime_seconds"
        ],
        "total_global_communication_mb": metrics[
            "total_global_communication_mb"
        ],
    }

    file_exists = history_path.exists()

    with open(history_path, "a", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=history_fields
        )

        if not file_exists or history_path.stat().st_size == 0:
            writer.writeheader()

        writer.writerow(history_record)

    print("Experiment history saved:", history_path)


    print("\n========== EXECUTION METRICS ==========")
    print(json.dumps(metrics, indent=4))
    print("Metrics saved:", metrics_path)

if __name__ == "__main__":
    import sys
    size = int(sys.argv[1]) if len(sys.argv) > 1 else 600
    main(size)
