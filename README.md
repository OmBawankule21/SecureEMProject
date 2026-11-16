cat > README.md <<'EOF'
# SecureEMProject
## Privacy-Preserving Expectation Maximization using Secure Multiparty Computation

SecureEMProject is a research prototype that implements a privacy-preserving Expectation-Maximization (EM) algorithm using MP-SPDZ and Python.

The project demonstrates how multiple data holders can collaboratively compute global Gaussian Mixture Model (GMM) parameters without directly sharing their raw datasets.

## Key Features

- Centralized Expectation-Maximization baseline.
- Distributed EM across three simulated hospitals.
- Secure aggregation using MP-SPDZ replicated-ring.
- Fixed-point encoding and decoding.
- Numerical validation against centralized EM.
- Scalability and repeated benchmarking.
- Streamlit dashboard for visualization.
- Automated tests using pytest.

## Architecture

1. Each hospital maintains its local dataset.
2. Each hospital performs a local E-step.
3. Local sufficient statistics are calculated.
4. Statistics are encoded using fixed-point representation.
5. MP-SPDZ securely aggregates the statistics.
6. The aggregate is revealed for the global M-step.
7. Global GMM parameters are updated.
8. The process repeats until convergence.

## Technology Stack

- Python
- NumPy
- Pandas
- Streamlit
- MP-SPDZ
- pytest
- Matplotlib

## Project Structure

```text
SecureEMProject/
├── backend/
│   ├── app/
│   │   └── algorithms/
│   │       ├── em.py
│   │       ├── fixed_point.py
│   │       └── secure_em.py
│   ├── tests/
│   ├── dashboard.py
│   ├── run_secure_em_mpc.py
│   ├── benchmark_centralized.py
│   ├── run_scalability.py
│   └── validate_secure_em.py
├── results/
│   ├── scalability_runtime.png
│   ├── scalability_communication.png
│   ├── validation_metrics.csv
│   └── benchmark CSV files
├── requirements.txt
└── README.md


## MP-SPDZ Setup

This project uses MP-SPDZ for secure multiparty computation.

### 1. Clone MP-SPDZ

From the project root directory:

```bash
git clone https://github.com/data61/MP-SPDZ.git
```

### 2. Install dependencies

```bash
sudo apt update
sudo apt install -y git build-essential libgmp-dev libssl-dev
```

### 3. Configure and build MP-SPDZ

```bash
cd MP-SPDZ
```

Follow the official MP-SPDZ installation instructions:

https://github.com/data61/MP-SPDZ

The project uses the `replicated-ring` protocol with three parties.

### 4. Compile Secure EM aggregation

```bash
./compile.py -R 64 secure_em_aggregate
```

### 5. Return to the project root

```bash
cd ..
source .venv/bin/activate
```

### 6. Execute Secure EM

```bash
PYTHONPATH=backend python backend/run_secure_em_mpc.py 600
```

**Note:** The current implementation simulates three MPC parties on one machine. It is a research prototype, not an independent multi-hospital deployment.
