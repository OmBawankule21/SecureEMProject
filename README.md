# SecureEMProject
## Privacy-Preserving Expectation Maximization Using Secure Multiparty Computation

**A Research Prototype for Privacy-Preserving Gaussian Mixture Model Learning**

SecureEMProject implements a privacy-preserving Expectation-Maximization (EM) algorithm using Python and MP-SPDZ. It demonstrates how multiple data holders can collaboratively estimate global Gaussian Mixture Model (GMM) parameters without directly sharing their raw datasets.

The project compares a centralized EM baseline with a distributed Secure EM implementation using secure multiparty computation (MPC).

**Live Dashboard:** https://secureemproject-htoeqwrap2fediaz8esczg.streamlit.app/

---

## 1. Research Problem

Healthcare institutions and other data holders may need to collaboratively analyze distributed datasets. However, directly sharing sensitive records can introduce privacy and confidentiality risks.

This project explores an alternative approach: performing distributed statistical computation using secure multiparty computation.

The objective is to implement and evaluate a privacy-preserving version of the Expectation-Maximization algorithm for Gaussian Mixture Models.

### Research Objectives

- Implement a centralized EM baseline.
- Distribute data across three simulated parties.
- Compute local sufficient statistics at each party.
- Securely aggregate statistics using MP-SPDZ.
- Update global GMM parameters using aggregated statistics.
- Evaluate numerical agreement with centralized EM.
- Measure runtime and communication overhead.
- Provide a reproducible experimental dashboard.

---

## 2. Proposed Methodology

The implementation follows an iterative distributed EM workflow.

1. Each party maintains its own local dataset.
2. Each party performs a local E-step using the current global GMM parameters.
3. Local sufficient statistics are calculated.
4. Statistics are converted into fixed-point integer representations.
5. Parties provide private inputs to the MPC protocol.
6. MP-SPDZ securely aggregates the statistics.
7. The aggregated statistics are revealed for the global M-step.
8. Global mixture weights, means and covariance matrices are updated.
9. The process repeats until convergence or the maximum iteration limit is reached.

### Architecture

```text
       Party 1          Party 2          Party 3
          |                |                |
          v                v                v
       Local E-step     Local E-step     Local E-step
          |                |                |
          v                v                v
     Local Statistics  Local Statistics  Local Statistics
          |                |                |
          v                v                v
      Fixed-point       Fixed-point       Fixed-point
       Encoding          Encoding          Encoding
          |                |                |
          +----------------+----------------+
                           |
                           v
                  MP-SPDZ MPC Protocol
                    replicated-ring
                           |
                           v
                 Secure Aggregation
                           |
                           v
                Aggregate Statistics
                           |
                           v
                    Global M-step
                           |
                           v
               Updated GMM Parameters
                           |
                           v
                      Convergence
```

### Sufficient Statistics

For each Gaussian component, the implementation aggregates:

- Component responsibility counts.
- Weighted feature sums.
- Weighted second-moment matrices.

For three Gaussian components and four features, the total number of aggregated statistics is:

- Counts: 3
- First-order sums: 3 x 4 = 12
- Second-order moments: 3 x 4 x 4 = 48

**Total: 63 sufficient statistics.**

---

## 3. Technology Stack

| Technology | Purpose |
|---|---|
| Python | Main implementation |
| NumPy | Numerical computation |
| Pandas | Dataset and experiment handling |
| MP-SPDZ | Secure multiparty computation |
| Streamlit | Interactive research dashboard |
| Matplotlib | Experiment visualization |
| Scikit-learn | ML utilities |
| Pytest | Automated testing |

---

## 4. Experimental Results

The following results are from the recorded experiments in this repository.

### 4.1 Repeated Runtime Benchmark

Five trials were conducted for each dataset size.

Mean runtime in seconds:

| Samples | Centralized EM | Secure EM | Slowdown |
|---:|---:|---:|---:|
| 600 | 0.001921 | 0.258690 | 134.7x |
| 1,200 | 0.002856 | 0.268802 | 94.1x |
| 2,400 | 0.005037 | 0.268967 | 53.4x |

Both implementations converged in an average of three iterations for all tested dataset sizes.

The slowdown is expected in this prototype because Secure EM includes fixed-point processing, private input handling and MPC execution.

These measurements represent a single-machine experimental environment, not a production multi-hospital deployment.

### 4.2 Communication Overhead

| Dataset Size | Secure EM Communication |
|---:|---:|
| 600 | 0.009072 MB |
| 1,200 | 0.009072 MB |
| 2,400 | 0.009072 MB |

The reported communication is based on the project's recorded MPC communication metric. It should not be interpreted as a measurement of total network traffic across independent physical hospitals.

### 4.3 Numerical Validation

The centralized and Secure EM implementations were compared using their resulting GMM parameters.

| Parameter | Maximum Absolute Error |
|---|---:|
| Mixture Weights | 5.438 x 10^-10 |
| Means | 1.284 x 10^-8 |
| Covariance Matrices | 4.588 x 10^-8 |

Both implementations converged in three iterations in the recorded validation experiment.

The small numerical differences indicate close agreement between the centralized and secure implementations under the tested configuration.

### 4.4 Corrected Secure EM Execution Example

Recorded run with 600 samples and three simulated parties:

| Metric | Result |
|---|---:|
| Dataset samples | 600 |
| Simulated parties | 3 |
| EM iterations | 3 |
| Algorithm runtime | 0.34853 seconds |
| Total runtime | 0.42198 seconds |
| Total MPC wall runtime | 0.32639 seconds |
| Total communication | 0.009072 MB |
| Peak Python memory | 35.28 MB |

Algorithm runtime includes local computation, fixed-point encoding, MPC execution, decoding and global parameter updates. Dataset generation is excluded. Total runtime includes dataset generation.

### 4.5 Result Interpretation

The experiments demonstrate:

- Numerical agreement between centralized and Secure EM.
- Successful secure aggregation of 63 sufficient statistics.
- Convergence in the tested configurations.
- Measurable computational overhead introduced by MPC.
- Nearly constant communication volume in the tested experiments.

The results are experimental observations, not a claim of production-scale performance.

---

## 5. Project Structure

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
│   ├── repeated_benchmark.py
│   ├── repeated_comparison.py
│   └── validate_secure_em.py
├── data/
├── docs/
├── results/
│   ├── experiment_history.csv
│   ├── repeated_comparison_runs.csv
│   ├── repeated_comparison_summary.csv
│   ├── scalability_runtime.png
│   ├── scalability_communication.png
│   └── validation_metrics.csv
├── requirements.txt
└── README.md
```

Generated logs, temporary MPC files and large intermediate result files may not be included in the Git repository.

---

## 6. Installation and Environment Setup

### Prerequisites

- Ubuntu Linux or Ubuntu through WSL2.
- Python 3.10 or a compatible version supported by the dependencies.
- Git.
- C/C++ build tools.
- MP-SPDZ dependencies.

### Step 1: Clone this repository

```bash
git clone https://github.com/OmBawankule21/SecureEMProject.git

cd SecureEMProject
```

### Step 2: Create a Python virtual environment

```bash
python3 -m venv .venv

source .venv/bin/activate
```

### Step 3: Install Python dependencies

```bash
python -m pip install --upgrade pip

pip install -r requirements.txt
```

---

## 7. MP-SPDZ Installation and Configuration

This project uses MP-SPDZ with the `replicated-ring` protocol and a 64-bit computation ring.

### Step 1: Install system dependencies

```bash
sudo apt update

sudo apt install -y \
    git \
    build-essential \
    libgmp-dev \
    libssl-dev \
    libboost-dev
```

### Step 2: Clone MP-SPDZ

Run this command from the SecureEMProject root:

```bash
git clone https://github.com/data61/MP-SPDZ.git
```

### Step 3: Configure MP-SPDZ

```bash
cd MP-SPDZ
```

Follow the official MP-SPDZ installation and compilation instructions:

https://github.com/data61/MP-SPDZ

Complete the required setup before compiling the project program.

### Step 4: Compile the Secure EM aggregation program

From the MP-SPDZ directory:

```bash
./compile.py -R 64 secure_em_aggregate
```

The program source is:

```text
MP-SPDZ/Programs/Source/secure_em_aggregate.mpc
```

### Step 5: Return to the project root

```bash
cd ~/SecureEMProject

source .venv/bin/activate
```

---

## 8. Running the Experiments

### 8.1 Run Secure EM

Execute the Secure EM experiment with 600 samples:

```bash
PYTHONPATH=backend python backend/run_secure_em_mpc.py 600
```

The script executes the distributed EM workflow and records experiment metrics and results.

### 8.2 Run Centralized EM

```bash
python backend/benchmark_centralized.py 600
```

### 8.3 Run Repeated Comparison Benchmarks

```bash
python backend/repeated_comparison.py
```

This runs repeated centralized and Secure EM comparisons for the configured dataset sizes.

### 8.4 Run Numerical Validation

```bash
PYTHONPATH=backend python backend/validate_secure_em.py
```

### 8.5 Run Automated Tests

```bash
PYTHONPATH=backend python -m pytest backend/tests -v
```

---

## 9. Launching the Streamlit Dashboard

The project includes an interactive Streamlit dashboard for exploring experiments and recorded results.

### Run Locally

From the project root:

```bash
cd ~/SecureEMProject

source .venv/bin/activate

pip install -r requirements.txt

streamlit run backend/dashboard.py
```

Streamlit will display a local URL, normally:

```text
http://localhost:8501
```

Open that URL in your browser.

### Public Deployment

The dashboard is also deployed on Streamlit Community Cloud:

**https://secureemproject-htoeqwrap2fediaz8esczg.streamlit.app/**

The public dashboard provides access to the project's visualization and experiment interfaces.

Note: Live MPC execution requires a correctly configured MP-SPDZ runtime. The public Streamlit deployment does not automatically provide the local MP-SPDZ environment used in the research experiments.

---

## 10. Security Considerations and Limitations

This implementation is a research prototype.

Important limitations:

1. Three parties are simulated on a single machine.
2. The implementation does not represent three independently operated hospitals.
3. The MPC workflow securely aggregates sufficient statistics, which are subsequently revealed for the global M-step.
4. The current implementation uses local private-input files as part of its execution workflow. These files must be protected and should not be treated as a production secure input channel.
5. The protocol's security guarantees depend on its configuration and threat model.
6. No formal security audit or production deployment assessment has been performed.
7. Runtime measurements are hardware- and environment-dependent.

The project demonstrates a privacy-preserving computation workflow but should not be interpreted as a certified healthcare privacy solution.

---

## 11. Future Work

Potential extensions include:

- Deployment across independently operated MPC parties.
- Formal threat-model analysis and security evaluation.
- Improved communication and computation efficiency.
- Differential privacy integration.
- Larger-scale datasets and additional benchmarks.
- Robust handling of missing and heterogeneous data.
- Evaluation on real-world healthcare datasets under appropriate privacy and access controls.

---

## 12. Reproducibility

To reproduce the main experiment:

1. Clone the repository.
2. Install Python dependencies.
3. Install and configure MP-SPDZ.
4. Compile `secure_em_aggregate`.
5. Run the Secure EM experiment.
6. Run the centralized baseline.
7. Run repeated benchmarks.
8. Run numerical validation and automated tests.
9. Launch the Streamlit dashboard to inspect the results.

Experiment metrics and CSV outputs are stored in the `results/` directory.

---

## 13. Repository

**GitHub:** https://github.com/OmBawankule21/SecureEMProject

**Live Dashboard:** https://secureemproject-htoeqwrap2fediaz8esczg.streamlit.app/

---

*SecureEMProject — A research prototype exploring privacy-preserving distributed statistical learning using Expectation-Maximization and Secure Multiparty Computation.*
