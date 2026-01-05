# Title

**GIP-MI**: Software Aging State Determination and Rejuvenation Strategy Generation for KubeEdge

## Abstract

Long-running edge platforms may suffer from software aging, which can manifest as resource pressure, increasing response latency, and even service interruptions. GIP-MI is a metrics-driven pipeline for KubeEdge edge systems that:

1. Collects time-series metrics such as CPU utilization, memory utilization, and task average response time from a monitoring backend (e.g., Prometheus).
2. Predicts near-future metric trends with **GCN-Informer** by modeling temporal dependencies and cross-metric correlations.
3. Identifies dynamic aging states with **ParNet** via multi-timepoint slicing and multi-resolution feature fusion.
4. Generates a lightweight rejuvenation strategy through task offloading using **MOEA/D-IFM** to reduce downtime and maintain service continuity.

This repository contains the following components:

- `GCN-Informer/`: multivariate time-series forecasting (CPU / memory / response time).
- `ParNet/`: sliding-window classifier (default window size = 21).
- `MOEAD-IFM/`: multi-objective optimization for task-to-node assignment (offloading/placement).

## Usage

Each component is runnable on its own. Run commands from the corresponding subfolder under `GIP/`.

### 1) GCN-Informer

```bash
cd GCN-Informer
pip install -r requirements.txt
python train_gcn_informer.py
```

### 2) ParNet

```bash
cd ParNet
pip install -r requirements.txt
python parnet.py
```

### 3) MOEAD-IFM

```bash
cd MOEAD-IFM
pip install -r requirements.txt
python moead_ifm.py
```
