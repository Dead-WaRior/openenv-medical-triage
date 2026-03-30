# 🏥 Medical Triage Environment

[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compatible-blue)](https://github.com/huggingface/openenv)
[![Healthcare](https://img.shields.io/badge/Domain-Healthcare-green)](https://github.com)
[![RL](https://img.shields.io/badge/RL-Environment-orange)](https://github.com)

# Problem Statement
Emergency departments worldwide face critical challenges: long wait times, patient LWBS (Left Without Being Seen), and resource constraints. This environment trains AI agents to optimize triage decisions, potentially saving lives and improving healthcare delivery.

# Key Features
- **Real-World Simulation**: Models actual ED operations with patient acuity, deterioration, and resource constraints
- **Clinical Guidelines**: Implements ESI (Emergency Severity Index) v4 triage protocol
- **Partial Progress Rewards**: Dense reward signals for learning complex behaviors
- **3 Progressive Tasks**: Easy (basic triage) → Medium (resource allocation) → Hard (mass casualty)

# Performance
| Task | Random Agent | Target |
|------|--------------|--------|
| Basic Triage | 0.45 | 0.70 |
| Resource Allocation | 0.38 | 0.60 |
| Mass Casualty | 0.32 | 0.50 |

# Quick Start
```bash
pip install -r requirements.txt
python inference.py