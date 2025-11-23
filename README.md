# RFCIL: Regularization-based Federated Class-Incremental Learning

> ⚠️ **Note: This repository contains a partial implementation of the RFCIL framework.**  
> The full codebase will be released upon formal acceptance of the associated research paper.  
> We appreciate your patience and interest in our work.

## 📖 Overview

Federated Learning (FL) enables collaborative model training across decentralized devices while preserving data privacy. However, it often suffers from **catastrophic forgetting** when encountering new, unseen task classes over time. Existing federated incremental learning methods typically require persistent local storage of historical data, introducing substantial storage overhead and potential privacy risks.

To address these challenges, we propose **RFCIL** — a novel framework for **Regularization-based Federated Class-Incremental Learning**. RFCIL enables client devices to adaptively recognize evolving tasks in distributed systems **without storing historical data**, through adaptive regularization and server-coordinated optimization.

## 🛠️ Implementation Status

| Component | Status | Notes |
|-----------|--------|-------|
| RFCIL Framework Core | 🔄 **Partial** | Basic federated learning pipeline implemented |
| MA-former Architecture | 🔄 **Partial** | Transformer backbone available; full parameter decomposition in progress |
| Adaptive Regularization Strategy | 🔄 **Partial** | Server coordination mechanism under development |
| Experimental Benchmarks | ❌ **Pending** | Full evaluation suite to be released with paper |

## 📋 Current Code Contents

This preliminary release includes:
- Basic federated learning infrastructure
- Core client-server communication protocol
- Initial implementation of the MA-former architecture
- Example training scripts for standard federated learning scenarios

## 🔮 Coming Soon

Upon paper acceptance, we will release:
- Complete RFCIL implementation with all adaptive regularization components
- Full MA-former architecture with structured parameter decomposition
- Pre-trained models and configuration files
- Comprehensive evaluation benchmarks and comparison scripts
- Detailed documentation and usage examples

## 📜 Citation

If you find our work useful, please consider citing our paper once it is officially published. Citation information will be updated here upon publication.

## 📧 Contact

For questions about this work or the upcoming full release, please contact the corresponding authors.

---

**Stay tuned for the complete implementation!**
