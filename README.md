# AE4350-Bio-inspired-Intelligence-and-Learning-for-Aerospace-Applications
Unified Python implementation of Ant System (AS), Ant Colony System (ACS), and Max–Min Ant System (MMAS) for solving Euclidean TSP, with reproducible experiments and parameter studies.

# Ant Colony Optimization for Euclidean TSP

This repository implements three Ant Colony Optimization (ACO) variants for solving the Euclidean Traveling Salesman Problem (TSP):

- **Ant System (AS)**
- **Ant Colony System (ACS)**
- **Max–Min Ant System (MMAS)**

The implementation is modular, reproducible, and designed to support systematic parameter studies across multiple algorithmic configurations. It includes tools for generating random Euclidean instances, visualizing convergence, and exporting high-quality figures and tables for reporting.

---

### ✏️ Background

Part of the motivation for this project comes from a small logistics-related path planning task I worked on during my bachelor exchange semester in Australia. That earlier experience introduced me to the idea of routing problems with geometric structure, and helped spark my interest in bio-inspired optimization. Only a very small part of the implementation here is adapted from that context—this project has been developed from the ground up, with a focus on ACO variants and a more systematic experimental setup.

---

### 🐜 Tour Evolution GIFs

To complement the report, this repository includes **animated GIFs** showing how the best path evolves over time for each algorithm (AS, ACS, MMAS). These visualizations help illustrate the search dynamics and make it easier to compare how each system explores and converges. The animations are located in the `/gifs/` folder.

---

### 📊 Features

- Clean Python implementation with modular architecture
- Parameter sweeps over $(\alpha, \beta)$, $\rho$, $q_0$, $\phi$, $(\tau_{\min}, \tau_{\max})$
- Equal-budget evaluation and fair comparison across solvers
- Random Euclidean instances and TSPLIB-compatible loading
- High-resolution plots and LaTeX-ready outputs
- Repeated trials, convergence curves, confidence intervals
- Animated visualizations of best-so-far tours over time

 
