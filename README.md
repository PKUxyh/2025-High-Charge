# 2025-High-Charge

Open-data repository for the manuscript accepted by **Physical Review Accelerators and Beams (PRAB)** — an APS journal.

This repository contains the figure codes (Python scripts and data files) used to generate every figure in the paper, in compliance with APS open-data policy.

---

## Paper information

| Field | Value |
|-------|-------|
| Journal | Physical Review Accelerators and Beams (PRAB) |
| Publisher | American Physical Society (APS) |
| Topic | High-charge electron beam generation and characterization |
| Year | 2025 |

---

## Repository structure

```
.
├── README.md
├── LICENSE
├── requirements.txt
├── fig1/          # Figure 1: Beam charge vs. laser pulse energy
│   ├── fig1.py
│   └── data/
├── fig2/          # Figure 2: Normalized emittance measurement
│   ├── fig2.py
│   └── data/
├── fig3/          # Figure 3: Energy spectrum
│   ├── fig3.py
│   └── data/
├── fig4/          # Figure 4: Transverse beam profile
│   ├── fig4.py
│   └── data/
└── fig5/          # Figure 5: Longitudinal bunch length
    ├── fig5.py
    └── data/
```

---

## Requirements

Python ≥ 3.8 is required. Install all dependencies with:

```bash
pip install -r requirements.txt
```

---

## Reproducing the figures

Each figure lives in its own sub-directory. Run the corresponding script to regenerate the figure:

```bash
cd fig1 && python fig1.py   # reproduces Figure 1
cd fig2 && python fig2.py   # reproduces Figure 2
cd fig3 && python fig3.py   # reproduces Figure 3
cd fig4 && python fig4.py   # reproduces Figure 4
cd fig5 && python fig5.py   # reproduces Figure 5
```

The output PDF/PNG files will be saved in the same directory.

---

## License

This repository is released under the [MIT License](LICENSE).
