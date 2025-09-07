# CVRP Dynamic Graph Transformers Paper

LaTeX document for the paper "Solving the Capacitated Vehicle Routing Problem with Dynamic Graph Transformers"

## Compilation

To compile the paper to PDF:

```bash
pdflatex main.tex
```

This will generate `main.pdf` from the LaTeX source.

## Files Structure

```
paper_dgt/
├── main.tex           # Main LaTeX document
├── figures/          # Figure directory
│   └── benchmark.eps # CPU benchmark comparison figure
└── README.md         # This file
```

## Figure Generation

The benchmark figure is generated from the benchmark data using:

```bash
cd ../benchmark_cpu
python plot_cpu_benchmark_paper.py --csv results/csv/benchmark_modified_20250827_074153.csv
cp benchmark.eps ../paper_dgt/figures/
```

## Requirements

- LaTeX distribution (TeXLive, MiKTeX, or MacTeX)
- `pdflatex` command available in PATH
- Standard LaTeX packages: `graphicx`

## Notes

- The figure uses EPS format for high-quality vector graphics
- Figure size is set to 86mm width for single-column layout
- Caption includes detailed explanation of both panels and solver behaviors
