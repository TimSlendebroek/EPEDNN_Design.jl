# EPEDNN.jl

Runs the EPEDNN pedestal model.

## This fork (EPEDNN_Design)

A fork of [ProjectTorreyPines/EPEDNN.jl](https://github.com/ProjectTorreyPines/EPEDNN.jl) by
Tim Slendebroek that adds a **12-input, height-only deep ensemble** (`EPEDNNEnsemble`) with
**uncertainty quantification** and out-of-distribution detection for the H-mode pedestal, trained
on the multi-machine EPED database plus new ITER-scale EPED scans. It powers the
**[EPEDexplorer](https://iter.fuseexplorer.com/EPED)** (interactive UI + prediction API).

### Citation

If you use these weights or this method, please cite the repository (see
[`CITATION.cff`](CITATION.cff) — GitHub's "Cite this repository" button), and the foundational
EPED-NN and pedestal-stability papers:

- O. Meneghini *et al.*, *Nucl. Fusion* **57**, 086034 (2017). doi:10.1088/1741-4326/aa7776
- P. B. Snyder *et al.*, *Phys. Plasmas* **9**, 2037 (2002). doi:10.1063/1.1449463

## Online documentation
For the upstream model, see the [online documentation](https://projecttorreypines.github.io/EPEDNN.jl/dev).

![Docs](https://github.com/ProjectTorreyPines/EPEDNN.jl/actions/workflows/make_docs.yml/badge.svg)
