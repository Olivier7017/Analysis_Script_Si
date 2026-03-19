# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

Analysis of Si and SiGe molecular dynamics trajectories, comparing interatomic potential models with and without ZBL (Ziegler-Biersack-Littmark) potential. The ZBL potential is relevant for high-energy collision simulations.

## Running Scripts

```bash
python scripts/analysis.py       # Main analysis pipeline (entry point)
python scripts/rdf_calc.py       # RDF calculation module (no standalone entry point)
```

## Dependencies

- `numpy`, `matplotlib`
- `ase` (Atomic Simulation Environment) — trajectory I/O and physical units
- `ovito` — primary analysis library (`CoordinationAnalysisModifier`, `import_file`)

## Architecture

**`rdf_calc.py`** — Core computation module. `partial_rdf(trajfile, n_bins, rcut, prefix)` loads `.traj` files via Ovito, applies `CoordinationAnalysisModifier` (partial=True), averages RDF across all frames, and writes `{prefix}_rdf_{fn}.dat`.

**`analysis.py`** — Orchestration script (work in progress). Iterates over sample folders (`ref`, `noZBL`, `ZBL`), calls `parse_traj(folder)` to group trajectory files by `{elems}{supercell}_{nsteps}*.traj` pattern, then dispatches to `partial_rdf`. Currently incomplete — contains `exit()` stubs and references unreferenced variables.

**`energydist_calc.py`** — Stub for energy distribution analysis (not yet implemented).

## Sample Data

Trajectory files live in `../samples/{ref,noZBL,ZBL}/` with naming pattern `{elems}{supercell}_{nsteps}.traj` (e.g., `Si111_250.traj`, `SiGe222_250.traj`). Supercell sizes: 111, 222, 333; 250 MD steps each.

## Known Issues in analysis.py

- `exit()` at line 16 (inside the folder loop) and line 42 (inside `parse_traj`) halt execution before any analysis runs
- `normal_fn` and `reftrajfile` are referenced but never defined (lines 18, 25)
- Lines 18–28 appear to be leftover code from an earlier Cu project (`01-Cu_ExciseAndRepaint`)
