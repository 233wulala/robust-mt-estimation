# Robust Regression Estimators for Magnetotelluric Impedance Estimation

[![DOI](https://zenodo.org/badge/1238823095.svg)](https://doi.org/10.5281/zenodo.20197019)

This repository contains Python implementations and experiment scripts for comparing robust regression approaches to magnetotelluric (MT) impedance estimation under contaminated and mixed-noise conditions. The code is organized around M-, S-, and MM-estimation, with local-regression integrations and synthetic/field-data processing utilities.

> **Reproducibility status:** this is a research code repository, not yet a turnkey package. Several scripts contain machine-specific Windows paths, and some plotting scripts expect intermediate CSV/weight files that are not present in the repository. See [Reproduction status and limitations](#reproduction-status-and-limitations) before running the scripts.

## Research question

MT impedance estimates are obtained from frequency-domain relationships between electric and magnetic field components. Noisy time windows, transient contamination, and influential observations can affect the spectral regression and the resulting impedance tensor. This project explores how robust regression estimators can reduce the influence of such observations, and how their behavior compares across synthetic and recorded data.

The repository is intended to support method comparison and inspection of implementation details. It does not, by itself, provide a complete, environment-independent reproduction of every result reported in the associated research.

## Methods at a glance

| Method | Implementation in this repository | Estimation outline |
| --- | --- | --- |
| **M-estimation** | `regression/robust.py`, used by `regression/local.py` | Iteratively reweighted regression. The implementation includes residual-based robust weights and leverage-based downweighting; the two-stage helper initializes with Huber weights and then applies bisquare weights. |
| **S-estimation** | `regression/s_robust.py`, used by `regression/s_local.py` | Estimates regression coefficients together with a robust residual scale. The implementation uses bisquare weighting by default and supports multiple starts. |
| **MM-estimation** | `regression/mm_robust.py`, used by `regression/mm_local.py` | Uses an S-estimate for an initial robust fit/scale, followed by an M-estimation refinement with the initial scale held fixed. The current helper requests 10 S-estimation starts and uses bisquare weights for the refinement. |

These summaries describe the code currently present in this repository. Exact options, convergence behavior, and the regression design used in a particular experiment should be checked in the corresponding source file and experiment configuration. The estimators are presented for comparison; this README does not claim that one method is universally best.

## Repository layout

| Path | Contents |
| --- | --- |
| `regression/` | Robust regression functions and local-regression implementations integrated with the `resistics` workflow. The `*_local.py` files perform frequency-local transfer-function processing; the matching `*_robust.py` files contain estimator routines. |
| `synthetic_experiments/` | Scripts for preparing a `resistics` project, processing spectra/transfer functions, and plotting a synthetic/noisy-data experiment. `run.py` is the visible experiment workflow entry, but it depends on local paths and project state. |
| `statistics/` | Scripts for preparing project metadata, calculating spectra/statistics, and plotting impedance/weight results. Some required CSV and weight outputs are not tracked. |
| `site1/meas/` | Five-channel ASCII time-series files and headers. The original README describes this as data with added noise. |
| `site2/meas/` | Five-channel ASCII time-series files and headers. The original README describes this as synthesized clean electromagnetic-field data. |
| `README.md` | Project overview, method summary, and current reproducibility notes. |

The checked-in headers identify the two measurement folders as 5-channel records sampled at 0.5 Hz with 430,000 samples (approximately 10 days). They do not, on their own, document the full generation/processing history or noise parameters.

## Environment

The code imports Python scientific packages including NumPy, SciPy, pandas, and Matplotlib, and uses the `resistics` package for project, spectral, window, and transfer-function processing.

The repository does **not** currently include a pinned dependency file (such as `requirements.txt` or `environment.yml`), a declared supported Python version, or a documented tested package-version matrix. Install versions compatible with the checked-out scripts and the relevant `resistics` API; record the versions used when reproducing results.

## Running the current scripts

Clone the repository and inspect the configured paths first:

```bash
git clone https://github.com/233wulala/robust-mt-estimation.git
cd robust-mt-estimation
```

The current scripts are not ready to run unchanged from a fresh clone. In particular:

1. `synthetic_experiments/datapaths.py` and `statistics/datapaths.py` point to absolute `D:/...` locations.
2. The project-preparation scripts create `resistics` project metadata, while the recorded data are under the repository's `site1/` and `site2/` folders; the scripts do not document a complete portable mapping between these paths.
3. `synthetic_experiments/run.py` assumes project data and output folders already exist and writes to local image/EDI paths.
4. `statistics/4_plot_Z.py` reads impedance CSV and weight CSV files that are not present in the tracked tree.

Before running, edit the path definitions and verify the input/output paths expected by each script. The main source entry points to inspect are:

- M-estimation: [`regression/local.py`](regression/local.py) and [`regression/robust.py`](regression/robust.py)
- S-estimation: [`regression/s_local.py`](regression/s_local.py) and [`regression/s_robust.py`](regression/s_robust.py)
- MM-estimation: [`regression/mm_local.py`](regression/mm_local.py) and [`regression/mm_robust.py`](regression/mm_robust.py)
- Synthetic workflow: [`synthetic_experiments/run.py`](synthetic_experiments/run.py)
- Statistics and plotting: [`statistics/2_spectra.py`](statistics/2_spectra.py), [`statistics/3_statistics.py`](statistics/3_statistics.py), and [`statistics/4_plot_Z.py`](statistics/4_plot_Z.py)

No single command is currently documented as a verified end-to-end reproduction command. The scripts should be treated as research workflow components until paths, dependencies, expected inputs, and output products are documented and tested together.

## Experiments and figure interpretation

The tracked workflow includes:

- A synthetic-experiment script that loads a `resistics` project, calculates spectra and transfer functions, and saves an impedance plot.
- Statistical/plotting scripts that calculate transfer-function statistics and create impedance/weight visualizations.
- Per-observation robust weights in the regression implementations, with commented code indicating intended weight export in some paths.

The plotting code indicates that some figures are intended to show impedance components and robust weights. However, the repository does not currently track a figure index, experiment manifest, complete intermediate outputs, or captions linking each figure to a defined noise realization and parameter set. Therefore no numerical performance claim or figure-specific result is asserted here. Add the relevant generated figures and a caption/manifest before using this section to present quantitative outcomes.

## Data and code limitations

- The checked-in time-series data are large ASCII files with headers. Their presence does not establish the full provenance, ownership, license, preprocessing steps, or noise-generation parameters.
- The top-level README labels `site1` as noise-added data and `site2` as synthesized clean field data, but a detailed data dictionary and generation recipe are not included.
- Data redistribution terms and any restrictions on field data are not documented. Confirm permission and add an explicit license/data-use statement before encouraging reuse.
- Machine-specific paths and untracked intermediate files prevent reliable reproduction from a clean checkout.
- There is no dependency lockfile, automated test suite, or continuous-integration workflow visible in the repository.
- Some tracked folders contain local IDE or Python cache artifacts; these are not part of the scientific workflow.

## Reproduction status and limitations

To make the results independently reproducible, the repository still needs:

1. A portable configuration for project/data/output paths and a clear command sequence from raw input to figures.
2. A pinned environment file with Python, `resistics`, NumPy, SciPy, pandas, and Matplotlib versions.
3. An experiment manifest documenting the clean reference, noise types and levels, random seeds, windowing/spectral settings, estimator options, and output names.
4. The required input/intermediate CSV and weight files, or scripts that regenerate them from the tracked data.
5. A figure directory with publication-quality plots and captions mapping each figure to a script/configuration.
6. A data dictionary and explicit licenses/permissions for both code and data.
7. A minimal smoke test or documented verification output demonstrating that each estimator can be run from a clean environment.

Until these items are added and verified, use the repository to inspect the implementation and workflow structure; do not treat the current scripts as a one-command reproduction package.

## Citation and DOI

A Zenodo DOI badge is present in the repository:

**https://doi.org/10.5281/zenodo.20197019**

For citation, use the Zenodo record's recommended citation and version-specific DOI. A publication title, author list, venue, and bibliographic citation are not specified in the repository README, so they are intentionally not reconstructed here. If this repository supports a manuscript, add its full citation once the publication record is confirmed.

## License

No repository license file was visible in the checked-out tree. All rights reserved by default unless a license is added; contact the author before reusing or redistributing code or data.
