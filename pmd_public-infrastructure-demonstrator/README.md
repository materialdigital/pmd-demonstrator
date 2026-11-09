# Public PMD infrastructure demonstrator

This directory holds two parts that share one environment:

- `notebooks/demonstrator.ipynb` uses `praeco` to access Ontodocker and the MaterialDigital Dataportal, query RDF data, and publish datasets and assets.
- A series of five notebooks on ten tensile tests of S355 steel. It analyses the raw curves with a semantic workflow built from the nodes of the package [`pmd_tensile`](pmd_tensile/) in this directory, on research graphs in two generations of the PMD core ontology (PMDco 2 and PMDco 3 with the Tensile Test Ontology, TTO).

## Notebooks

| Notebook | Main audience | Content |
| --- | --- | --- |
| [`demonstrator.ipynb`](notebooks/demonstrator.ipynb) | all; a guided presentation | Ontodocker and Dataportal with `praeco`: what is available, what it means, what belongs together, which metadata can be reused, what others can access |
| [`01-tensile-data-from-graphs.ipynb`](notebooks/01-tensile-data-from-graphs.ipynb) | ontologists and RDF experts | the same test in PMDco 2 and PMDco 3, ontology profiles mapping both to the same neutral records, and publication metadata harvested from the graphs with `praeco` 0.2.0 |
| [`02-tensile-analysis-nodes.ipynb`](notebooks/02-tensile-analysis-nodes.ipynb) | MSE experts | the analysis nodes one by one on Zx3, Zy3 and Zd3: what each computes, on which part of the curve, and where it fails |
| [`03-tensile-workflow-building.ipynb`](notebooks/03-tensile-workflow-building.ipynb) | workflow developers | atomic nodes, the per-specimen workflow and the batch with `pyiron_workflow` and `flowrep`, a parameter change, and reuse of the nodes |
| [`04-tensile-workflow-knowledge-graph.ipynb`](notebooks/04-tensile-workflow-knowledge-graph.ipynb) | ontologists, workflow developers | what `semantikon` records about a recipe and a run, unit annotations, validation, and one query across run graph and data graph |
| [`05-tensile-analysis-overview.ipynb`](notebooks/05-tensile-analysis-overview.ipynb) | all; entry point for MSE experts | the analysis of all ten specimens from PMDco 2 and PMDco 3 with identical results, the comparison with the lab values, two overview figures, and the limitations |

Reading paths through the tensile series:

- MSE experts: 05 → 02 → 01
- Ontologists and RDF experts: 01 → 04 → 05
- Workflow developers: 03 → 04 → 02 → 05

The tensile-test properties (E, ReH, ReL, Rp0.2, Fm, Rm, Ag, Agt, A, Z, with flags and uncertainty components) are method-defined and ISO 6892-1-inspired, not ISO-conformant. The comparison with the lab-reported values is a consistency check, not a validation.

## Setup

There is one environment file, [`environment.yml`](environment.yml), for the environment `pmd-demonstrator`. Run the commands from this directory: the file's pip section installs `pmd_tensile` editable from [`pyproject.toml`](pyproject.toml) with `-e .`, relative to the directory pip runs in.

```bash
cd pmd_public-infrastructure-demonstrator
mamba env create -f environment.yml                       # a new environment
mamba env update -n pmd-demonstrator -f environment.yml   # or: update an existing one
conda activate pmd-demonstrator
python -m ipykernel install --user --name pmd-demonstrator --display-name "Python 3.12 (pmd-demonstrator)"
cd notebooks
jupyter lab
```

`conda` works in place of `mamba`. An update without `--prune` keeps packages installed locally in addition, such as `pytest` or `papermill`. With mamba 2.3.3, `mamba env update --dry-run` still runs the pip step of the file, so it is not free of side effects.

Select the kernel "Python 3.12 (pmd-demonstrator)". For `demonstrator.ipynb`, keep the kernel's working directory in this `notebooks` directory; the notebook checks that location and derives its data and download paths from the parent directory. The tensile notebooks import `pmd_tensile` like any installed package and do not depend on the working directory.

The environment installs `praeco` 0.2.0 from conda-forge, which brings the required pandas, RDFLib, Pydantic, Requests and SPARQLWrapper versions. For the tensile series it pins the workflow stack (`pyiron_workflow` 0.20.3, `semantikon` 1.4.4, `flowrep` 0.6.7) and adds NumPy, SciPy, Matplotlib, Pint and python-graphviz. Notebook 03 draws workflows with Graphviz's `dot`, which is on `PATH` in the activated environment; when running notebooks with papermill without activating the environment, put the environment's `bin` directory first on `PATH`. `pip` and `git` stay in the environment, e.g. for the editable install below.

For local development of `praeco`, replace the installed package with an editable checkout in the activated environment:

```bash
python -m pip install -e /path/to/praeco
python -m pip check
```

When migrating an existing environment, first remove the old editable package with `python -m pip uninstall courier`. Restart any running notebook kernel after changing the installation.

## Credentials and execution

Set `ONTODOCKER_ADDRESS`, `ONTODOCKER_TOKEN` (if required by the service) and, for publishing from `demonstrator.ipynb`, `DATAPORTAL_TOKEN` in the environment before starting Jupyter. Keep credentials out of the notebooks and version control.

`demonstrator.ipynb` combines local computation, remote queries, and operations that create datasets and upload assets. Review the creation, publication, and cleanup cells before executing them against a service.

The tensile notebooks read only `ONTODOCKER_ADDRESS` and `ONTODOCKER_TOKEN` and print neither. Credentials are never workflow inputs, so they cannot enter results or knowledge graphs. No Dataportal token is needed: notebook 01 reads the two Dataportal records publicly. Each notebook states its execution class; all five are network-read: SPARQL queries and Turtle downloads from Ontodocker, curve CSVs from the public data host, and public Dataportal reads. They create, upload or delete no remote resources. Notebook 04 writes graph files only to a temporary directory that it creates and removes. Each tensile notebook can be run with papermill, e.g. from `notebooks/`: `papermill -k pmd-demonstrator 05-tensile-analysis-overview.ipynb /tmp/05-tensile-analysis-overview.ipynb`.

## What is hosted where

| Where | What |
| --- | --- |
| Ontodocker | `pmdco2_tto_example_{parallel,perpendicular,diagonal}`: the PMDco 2 graphs of the tests, one dataset per orientation group. Local copies are in [`downloaded/`](downloaded/). |
| Ontodocker | `pmdco3_tto_example_{parallel,perpendicular,diagonal}`: the PMDco 3 graphs, cut by [`scripts/slice_pmdco3.py`](scripts/slice_pmdco3.py) from the original graph of the TTO repository at commit `08a7aec` (committed unchanged as [`datasets/S355_data_tto_pmdco3.ttl`](datasets/S355_data_tto_pmdco3.ttl)). The script adds one CSVW curve table per test, relinks the lab table to the data host, and appends the orientation to the title; the added triples are marked with `rdfs:comment`. The slices and the full file [`datasets/pmdco3_tto_example.ttl`](datasets/pmdco3_tto_example.ttl) are committed. Ontodocker returns `xsd:string` literals as plain literals, which is the same term under RDF 1.1. |
| Data host `https://data.pmds-mpcdf-2.pyiron.com/` | the ten raw curves `Zx1.csv` … `Zx4.csv`, `Zy1.csv` … `Zy4.csv`, `Zd2.csv`, `Zd3.csv`, linked by both generations, and the lab table `S355_data.csv`, unchanged from the TTO repository at commit `08a7aec`; public, read without a token |
| Dataportal | [`pmdco2-tto-recombined`](https://dataportal.material-digital.de/dataset/pmdco2-tto-recombined): the combined PMDco 2 graph, published from `demonstrator.ipynb` |
| Dataportal | [`pmdco3-tto-recombined`](https://dataportal.material-digital.de/dataset/pmdco3-tto-recombined): the full PMDco 3 file with links to the eleven CSV files, with metadata harvested from the graph and reviewed (notebook 01) |

## The `pmd_tensile` package

`pmd_tensile` holds the workflow nodes and macros, so other notebooks can import them. It is installed editable by `environment.yml`.

- **Node modules** contain only `@fr.atomic` nodes, `@fr.workflow` macros, and the types and constants they need:
  - `exploration` and `acquisition`: the only ontology-aware nodes (conformance check, specimen listing, specimen record, lab-reported values, curve download);
  - `analysis`: the ontology-agnostic analysis nodes and their result types, with unit and quantity-kind annotations;
  - `workflows`: adapter nodes between the neutral records and the analysis, and the macros `exploration_workflow` and `specimen_workflow`.
- `contract`: the neutral, ontology-free records and unit conversion.
- `profiles`: the ontology profiles `pmdco2-tto` and `pmdco3-tto` (SPARQL queries, unit aliases, conventions) and their registry.
- **Private helpers** in `_`-prefixed modules: graph sources and the token-free curve download, the plain-function forms of the nodes, building and running the batch workflow (`run_tensile_analysis`), the method parameters, knowledge-graph export and queries, and the figures. Notebooks reach them through the facade `pmd_tensile.demo`.

This layout makes the package ready for collection by [sim_atlas](https://github.com/XzzX/sim_atlas/): [`tests/test_sim_atlas_collection.py`](tests/test_sim_atlas_collection.py) imitates its collector (commit `753205b`) and checks that only nodes, macros and dataclass types are collected, no helper, and that importing the package opens no network connection. The nodes have not been uploaded to a sim_atlas instance.

## Tests

The tests in [`tests/`](tests/) run offline and block outbound connections. They use the local graphs (the PMDco 2 copies in `downloaded/`, the PMDco 3 files in `datasets/`) and synthetic curves. `pytest` is not part of `environment.yml`; install it locally to run the tests from this directory:

```bash
python -m pytest -p no:cacheprovider tests -m "not network"
python -m pytest -p no:cacheprovider tests --run-statistical -m statistical  # Monte Carlo tolerance checks, slower
python -m pytest -p no:cacheprovider tests --run-network -m network  # checks that the connection block is lifted (loopback only)
```

## Presenter preparation

The walkthrough follows five questions: what is available, what it means, what belongs together, which metadata can be reused, and what others can access. Allow 45 minutes for the notebook, following the 20-minute introduction.

The settings default to `reuse` for both Ontodocker and the Dataportal. This inspects existing remote datasets while downloading and assembling local files. For a new publication, set `publication_mode = "create"`, choose an unused `publication_name`, review the metadata, and set `publication_reviewed = True`. The Ontodocker mode is independent: creating a new publication can reuse the existing research graphs. Creation stops if the chosen name already exists.

Confirm `owner_org` and the intended license. The current source RDF has an ambiguous creator-name/ORCID mapping; the notebook flags it and retains names without assigning those identifiers. The original identifiers remain in audit metadata for review.

Before presenting, rehearse from a fresh kernel with the intended settings and inspect the compact outputs. Retained outputs from earlier executions must be refreshed after code changes. Keep a previously executed copy available as a fallback, and leave cleanup cells commented out during the session.
