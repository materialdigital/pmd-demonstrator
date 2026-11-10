# PMD Demonstrators

This repository collects technical demonstrators for the Platform MaterialDigital (PMD). It shows concrete usage patterns for public PMD infrastructure, semantic data, notebooks and supporting Python tooling, in one series of notebooks that share one environment:

- Notebook 00 uses `praeco` directly to access Ontodocker and the MaterialDigital Dataportal, query RDF data, harvest publication metadata and publish records.
- Notebooks 01–06 on ten tensile tests of S355 steel. It analyses the raw curves with a semantic workflow built from the nodes of the package [`pmd_tensile`](pmd_tensile/) in this repository, on research graphs in two generations of the PMD core ontology (PMDco 2 and PMDco 3, both used together with the Tensile Test Ontology, TTO).

## Notebooks

| Notebook | Main audience | Content |
| --- | --- | --- |
| [`00-infrastructure-tour.ipynb`](notebooks/00-infrastructure-tour.ipynb) | all; the optional start | Ontodocker and the Dataportal with `praeco`: what is available, what it means, what belongs together, which metadata the harvester finds and a person reviews, publishing, and what others can access |
| [`01-tensile-data-from-graphs.ipynb`](notebooks/01-tensile-data-from-graphs.ipynb) | ontologists and RDF experts | the same test in PMDco 2 and PMDco 3, and ontology profiles mapping both to the same neutral records |
| [`02-tensile-method.ipynb`](notebooks/02-tensile-method.ipynb) | MSE domain experts | the method: for every property its definition, rule, node, parameters, flags and limits, drawn on Zx3, Zy3 and Zd3 and checked on synthetic curves; the one place for the limitations |
| [`03-tensile-workflow-building.ipynb`](notebooks/03-tensile-workflow-building.ipynb) | workflow developers | atomic nodes, the source of the per-specimen workflow, the batch assembled with `pyiron_workflow` and `flowrep`, a parameter change, and reuse of the nodes |
| [`04-tensile-workflow-knowledge-graph.ipynb`](notebooks/04-tensile-workflow-knowledge-graph.ipynb) | ontologists, workflow developers | what `semantikon` records about a recipe and a run (called directly), unit annotations, validation, the example queries printed, and one query across run graph and data graph |
| [`05-tensile-analysis-overview.ipynb`](notebooks/05-tensile-analysis-overview.ipynb) | all; entry point for MSE domain experts | the analysis of all ten specimens from PMDco 2 and PMDco 3 with identical results, the comparison with the lab values in steps of the lab's rounding, and two overview figures |

Reading paths through the tensile series, each ending with the publication of the results (notebook 06); readers new to Ontodocker and the Dataportal may start with notebook 00:

- MSE domain experts: 05 → 02 → 01 → 06
- Ontologists and RDF experts: 01 → 04 → 05 → 06
- Workflow developers: 03 → 04 → 02 → 05 → 06

The tensile-test properties (E, ReH, ReL, Rp0.2, Fm, Rm, Ag, Agt, A, Z, with flags and uncertainty components) are defined by the method of notebook 02, inspired by ISO 6892-1; notebook 02 also lists the limitations of the method and of the data.

## Setup

There is one environment file, [`environment.yml`](environment.yml), for the environment `pmd-demonstrator`. Run the commands from the repository root: the file's pip section installs `pmd_tensile` editable from [`pyproject.toml`](pyproject.toml) with `-e .`, relative to the directory pip runs in.

```bash
mamba env create -f environment.yml                       # a new environment
mamba env update -n pmd-demonstrator -f environment.yml   # or: update an existing one
conda activate pmd-demonstrator
python -m ipykernel install --user --name pmd-demonstrator --display-name "Python 3.12 (pmd-demonstrator)"
cd notebooks
jupyter lab
```

`conda` works in place of `mamba`. An update without `--prune` keeps packages installed locally in addition, such as `pytest` or `papermill`. With mamba 2.3.3, `mamba env update --dry-run` still runs the pip step of the file, so it is not free of side effects.

Select the kernel "Python 3.12 (pmd-demonstrator)". The notebooks import `pmd_tensile` like any installed package and find the repository's files through it, so they do not depend on the working directory.

The environment installs `praeco` 0.2.0 from conda-forge, which brings the required pandas, RDFLib, Pydantic, Requests and SPARQLWrapper versions. For the tensile series it pins the workflow stack (`pyiron_workflow` 0.20.3, `semantikon` 1.4.4, `flowrep` 0.6.7) and adds NumPy, SciPy, Matplotlib, Pint and python-graphviz. Notebook 03 draws workflows with Graphviz's `dot`, which is on `PATH` in the activated environment; when running notebooks with papermill without activating the environment, put the environment's `bin` directory first on `PATH`. `pip` and `git` stay in the environment, e.g. for the editable install below.

For local development of `praeco`, replace the installed package with an editable checkout in the activated environment:

```bash
python -m pip install -e /path/to/praeco
python -m pip check
```

Restart any running notebook kernel after changing the installation.

## Credentials and execution

Set `ONTODOCKER_ADDRESS`, `ONTODOCKER_TOKEN` (if required by the service) and, for publishing, `DATAPORTAL_TOKEN` in the environment before starting Jupyter. Keep credentials out of the notebooks and version control.

Notebook 00 only reads by default; with its papermill parameter `publish = True` it creates the two data records of its section 5, which must not exist. Review its reviewed metadata before setting it.

For the tensile notebooks:

- They only check that `ONTODOCKER_ADDRESS` is set; the package `pmd_tensile` reads `ONTODOCKER_ADDRESS` and `ONTODOCKER_TOKEN` (`pmd_tensile.sources.ontodocker_client()`) and prints neither. No Dataportal token is needed.
- Credentials are never workflow inputs, so they cannot enter results or knowledge graphs; [`tests/test_separation.py`](tests/test_separation.py) checks this with a dummy token.
- Each notebook states its execution class. Notebooks 01–05 are network-read: SPARQL queries and Turtle downloads from Ontodocker and curve CSVs from the public data host. They create, upload or delete no remote resources; notebook 04 writes graph files only to a temporary directory that it creates and removes.
- Each can be run with papermill, a development tool that is not part of `environment.yml` (install it locally, like `pytest`), e.g. from `notebooks/`: `papermill -k pmd-demonstrator 05-tensile-analysis-overview.ipynb /tmp/05-tensile-analysis-overview.ipynb`.

## What is hosted where

| Where | What |
| --- | --- |
| Ontodocker | `pmdco2_tto_example_{parallel,perpendicular,diagonal}`: the PMDco 2 graphs of the tests, one dataset per orientation group. Local copies are in [`downloaded/`](downloaded/). |
| Ontodocker | `pmdco3_tto_example_{parallel,perpendicular,diagonal}`: the PMDco 3 graphs, cut by [`scripts/slice_pmdco3.py`](scripts/slice_pmdco3.py) from the original graph of the TTO repository at commit `08a7aec` (committed unchanged as [`datasets/S355_data_tto_pmdco3.ttl`](datasets/S355_data_tto_pmdco3.ttl)). The script adds one CSVW curve table per test, relinks the lab table to the data host, and appends the orientation to the title; the added triples are marked with `rdfs:comment`. The slices and the full file [`datasets/pmdco3_tto_example.ttl`](datasets/pmdco3_tto_example.ttl) are committed. Ontodocker returns `xsd:string` literals as plain literals, which is the same term under RDF 1.1. |
| Data host `https://data.pmds-mpcdf-2.pyiron.com/` | the ten raw curves `Zx1.csv` … `Zx4.csv`, `Zy1.csv` … `Zy4.csv`, `Zd2.csv`, `Zd3.csv`, linked by both generations, and the lab table `S355_data.csv`, unchanged from the TTO repository at commit `08a7aec`; public, read without a token |
| Dataportal | [`pmdco2-tto-recombined`](https://dataportal.material-digital.de/dataset/pmdco2-tto-recombined): the combined PMDco 2 graph, published by notebook 00 |
| Dataportal | [`pmdco3-tto-recombined`](https://dataportal.material-digital.de/dataset/pmdco3-tto-recombined): the full PMDco 3 file with links to the eleven CSV files, with metadata harvested from the graph with `praeco` and reviewed by a person |

## The `pmd_tensile` package

`pmd_tensile` holds the workflow nodes and macros, so other notebooks can import them. It is installed editable by `environment.yml`.

- **`pmd_tensile.nodes`** contains only `@fr.atomic` nodes, `@fr.workflow` macros, and the types and constants they need:
  - `nodes.exploration` and `nodes.acquisition`: the only ontology-aware nodes (conformance check, specimen listing, specimen record, lab-reported values, curve download);
  - `nodes.analysis`: the ontology-agnostic analysis nodes, one module per topic (extensometer, elastic line, yielding, maximum force, geometry, comparison). Their inputs and outputs are numbers and arrays with unit and quantity-kind annotations, plus one diagnostics object per node;
  - `nodes.workflows`: adapter nodes between the neutral records and the analysis, and the macros `exploration_workflow` and `specimen_workflow`.
  - `nodes.results`: the ontology-aware results node, which writes the computed properties as RDF in PMDco 3 terms;
  - `nodes.publishing`: nodes that write the files of a publication, create an Ontodocker dataset and a Dataportal record, and the macro `publishing_workflow`.
- **Helper modules**, ordinary modules outside `nodes`: `contract` (the neutral, ontology-free records), `profiles` (the ontology profiles `pmdco2-tto` and `pmdco3-tto` with their SPARQL queries, unit aliases and conventions), `sources` (graph sources, the token-free curve download, and the Ontodocker and Dataportal clients built from the environment), `parameters` (the method parameters), `orchestration` (building and running the batch, `run_tensile_analysis`), `knowledge_graph` (export, validation, queries) and `plotting` (the figures).

This layout prepares the package for collection by [sim_atlas](https://github.com/XzzX/sim_atlas/): [`tests/test_sim_atlas_collection.py`](tests/test_sim_atlas_collection.py) imitates its collector (commit `753205b`) and checks that only nodes, macros and dataclass types are collected from `pmd_tensile.nodes`, no helper, and that importing the package opens no network connection. The nodes have not been uploaded to a sim_atlas instance, and one gap remains: semantikon stores units and quantity kinds as a flat tuple in the type hint's metadata, while sim_atlas at that commit reads units only from a dictionary, so a catalogue entry would currently lose the units (notebook 03, section 7).

## Tests

The tests in [`tests/`](tests/) run offline and block outbound connections. They use the local graphs (the PMDco 2 copies in `downloaded/`, the PMDco 3 files in `datasets/`) and synthetic curves. `pytest` is not part of `environment.yml`; install it locally to run the tests from the repository root:

```bash
python -m pytest -p no:cacheprovider tests -m "not network"               # the default run
python -m pytest -p no:cacheprovider tests -m "not network" --run-slow    # also the slow tests (full graphs, knowledge-graph exports)
python -m pytest -p no:cacheprovider tests --run-statistical -m statistical  # Monte Carlo tolerance checks, slower
python -m pytest -p no:cacheprovider tests --run-network -m network  # checks that the connection block is lifted (loopback only)
```

## Archive

[`archive/mesh-demonstrator/`](archive/mesh-demonstrator/) holds the earlier PMD Mesh demonstrator: service discovery inside the PMD Mesh, Ontodocker, federated SPARQL queries and a first exploration of the same S355 data. It is no longer maintained and needs access to the PMD Mesh; see its [README](archive/mesh-demonstrator/README.md).

## Planned extensions

- Docker images pre-populated with this repository, or selected parts of it, for use in JupyterHub environments.
- Smaller images or examples that contain only one part and its required datasets and tools.
