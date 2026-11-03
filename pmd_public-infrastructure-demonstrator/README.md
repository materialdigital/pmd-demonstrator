# Public PMD infrastructure demonstrator

The notebook uses `praeco` to access Ontodocker and the MaterialDigital
Dataportal, query RDF data, and publish datasets and assets.

## Setup

From the repository root:

```bash
conda env create -f pmd_public-infrastructure-demonstrator/environment.yml
conda activate pmd-demonstrator
python -m ipykernel install --user --name pmd-demonstrator --display-name "Python 3.12 (pmd-demonstrator)"
cd pmd_public-infrastructure-demonstrator/notebooks
jupyter lab
```

Open `demonstrator.ipynb` and select the `pmd-demonstrator` kernel. Keep the
kernel's working directory in this `notebooks` directory. The notebook checks
that location and derives its data and download paths from the parent directory.

The environment installs the `praeco` 0.1.0 release from conda-forge, which brings the required pandas, RDFLib, Pydantic, Requests, and SPARQLWrapper versions. For the tensile-test workflow notebook, it also pins the pyiron workflow stack (`pyiron_workflow` 0.20.3, `semantikon` 1.4.4, `flowrep` 0.6.7) and adds NumPy, SciPy, Matplotlib, Pint, and python-graphviz. `pip` and `git` stay in the environment, e.g. for the editable install below. Environments created while `praeco` was pinned to a Git commit should be recreated from the updated file.

For local development, replace the installed package with an editable checkout
in the activated environment:

```bash
python -m pip install -e /path/to/praeco
python -m pip check
```

When migrating an existing environment, first remove the old editable package
with `python -m pip uninstall courier`. Restart any running notebook kernel
after changing the installation.

## Credentials and execution

Set `ONTODOCKER_ADDRESS`, `ONTODOCKER_TOKEN` (if required by the service), and
`DATAPORTAL_TOKEN` in the environment before starting Jupyter. Keep credentials
out of the notebook and version control.

The notebook combines local computation, remote queries, and operations that
create datasets and upload assets. Review the creation, publication, and cleanup
cells before executing them against a service.

## Tensile-test workflow notebook

`notebooks/tensile-test-workflow.ipynb` analyses the raw stress–strain curves of the ten S355 tensile tests with a `pyiron_workflow` workflow. It shows how semantic concepts separate the description of data from its analysis:

- Only the exploration steps (conformance check, specimen listing) and the acquisition steps (specimen metadata, curve location and units, lab-reported values) know an ontology. An ontology profile, selected by a workflow input, holds the ontology-specific queries and conventions.
- Acquisition delivers neutral records: numbers, arrays and units, without IRIs. The analysis steps are ontology-agnostic.
- The same workflow runs on the PMDco 2 graphs of this demonstrator (`pmdco2-tto`) and on the PMDco3/TTO version of the same tests (`pmdco3-tto`, a pinned graph from materialdigital/tensile-test-ontology). That graph does not link the per-test curves, so the notebook passes an explicit, non-semantic curve locator. For identical curves, the results are identical.
- `semantikon` exports the workflow recipes and runs as knowledge graphs, which the notebook saves as N-Triples, queries and validates.

The properties (E, ReH, ReL, Rp0.2, Fm, Rm, Ag, Agt, A, Z, with flags and uncertainty components) are method-defined and ISO 6892-1-inspired, not ISO-conformant. The comparison with the lab-reported values is a consistency check, not a validation.

The analysis code is the `tensile_workflow` package in this directory. The notebook imports it the same way as `query_collection.py`:

- `contract`: the neutral data records and unit conversion;
- `profiles`: the `pmdco2-tto` and `pmdco3-tto` ontology profiles;
- `sources`: Ontodocker access via `praeco`, local or remote Turtle via RDFLib, and the token-free curve download;
- `exploration`, `acquisition`: the ontology-aware workflow steps;
- `analysis`: the ontology-agnostic analysis steps;
- `workflows`: the two-stage workflow and `run_tensile_analysis`;
- `knowledge_graph`: knowledge-graph export, validation and example queries.

The tests in `tests/` run offline: they use the local Turtle files in `datasets/` and synthetic curves, and block outbound connections. `pytest` is not part of `environment.yml`; install it locally to run them from this directory:

```bash
python -m pytest tests -m "not network"
python -m pytest tests --run-statistical -m statistical  # Monte Carlo tolerance checks, slower
python -m pytest tests --run-network -m network  # reads the pinned PMDco3/TTO graph from GitHub
```

The notebook reads `ONTODOCKER_ADDRESS`, and `ONTODOCKER_TOKEN` if the service requires it, from the environment. They are needed only for the default Ontodocker source; the local Turtle files in `datasets/` are an alternative source, given relative to this directory. Credentials are never workflow inputs, so they cannot enter results or knowledge graphs. No Dataportal token is needed.

Each notebook cell that touches a service states its execution class. All cells are local (read-only) or network-read: SPARQL queries to Ontodocker, curve CSVs from `data.pmds-mpcdf-2.pyiron.com`, and, in the PMDco3 section, the pinned graph and curves from GitHub. The notebook creates, uploads or deletes no remote resources. Knowledge-graph files are written only to `kg_output_dir`, which must be outside the repository, or to a temporary directory. The code cell that sets the inputs is tagged `parameters` for papermill, e.g. from `notebooks/`: `papermill tensile-test-workflow.ipynb /tmp/tensile-test-workflow.ipynb`.

## Presenter preparation

The walkthrough follows five questions: what is available, what it means, what
belongs together, which metadata can be reused, and what others can access.
Allow 45 minutes for the notebook, following the 20-minute introduction.

The settings default to `reuse` for both Ontodocker and the Dataportal. This
inspects existing remote datasets while downloading and assembling local files.
For a new publication, set `publication_mode = "create"`, choose an unused
`publication_name`, review the metadata, and set `publication_reviewed = True`.
The Ontodocker mode is independent: creating a new publication can reuse the
existing research graphs. Creation stops if the chosen name already exists.

Confirm `owner_org` and the intended license. The current source RDF has an
ambiguous creator-name/ORCID mapping; the notebook flags it and retains names
without assigning those identifiers. The original identifiers remain in audit
metadata for review.

Before presenting, rehearse from a fresh kernel with the intended settings and
inspect the compact outputs. Retained outputs from earlier executions must be
refreshed after code changes. Keep a previously executed copy available as a
fallback, and leave cleanup cells commented out during the session.
