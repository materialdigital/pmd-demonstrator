# PMD Mesh Demonstrator

This demonstrator shows how semantically described materials data can be
distributed, discovered, queried, and analyzed across services connected through
the PMD Mesh.

The current example centers on S355 tensile-test data represented with PMD
semantic vocabularies. The notebooks demonstrate how to split RDF data into
mesh-distributed datasets, upload/query those datasets through Ontodocker, and
use the resulting semantic metadata to find and analyze raw test data.

## What This Demonstrator Covers

- discovery of PMD Mesh participants and their registered services
- construction of a local participant/service registry
- token handling for protected service APIs
- Ontodocker dataset creation, upload, query, and deletion
- federated SPARQL queries over multiple Ontodocker instances
- semantic exploration of S355 tensile-test data
- retrieval and augmentation of raw CSV resources
- force-displacement and stress-strain analysis
- a preview of a future `pyiron_workflow` implementation of the same analysis

Most of the explanation is currently inside the notebooks. This README is meant
as an entry point and run guide.

## Requirements

You need:

- access to the PMD Mesh network for the mesh-based notebooks
- access to the internal mesh listing service
- valid service tokens for protected Ontodocker or CKAN instances
- a working conda/mamba installation
- Jupyter or JupyterLab

The environment file assumes Python 3.11 or newer and installs the notebook
dependencies, including `rdflib`, `SPARQLWrapper`, `pandas`, `scipy`,
`matplotlib`, `pyiron_workflow`, and `pyironflow`.

## Setup

From this directory:

```bash
mamba env create -n pmd-demonstrator -f environment.yml
conda activate pmd-demonstrator
python -m pip install -e pmd_demo_tools
```

`conda` can be used instead of `mamba` if needed.

Some notebooks set `REQUESTS_CA_BUNDLE` explicitly for HTTPS requests. Depending
on the execution environment, you may need to adapt certificate handling for PMD
Mesh services.

## Data and Assets

The demonstrator includes RDF/Turtle data in [`datasets/`](datasets/):

- `test_dataset.ttl`: a small example dataset used for basic Ontodocker upload
  and SPARQL-query tests
- `pmdco2_tto_example.ttl`: the full S355 tensile-test example graph
- `pmdco2_tto_example_parallel.ttl`: S355 slice for specimens cut parallel to
  the rolling direction
- `pmdco2_tto_example_perpendicular.ttl`: S355 slice for specimens cut
  perpendicular to the rolling direction
- `pmdco2_tto_example_diagonal.ttl`: S355 slice for specimens cut diagonal to
  the rolling direction

The notebooks also reference raw CSV resources through URLs stored in the RDF
data. The generated overview figure is stored as
[`notebooks/S355_stress-strain-overview.png`](notebooks/S355_stress-strain-overview.png).

## Helper Package

The local package [`pmd_demo_tools`](pmd_demo_tools/) contains reusable notebook
support code:

- `mesh_tools`: fetches PMD Mesh participant information, groups it by company
  and server, attaches service listings, filters registries, and attaches tokens
- `sparql_tools`: lists Ontodocker SPARQL endpoints, sends SPARQL queries, and
  runs federated queries over matching mesh services
- `query_collection`: contains predefined SPARQL queries for the small test
  dataset and the S355 tensile-test example

The package is intentionally lightweight and demonstrator-specific. It should be
installed in editable mode while working on the notebooks.

## Recommended Notebook Order

The notebooks are numbered, but the most useful reading order starts with the
infrastructure overview:

| Notebook | Role | Notes |
| --- | --- | --- |
| [`01-pmd-mesh.ipynb`](notebooks/01-pmd-mesh.ipynb) | PMD server, PMD Mesh, registry, services, and helper-package basics | Includes an intentional duplicate-key example when a flat company grouping is too coarse. |
| [`02-ontodocker.ipynb`](notebooks/02-ontodocker.ipynb) | Ontodocker API usage from notebooks | Demonstrates real dataset creation, upload, query, and deletion. |
| [`00-semantics_via_python.ipynb`](notebooks/00-semantics_via_python.ipynb) | RDF graph manipulation and federated upload | Splits the full S355 graph into orientation-specific datasets and uploads them to mesh services. |
| [`03-data_exploration.ipynb`](notebooks/03-data_exploration.ipynb) | Main semantic data exploration | Queries available datasets, explores S355 metadata, resolves raw CSV resources, and plots force-displacement curves. |
| [`04-stress_strain_analysis.ipynb`](notebooks/04-stress_strain_analysis.ipynb) | Stress-strain analysis | Builds on the semantic query results, computes stress/strain quantities, estimates elastic modulus and 0.2 percent offset yield strength, and saves the overview figure. |
| [`05-pyiron_workflow.ipynb`](notebooks/05-pyiron_workflow.ipynb) | Workflow preview | Early preview of expressing the analysis from notebook `04` as a `pyiron_workflow` workflow. It currently focuses on node definitions and workflow structure rather than a finished end-to-end execution. |

## Operations That Modify Remote Services

Some notebooks are not read-only:

- `02-ontodocker.ipynb` creates, fills, queries, and deletes Ontodocker datasets.
- `00-semantics_via_python.ipynb` uploads S355 dataset slices to selected
  Ontodocker instances.

Check service names, dataset names, and tokens before running these notebooks
against shared infrastructure.

## Secrets and Tokens

Service tokens are not part of the repository. The notebooks expect tokens to be
provided locally, for example through a JSON file under `secrets/`. The exact
token structure used in the notebooks follows the local participant/service
registry so tokens can be attached to discovered services.

Do not commit token files or generated registries containing credentials. The
repository `.gitignore` excludes common local secret files such as `secrets/`,
`tokens.json`, `token.txt`, `endpoints.json`, and `partners.json`.

## Expected Warnings and Limitations

The notebooks query live PMD Mesh services. Outputs may differ from the saved
notebook state if services, datasets, tokens, or network routes change.

Some warnings are expected when a mesh participant or service is unreachable
from the current environment. The registry helpers can keep going and mark such
services as unavailable.

Several notebooks contain mesh-internal hostnames and outputs from a specific
execution environment. They are useful as examples, but should not be treated as
stable public endpoints.

Notebook `05-pyiron_workflow.ipynb` is intentionally a preview. The goal is to
show the same analysis as `04-stress_strain_analysis.ipynb`, but represented
with `pyiron_workflow` as the workflow manager.

## Relation to Future Demonstrators

This demonstrator assumes PMD Mesh-local service discovery and mostly internal
service URLs. A future demonstrator using publicly reachable services can reuse
the semantic data, SPARQL-query, and analysis ideas while replacing the mesh
registry and private-network assumptions with explicit public service
configuration.
