# PMD Demonstrators

This repository collects technical demonstrators for the Platform MaterialDigital
(PMD). The demonstrators are intended to show concrete usage patterns for PMD
infrastructure, semantic data, notebooks, and supporting Python tooling.

At the moment, the repository contains one demonstrator:

| Demonstrator | Status | Main focus | Main requirements |
| --- | --- | --- | --- |
| [`pmd_mesh-demonstrator`](pmd_mesh-demonstrator/) | active prototype | PMD Mesh service discovery, Ontodocker, federated SPARQL queries, and S355 tensile-test data exploration | PMD Mesh access, suitable service credentials, conda/mamba |

## Repository Layout

```text
.
├── README.md
└── pmd_mesh-demonstrator/
    ├── README.md
    ├── environment.yml
    ├── datasets/
    ├── notebooks/
    └── pmd_demo_tools/
```

Each demonstrator should be self-contained enough to explain:

- what infrastructure or services it assumes
- how to create the required software environment
- which notebooks or scripts are meant to be run, and in which order
- which operations are read-only and which modify remote services
- how credentials, tokens, or private endpoints are handled

Common repository-level information belongs here. Demonstrator-specific setup,
service assumptions, notebook descriptions, and troubleshooting belong in the
README inside the corresponding demonstrator directory.

## Usage Model

The current demonstrator is notebook-first. The notebooks contain the main
explanation and show executed examples. Python packages inside demonstrator
directories provide reusable helper code for service discovery, querying, and
analysis.

For setup and execution details, start with the demonstrator README:

- [`pmd_mesh-demonstrator/README.md`](pmd_mesh-demonstrator/README.md)

## Planned Extensions

The repository is expected to grow beyond the PMD Mesh demonstrator. Likely next
steps include:

- a demonstrator using publicly reachable services instead of services located
  inside the PMD Mesh
- Docker images pre-populated with this repository, or selected parts of it, for
  use in JupyterHub environments
- smaller images or examples that contain only one demonstrator and its required
  datasets/tools

These extensions should keep the same separation of concerns: the root README
describes the collection, while each demonstrator explains its own assumptions
and execution path.

## Credentials and Private Infrastructure

Some demonstrators may require private network access, service-specific tokens,
or internal certificates. Credentials and generated local state should not be
committed. The current repository already excludes common token and secret files
through `.gitignore`.
