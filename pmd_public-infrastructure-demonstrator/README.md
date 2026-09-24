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

The environment pins `praeco` to a Git commit. Its package requirements install
the required pandas, RDFLib, Pydantic, Requests, and SPARQLWrapper versions.

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
