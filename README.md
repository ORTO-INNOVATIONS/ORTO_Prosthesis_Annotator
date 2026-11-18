# ORTO Prosthesis Annotator

Local desktop application for semantic annotation of unified lower-limb prosthesis scans (tibia-level prostheses only).

**Current version:** 0.2.4 TEST

## Scope

- Annotate one unified STL surface as three semantic classes: socket, connector, and foot.
- Support complete scans and scans where parts of the connector are obscured by adjacent components.
- Preserve source mesh triangle identity; do not split or rewrite the source geometry during annotation.
- Export annotations as per-face labels tied to a source-mesh fingerprint.
- Build a reviewed dataset for a future local segmentation model. No model training or automatic predictions are included in the first milestone.

## Development status

The first milestone is the annotation workstation: STL import, 3D inspection, face painting, label correction, project save/load, and dataset export.

See [docs/REQUIREMENTS.md](docs/REQUIREMENTS.md), [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), and [docs/DATASET_FORMAT.md](docs/DATASET_FORMAT.md).

## Development

Python 3.11 is the target runtime. Install dependencies with:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .[dev]
python -m orto_annotator
```

## Data and privacy

Patient scans and annotation projects are local user data. Never commit STL scans, patient identifiers, annotation projects, model weights, or generated datasets to this repository. The application must not upload scan data or require cloud services.
