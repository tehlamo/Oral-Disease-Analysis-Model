# Oral Disease Detection Model

## Overview
This project uses a Faster R-CNN model to detect oral diseases from intraoral photos.

## Setup Instructions
1. Create a virtual environment: `python -m venv venv`
2. Install dependencies: `pip install -r requirements.txt`
3. Run data setup: `python smart_build.py` then `python create_folds.py`

## Datasets
The model was trained on a composite dataset aggregating approximately 6,600 images from the following specific sources:

1. **ORAL DETECTOR (Roboflow Universe)**
   * **Source:** [https://universe.roboflow.com/clients-mpvn2/oral-detector/dataset/3](https://universe.roboflow.com/clients-mpvn2/oral-detector/dataset/3)
   * **Credits:** User `clients-mpvn2` on Roboflow.
   * **Contribution:** Provided 4,131 images covering multiple disease classes (Gingivitis, Calculus, Ulcers, etc.).

2. **Annotated Intraoral Image Dataset for Dental Caries Detection (Zenodo)**
   * **Source:** [https://zenodo.org/records/14827784](https://zenodo.org/records/14827784)
   * **Authors:** Ahmed, Syed Muhammad Faizan; Ghori, Huzaifa; et al.
   * **DOI:** 10.5281/zenodo.14827784
   * **Contribution:** The primary source for the 'Caries' specific data subset.

3. **Healthy Teeth (Roboflow Universe)**
   * **Source:** [https://universe.roboflow.com/sultan-qyobm/healthy-teeth-hgddf/dataset/1](https://universe.roboflow.com/sultan-qyobm/healthy-teeth-hgddf/dataset/1)
   * **Credits:** User `sultan-qyobm` on Roboflow.
   * **Contribution:** Provided 235 healthy control images to prevent false positives.

## Logging & Auditing
The data processing scripts automatically generate execution logs to track data integrity:
- `logs/build_log.txt`: Records extraction counts and file origins.
- `logs/cleaning_log.txt`: Tracks class remapping and ghost label removal.
- `logs/splitting_log.txt`: documents the random seed and exact size of every K-Fold split.