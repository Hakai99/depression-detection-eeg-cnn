# Depression Detection Using EEG Signals with Deep Learning

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-orange)
![Flask](https://img.shields.io/badge/Flask-Web%20App-black)
![MNE](https://img.shields.io/badge/MNE--Python-EEG-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)
![Status](https://img.shields.io/badge/Status-Research%20Prototype-yellow)

An end-to-end EEG-based screening tool that classifies a resting-state EEG recording as **Healthy Control (HC)** or **Major Depressive Disorder (MDD)**. It combines an automated signal-processing pipeline, an EEGNet-inspired Convolutional Neural Network, and a Flask web app that runs locally in your browser.

> **Disclaimer:** This project is a research and academic prototype. It is **not** a medical device, has **not** been clinically validated, and must **not** be used as the sole basis for any clinical decision. If you are struggling with your mental health, please consult a qualified professional.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Demo](#demo)
- [How It Works](#how-it-works)
- [Model Architecture](#model-architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [Authors](#authors)
- [Acknowledgements](#acknowledgements)
- [License](#license)

---

## Overview

Depression is usually assessed with interviews and questionnaires such as PHQ-9 and the Beck Depression Inventory. These are subjective and depend on specialist availability. EEG records brain activity non-invasively and carries patterns linked to depression, such as frontal alpha asymmetry and elevated frontal theta power.

This project asks a simple question: can a lightweight CNN, working on standardised multichannel EEG windows, separate HC recordings from MDD recordings, and can it be packaged as a tool anyone can use from a browser? The main contribution is **integration**: a fully automated path from an `.edf` file to a result page, running on a normal CPU with no GPU.

## Features

- Upload a `.edf` EEG file (up to 100 MB) through a browser, with drag-and-drop
- Automatic preprocessing: channel selection, filtering, normalisation, windowing
- Compact EEGNet-inspired CNN, CPU-only inference in about 2-5 seconds
- Subject-level prediction by averaging window-level probabilities
- Result page with predicted class, confidence bar and probability breakdown
- Runs fully locally; uploaded recordings are not sent to any external server

## Demo

<!-- Replace these with your own screenshots. Put images in a docs/ or static/ folder. -->

| Home | Analyze | Result |
| --- | --- | --- |
| ![Home](docs/home.png) | ![Analyze](docs/analyze.png) | ![Result](docs/result.png) |

## How It Works

```
EDF Upload
   -> 19-channel selection (10-20 system)
   -> Band-pass filter (1-40 Hz)
   -> Z-score normalisation (per channel)
   -> 4-second windows, 50% overlap (1,024 samples at 256 Hz)
   -> Tensor of shape (N, 19, 1024, 1)
   -> CNN inference (per window)
   -> Mean of window probabilities
   -> HC / MDD + confidence score
```

1. **Load:** `mne.io.read_raw_edf()` reads the recording.
2. **Select channels:** Fp1, Fp2, F3, F4, F7, F8, Fz, C3, C4, Cz, P3, P4, Pz, O1, O2, T3, T4, T5, T6.
3. **Filter:** zero-phase FIR band-pass, 1-40 Hz, removes drift and muscle/line noise while keeping delta, theta, alpha and beta bands.
4. **Normalise:** per-channel Z-score removes amplitude differences between people.
5. **Window:** 4-second windows with a 2-second step.
6. **Predict:** the CNN outputs `[P(HC), P(MDD)]` for every window.
7. **Aggregate:** probabilities are averaged over all windows. The higher mean decides the class, and its value is the confidence.

## Model Architecture

```
Input (19, 1024, 1)
 -> Conv2D (8 filters, kernel 1x64)        temporal features
 -> BatchNorm
 -> DepthwiseConv2D (kernel 19x1)          spatial features across electrodes
 -> AvgPool (1x4) -> Dropout (0.5)
 -> SeparableConv2D (16 filters)
 -> BatchNorm
 -> AvgPool (1x8) -> Dropout (0.5)
 -> Flatten -> Dense (256, ReLU)
 -> Softmax (2)                            P(HC), P(MDD)
```

| Setting | Value |
| --- | --- |
| Optimizer | Adam (learning rate 0.001) |
| Loss | Categorical cross-entropy |
| Batch size | 32 |
| Max epochs | 100 |
| Early stopping | Patience 10 on `val_loss` |
| Saved model | `model/best_model_fixed.keras` |

<!-- TODO: run model.summary() and add the exact trainable parameter count here. -->

Based on: Lawhern et al., *EEGNet: A compact convolutional neural network for EEG-based brain-computer interfaces*, Journal of Neural Engineering, 2018.

## Tech Stack

| Component | Technology |
| --- | --- |
| Language | Python 3.10 |
| Web framework | Flask (Jinja2 templates) |
| Deep learning | TensorFlow 2.20 / Keras 3.12 |
| EEG processing | MNE-Python |
| Numerical computing | NumPy, SciPy |
| Frontend | HTML5, CSS3, JavaScript, Bootstrap 5 |

## Project Structure

```
.
├── app.py                      # Flask app: routes, upload, inference
├── preprocess.py               # EEG preprocessing pipeline
├── requirements.txt            # Python dependencies
├── test_model_load.py          # Quick check that the model loads
├── model/
│   └── best_model_fixed.keras  # Trained model
├── templates/                  # base, index, predict, result, about
├── static/                     # CSS, JS, images
├── utils/                      # Helper modules
└── uploads/                    # Runtime uploads (not tracked)
```

## Dataset

<!-- TODO: fill this in. Examiners and readers will look for it first. -->

- **Name:** [DATASET NAME]
- **Source:** [LINK]
- **Subjects:** [N] Healthy Controls, [N] MDD patients
- **Format:** EDF, resting-state, [recording length], 256 Hz
- **Channels:** 19 (10-20 system)

The dataset is **not** included in this repository. Download it from the source above and place the files in `Depression Detection dataset/` (or update the path in your code). Please follow the dataset's license and terms of use.

## Installation

**Requirements:** Python 3.10 and `pip`.

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# 2. Create and activate a virtual environment
python -m venv venv
# Windows
.\venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Check the Python version
python --version   # expected: Python 3.10.x
```

## Usage

```bash
python app.py
```

Open **http://127.0.0.1:5000** in your browser, then:

1. Click **Analyze an EEG Recording**.
2. Choose a `.edf` file (or drag and drop it).
3. Click **Upload & Analyze**.
4. Read the predicted class, confidence and probability breakdown.

**Input requirements:** `.edf` format, at least 15 of the 19 standard electrodes, at least 4 seconds of data. The sampling rate is handled automatically.

## Results

| Metric | Reported range |
| --- | --- |
| Accuracy | 87 - 90% |
| Sensitivity (recall) | 85 - 89% |
| Specificity | 88 - 92% |
| Precision | 86 - 91% |
| Inference time | 2 - 5 s per file (CPU) |

These figures come from our project evaluation and are **indicative, not definitive**. A formally documented held-out test set, subject-wise cross-validation and a full confusion matrix are still to be added.

<!-- TODO: add confusion matrix, test-set size, and how the data was split (by subject). -->

## Limitations

- Dataset size and demographics are limited, so generalisation is unconfirmed.
- No artefact rejection beyond band-pass filtering (no ICA, no epoch rejection).
- No explainability: the model does not show which EEG features drive a prediction.
- Binary classification only (HC vs MDD); no severity grading or other disorders.
- Not clinically validated against a gold-standard psychiatric diagnosis.
- Confidence is the mean window probability, not a calibrated certainty.

## Future Work

- Larger, multi-site datasets with subject-wise cross-validation
- Automated artefact removal (ICA)
- Explainable AI (Grad-CAM, Integrated Gradients)
- Multi-class detection (for example anxiety, bipolar disorder)
- Real-time streaming via Lab Streaming Layer
- Docker and cloud deployment
- PDF report generation and clinical validation

## Authors

- **Arnab Pratim Kashyap** (CS-BTA24-36)
- **Sania Parvin** (CS-BTA24-16)

**Guide:** Pranjit Sir, Department of Computer Science and Engineering
**Institution:** Birangana Sati Sadhani Rajyik Vishwavidyalaya, Golaghat, Assam
B.Tech in Computer Science and Engineering (AI & ML), 2025-2026

## Acknowledgements

- V. J. Lawhern et al., *EEGNet*, Journal of Neural Engineering, 2018
- A. Gramfort et al., *MNE software for processing MEG and EEG data*, NeuroImage, 2014
- R. J. Davidson, *Anterior cerebral asymmetry and the nature of emotion*, Brain and Cognition, 1992
- World Health Organization, depression and mental health estimates
- The MNE-Python, TensorFlow and Flask open-source communities

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
