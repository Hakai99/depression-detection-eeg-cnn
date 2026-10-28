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
- [Dataset](#dataset)
- [How It Works](#how-it-works)
- [Model Architecture](#model-architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Troubleshooting](#troubleshooting)
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

## Dataset

**Source:** [Raw 19-Channel EEG Data — Figshare](https://figshare.com/articles/Raw_19_channel_EEG_Data/1601695)  
**Channels:** 19 (standard 10-20 system: Fp1, F3, C3, P3, O1, F7, T3, T5, Fz, Fp2, F4, C4, P4, O2, F8, T4, T6, Cz, Pz)  
**Format:** EDF  
**Task:** Resting-state EEG (eyes closed)  
**Classes:** Healthy Control (HC) vs Major Depressive Disorder (MDD)  

> The dataset is **not** included in this repository due to file size and data-sharing terms.  
> Download it from the Figshare link above and follow its license.
>
> - **To just try the app:** download any one `.edf` file from the dataset and upload it in the web interface. No dataset folder is needed.
> - **To retrain or re-run preprocessing:** place the `.edf` files inside a `dataset/` folder in the project root.

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

## Installation

### Prerequisites

- [Git](https://git-scm.com/downloads)
- [Python 3.10](https://www.python.org/downloads/) (TensorFlow 2.20 may not install on other versions)
- A 19-channel `.edf` EEG file to test with (see [Dataset](#dataset))

Check your Python version first:

```bash
python --version   # expected: Python 3.10.x
```

### Windows

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

If `python` is not version 3.10, create the environment with `py -3.10 -m venv venv` instead.

### macOS / Linux

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
```

Replace `<your-username>` and `<repo-name>` with the real GitHub username and repository name.

## Usage

After running `python app.py`, open **http://127.0.0.1:5000** in your browser, then:

1. Click **Analyze an EEG Recording**.
2. Choose a `.edf` file (or drag and drop it).
3. Click **Upload & Analyze**.
4. Read the predicted class, confidence and probability breakdown.

**Input requirements:** `.edf` format, at least 15 of the 19 standard electrodes, at least 4 seconds of data. The sampling rate is handled automatically.

**Next time:** you only need to activate the environment and start the app:

```bash
# Windows
.\venv\Scripts\activate
python app.py

# macOS / Linux
source venv/bin/activate
python app.py
```

Press `Ctrl + C` in the terminal to stop the server.

## Troubleshooting

| Problem | Likely cause | Fix |
| --- | --- | --- |
| `pip install` fails on TensorFlow | Unsupported Python version | Use Python 3.10 |
| `ModuleNotFoundError` | Virtual environment not activated, or install incomplete | Activate `venv`, then run `pip install -r requirements.txt` again |
| Model file not found | Running from the wrong folder | Run `python app.py` from the project root |
| Upload error or file rejected | Not a valid `.edf`, fewer than 15 of the 19 channels, or under 4 seconds | Use a 19-channel `.edf` recording of at least 4 seconds |
| Port 5000 already in use | Another app is using it | Close the other app, or change the port in `app.py` |

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

- **Dev Kumar Lahkar**

## Acknowledgements

- V. J. Lawhern et al., *EEGNet*, Journal of Neural Engineering, 2018
- A. Gramfort et al., *MNE software for processing MEG and EEG data*, NeuroImage, 2014
- R. J. Davidson, *Anterior cerebral asymmetry and the nature of emotion*, Brain and Cognition, 1992
- World Health Organization, depression and mental health estimates
- The MNE-Python, TensorFlow and Flask open-source communities

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
