# Visual Priming and Deepfake Audio Perception Study

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Analysis code for the paper: **"Calibrating the Ear: Prior AI Audio Exposure Reduces Susceptibility to Deepfake Vocals"**

## Overview

This repository contains the analysis code supporting the research findings on how visual priming and prior exposure to AI-generated audio affects:
- False memory formation for deepfake audio
- Deepfake audio detection abilities
- Neural processing of voice authenticity (EEG)

### Key Finding

Contrary to traditional priming theory predictions, visual priming with idol-related content **reduced** false memory rates:

| Condition | False Memory Rate | p-value |
|-----------|-------------------|---------|
| Treatment (Visual Priming) | 7.1% | |
| Control (Neutral) | 17.8% | **.030** |

This suggests a "calibration effect" where controlled exposure to synthetic media enhances authenticity discrimination.

## Repository Structure

```
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── scripts/
│   ├── analyze_behavioral.py    # Main behavioral data analysis
│   ├── eeg_analysis.py          # EEG preprocessing pipeline
│   ├── eeg_erp_analysis.py      # Event-related potential analysis
│   ├── eeg_frequency_analysis.py # Frequency band power analysis
│   ├── eeg_advanced_analysis.py  # ML classification & connectivity
│   ├── confidence_calibration.py # Confidence-accuracy calibration
│   └── order_effects_analysis.py # Robustness checks
├── data/
│   └── README.md                # Data availability statement
└── results/                     # Generated outputs (after running)
    ├── figures/
    └── reports/
```

## Installation

### Requirements

- Python 3.8 or higher
- Dependencies listed in `requirements.txt`

### Setup

```bash
# Clone the repository
git clone https://github.com/[username]/visual-priming-deepfake-audio.git
cd visual-priming-deepfake-audio

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Behavioral Analysis

Analyzes survey responses for false memory rates, audio source judgments, and perceptual factors.

```bash
python scripts/analyze_behavioral.py --data-dir /path/to/data --output-dir results/
```

**Key outputs:**
- False memory rate comparisons (Treatment vs. Control)
- Audio source judgment distributions (AI vs. Real)
- Confidence rating analysis
- Perceptual factor importance rankings

### EEG Analysis

#### 1. Preprocessing

```bash
python scripts/eeg_analysis.py --data-dir /path/to/eeg --output-dir results/
```

Applies standard preprocessing pipeline:
- Bandpass filtering (0.1-40 Hz)
- Notch filtering (60 Hz)
- Average re-referencing
- Bad channel detection

#### 2. ERP Analysis

```bash
python scripts/eeg_erp_analysis.py --data-dir /path/to/eeg --output-dir results/
```

Extracts event-related potential components:
- **N250** (200-300 ms): Voice familiarity processing
- **P300** (250-500 ms): Attention and evaluation
- **LPP** (400-700 ms): Late positive potential

#### 3. Frequency Analysis

```bash
python scripts/eeg_frequency_analysis.py --data-dir /path/to/eeg --output-dir results/
```

Computes power spectral density in frequency bands:
- Theta (4-8 Hz)
- Alpha (8-12 Hz)
- Beta (12-30 Hz)
- Gamma (30-40 Hz)

#### 4. Advanced Analysis

```bash
python scripts/eeg_advanced_analysis.py --data-dir /path/to/eeg --output-dir results/
```

Includes:
- Time-frequency analysis (ERSP, ITPC)
- Connectivity analysis (PLV, Coherence)
- Machine learning classification

### Statistical Robustness Checks

```bash
python scripts/order_effects_analysis.py --data-dir /path/to/data --output-dir results/
```

Tests for potential confounds:
- Clip presentation order effects
- First idol presentation order
- Audio type order (Real vs. AI first)

## Data Availability

Due to privacy concerns related to participant data and Institutional Review Board (IRB) restrictions, the raw data cannot be made publicly available.

**Available upon request:**
- Anonymized behavioral data
- Aggregated EEG metrics

Please contact the corresponding author for data access requests.

See `data/README.md` for detailed data structure documentation.

## Experimental Design

### Participants
- N = 54 female undergraduates
- Age: 19-29 years (M = 22.4, SD = 2.1)
- All K-pop fans with moderate-to-high idol familiarity

### Conditions
- **Treatment (Groups 1-8):** Visual priming with idol-related content
- **Control (Groups 9-16):** Neutral visual content

### Audio Stimuli
- **Real:** Authentic vocal recordings
- **AI-generated:** Voice conversion using deep learning

### Measures
1. False memory ("Have you seen this idol perform this song?")
2. Audio source judgment (AI vs. Real detection)
3. Confidence ratings (1-5 scale)
4. Perceptual factors (audio quality, breathing, intonation, etc.)
5. EEG during audio evaluation

## Key Results

### Behavioral Findings

| Measure | Treatment | Control | Statistics |
|---------|-----------|---------|------------|
| False Memory Rate | 7.1% | 17.8% | χ²(1) = 4.71, p = .030, V = .15 |
| AI Detection Rate | 47.3% | 41.6% | χ²(1) = 0.71, p = .399 |
| Confidence (Q4) | 3.42 | 3.38 | t(214) = 0.31, p = .759 |

### EEG Findings

| Component | Treatment | Control | Statistics |
|-----------|-----------|---------|------------|
| N250 Amplitude | -2.14 µV | -1.42 µV | t(46) = 2.39, p = .021, d = -0.69 |
| Frontal N2 Latency | 198 ms | 214 ms | t(46) = 2.19, p = .033, d = -0.64 |
| ML Classification | - | - | AUC = 0.64 |

## Citation

If you use this code in your research, please cite:

```bibtex
@article{jung2026calibrating,
  title={Calibrating the Ear: Prior AI Audio Exposure Reduces Susceptibility to Deepfake Vocals},
  author={Jung, Olivier Jiyoun},
  journal={[Journal Name]},
  year={2026},
  doi={[DOI]}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

**Corresponding Author:**
Olivier Jiyoun Jung
Division of Communication and Media
Ewha Womans University, Seoul, Korea
Email: [email]

## Acknowledgments

- Participants who contributed to this study
- [Funding source, if applicable]
- MNE-Python developers for EEG analysis tools
