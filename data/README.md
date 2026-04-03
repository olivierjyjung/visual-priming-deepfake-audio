# Data Availability Statement

## Overview

Due to privacy concerns related to participant data and Institutional Review Board (IRB) restrictions, the raw data cannot be made publicly available in this repository.

## Data Types

### 1. Behavioral Data (Survey Responses)

**Format:** CSV/Excel
**Contents:**
- False memory judgments (Q1)
- Confidence ratings (Q2, Q4)
- Audio source judgments (Q3)
- Perceptual factor ratings (Q5)
- Likeability ratings (Q6)
- Voice familiarity measures (Q7-Q9)
- Open-ended responses (Q10-Q11)

**Variables per observation:**
| Variable | Type | Description |
|----------|------|-------------|
| `participant_id` | string | Anonymized participant identifier |
| `group_number` | int | Experimental group (1-16) |
| `condition` | string | "Treatment" or "Control" |
| `clip_number` | int | Audio clip number (1-4) |
| `Q1_false_memory` | binary | 0 = No, 1 = Yes |
| `Q2_confidence` | int | 1-5 scale |
| `Q3_audio_judgment` | int | 1-6 options |
| `Q4_confidence` | int | 1-5 scale |
| `Q5_*` | int | Factor ratings (1-6 scale) |
| `Q6_likeability` | int | 1-5 scale |

### 2. EEG Data

**Format:** XDF (Lab Streaming Layer), BIDS-compliant directory structure
**Equipment:** DSI-24 wireless EEG headset
**Sampling Rate:** 300 Hz
**Channels:** 24 EEG channels

**Structure:**
```
eeg_data/
├── sub-001/
│   └── ses-S001/
│       └── eeg/
│           └── sub-001_ses-S001_task-audio_eeg.xdf
├── sub-002/
│   └── ...
```

**Protected under:** IRB Protocol [Number]

### 3. Stimuli

**Video stimuli:** K-pop idol performance clips
**Audio stimuli:**
- Real recordings: Authentic vocal performances
- AI-generated: Voice conversion using [method]

**Note:** Stimuli may be subject to copyright restrictions.

## Data Access

### Available Upon Request

Anonymized data supporting specific analyses reported in this article are available from the corresponding author upon reasonable request.

**To request data access:**

1. Email the corresponding author with:
   - Your institutional affiliation
   - Intended use of the data
   - Data protection measures in place

2. Complete a Data Use Agreement (DUA)

3. Receive IRB-approved anonymized dataset

### What Can Be Shared

| Data Type | Availability | Conditions |
|-----------|--------------|------------|
| Aggregated behavioral statistics | Available | Citation required |
| Anonymized individual-level behavioral data | Upon request | DUA required |
| EEG summary metrics | Upon request | DUA required |
| Raw EEG data | Restricted | IRB approval + DUA |
| Stimuli | Restricted | Copyright clearance |

## Example Data Format

For code testing purposes, an example data format file is provided:

```
data/example_data_format.csv
```

This file contains:
- Column headers matching the actual dataset
- Synthetic/random data (NOT real participant data)
- Correct data types and value ranges

## Reproducing Analyses

To reproduce the analyses with your own data:

1. Format your data according to `example_data_format.csv`
2. Place data files in the `data/` directory
3. Update file paths in analysis scripts
4. Run scripts as documented in the main README

## Contact

For data access requests:

**Corresponding Author:**
Olivier Jiyoun Jung
Division of Communication and Media
Ewha Womans University
Seoul, Korea

Email: [corresponding author email]

## Ethical Approval

This study was approved by the Institutional Review Board of Ewha Womans University (Protocol #[Number]).

All participants provided written informed consent prior to participation.
