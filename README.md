# README.md

# MediScan — AI Medical Analysis Suite

A modern 4-module AI medical web application for disease prediction, fracture detection, blood analysis, and skin disease recognition.

## Modules

| Tab                | Model                                  | Input                                | Output                                    |
| ------------------ | -------------------------------------- | ------------------------------------ | ----------------------------------------- |
| 🔬 Symptom Checker | Random Forest (scikit-learn)           | 132 symptoms                         | 41 diseases                               |
| 🦴 Bone Fracture   | MobileNet CNN (Keras/TensorFlow)       | X-ray image 224×224                  | Normal / Fracture                         |
| 🩸 Blood Analysis  | DistilBERT (Hugging Face Transformers) | 9 blood biomarkers converted to text | 5 conditions                              |
| 🩹 Skin Disease    | CNN (Keras/TensorFlow)                 | Skin photo 224×224                   | Top 5 predictions from 23 skin conditions |

## Features

* Symptom-based disease prediction
* Bone fracture detection from X-rays
* Blood report analysis using Hugging Face Transformers
* Skin disease prediction from uploaded images
* REST API backend with Flask
* JSON-based responses for easy frontend integration

## Installation

```bash
git clone <your-repo-url>
cd <your-project-folder>

pip install -r requirements.txt
```

## Running the App

```bash
python app.py
```

Open in browser:

```text
http://localhost:3000
```

## Project Structure

```text
project/
├── app.py
├── assets/
├── static/
│   └── index.html
├── Disease/
│   ├── disease_model.pkl
│   └── symptoms_list.pkl
├── Bone/
│   └── keras_model.h5
├── Blood/
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   ├── special_tokens_map.json
│   └── vocab.txt
├── Skin/
│   └── keras_smodel.h5
├── requirements.txt
└── README.md
```

## API Endpoints

### Status

```http
GET /api/status
```

### Symptoms List

```http
GET /api/symptoms
```

### Predict Disease from Symptoms

```http
POST /api/predict/symptoms
```

### Predict Bone Fracture

```http
POST /api/predict/fracture
```

### Blood Feature Metadata

```http
GET /api/blood-features
```

### Predict Blood Condition

```http
POST /api/predict/diagnostic
```

### Predict Skin Disease

```http
POST /api/predict/skin-disease
```
