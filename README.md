# Sepsis-early-detection

## Overview

**SepsisDefender** is a biomedical engineering project developed during a summer internship in the Medical Intensive Care Unit of M’hamed Bourguiba Regional Hospital in Kef, Tunisia.

The project aims to support the early detection of sepsis by combining real-time physiological data acquisition, machine learning, and a monitoring application. It uses an XGBoost classification model trained on the PhysioNet/Computing in Cardiology Challenge 2019 dataset to estimate the risk of sepsis and generate early warnings.

## Objectives

- Acquire physiological parameters from an EDAN multiparameter patient monitor through an Ethernet connection.
- Process and prepare clinical time-series data for machine learning.
- Develop an XGBoost model to estimate the risk of sepsis.
- Build a monitoring application to display physiological parameters, prediction results, and early warnings.
- Explore the potential of artificial intelligence to support clinical monitoring in intensive care units.

## System Architecture

The system follows a processing pipeline:

1. **Data acquisition:** Physiological measurements are collected from the EDAN patient monitor through Ethernet communication.
2. **Data preprocessing:** Clinical data are prepared and organized into the format required by the prediction model.
3. **Machine learning:** The trained XGBoost model estimates the probability of sepsis.
4. **Risk assessment:** A decision threshold is applied to identify patients requiring further clinical evaluation.
5. **Visualization and alerts:** The SepsisDefender application displays patient parameters, estimated risk, and early warnings.

## Technologies and Tools

- **Programming language:** Python
- **Machine learning:** XGBoost
- **Clinical dataset:** PhysioNet/Computing in Cardiology Challenge 2019
- **Data acquisition:** Ethernet communication with an EDAN patient monitor
- **Application:** SepsisDefender monitoring and alert interface
- **Domain:** Biomedical engineering, physiological signal and data processing, artificial intelligence, and intensive care monitoring

## Results

During model evaluation, the selected decision threshold was **0.3854**. The model detected **96.1% of septic patients**, demonstrating promising sensitivity for early risk detection.

However, the system also produced a high false-alarm rate of **69.1% per patient**, highlighting the need for further optimization before any clinical deployment.

## Limitations and Future Work

The current system requires further development and validation. The main limitations include missing values and differences in measurement frequencies in the training dataset, the need for validation using local hospital data, and the high number of false alarms.

Future improvements include optimizing the decision threshold, improving data preprocessing, comparing alternative machine learning models, and strengthening the reliability of real-time data acquisition.

## Disclaimer

SepsisDefender is an educational and research prototype developed as part of a biomedical engineering internship. It is **not a certified medical device** and is not intended to replace clinical judgment or professional medical diagnosis. Clinical validation is required before any real-world medical use.

## Author

**Abir Ounifi**
Biomedical Engineering Student
Institut National des Technologies et des Sciences du Kef (INTeK), University of Jendouba, Tunisia

**Project period:** June–July 2026

