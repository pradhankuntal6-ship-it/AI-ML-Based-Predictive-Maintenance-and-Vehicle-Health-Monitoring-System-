# AI/ML-Based Predictive Maintenance and Vehicle Health Monitoring

A machine learning project that uses vehicle sensor data to estimate the chance of failure in the near future, track vehicle health, and explain the factors behind each prediction. It compares traditional machine learning and deep learning models on a synthetic vehicle dataset.

> **Data note:** The report describes a synthetic simulator, not real fleet data. Its very high test scores should not be treated as expected performance on real vehicles.

## Project goals

- Predict whether a vehicle is likely to fail within the next 40 time steps.
- Compare classical machine learning models with sequence models.
- Convert risk estimates into a Vehicle Health Score and an easy-to-read status.
- Explain predictions with SHAP feature attribution.
- Generate maintenance guidance from the predicted risk.

## How it works

1. **Generate sensor data:** A simulator creates time-series records for 80 vehicles (27,751 records and 12 columns). Most simulated vehicles gradually degrade and eventually fail; the remainder stay healthy.
2. **Prepare features:** Data is ordered by time and scaled. For each vehicle, the project derives rolling means, rolling standard deviations, and 10-step trends from sensor readings. These features help capture gradual changes rather than relying on a single measurement.
3. **Train and compare models:** Classical models use engineered tabular features. LSTM, GRU, and Transformer models use sequences of 30 consecutive readings.
4. **Estimate risk and explain it:** The selected XGBoost model estimates failure probability. SHAP identifies sensor features that push a prediction toward failure.
5. **Show vehicle status:** Risk is translated into NORMAL, WARNING, or CRITICAL status and a maintenance action.

## Sensors used

The nine input signals are:

- Engine temperature
- Engine RPM
- Oil pressure
- Vibration
- Battery voltage
- Coolant temperature
- Fuel consumption
- Vehicle speed
- Operating hours

The simulator increases vibration, engine and coolant temperature, and fuel consumption as failure approaches. Oil pressure and battery voltage decrease.

## Models compared

| Model | Approach |
| --- | --- |
| Logistic Regression | Linear baseline with class balancing |
| Decision Tree | Interpretable tree, depth 8 |
| Random Forest | 300 trees, maximum depth 12 |
| XGBoost | 300 boosted trees, maximum depth 5 |
| LSTM | Two recurrent layers, 64 hidden units |
| GRU | Two recurrent layers, 64 hidden units |
| Transformer | Two self-attention encoder layers |

The report selects XGBoost as the overall trade-off between precision and recall. The recurrent models achieved recall close to 1.0 in the reported experiment, with more false alarms. These findings are specific to the synthetic dataset and split described in the report.

## Vehicle health and alerts

The full health score combines predicted risk with deviation from an early-life healthy baseline:

```text
Vehicle Health (%) = 100 × [0.5 × (1 − failure probability) + 0.5 × sensor health]
```

The reported alert thresholds are:

| Failure probability | Status | Suggested action |
| --- | --- | --- |
| Below 35% | NORMAL | No maintenance action |
| 35% to 75% | WARNING | Schedule an inspection |
| Above 75% | CRITICAL | Stop the vehicle for immediate maintenance |

For XGBoost predictions, the system reports the top three sensor factors from SHAP. The report found vibration, battery voltage, engine temperature, and oil pressure among the most influential signals.

## Reported experiment

Vehicles were split by vehicle identity, with 75% used for training and 25% held out for testing. This avoids placing records from the same vehicle in both sets. The report lists 20,548 training rows and 7,203 test rows with 33 engineered features. Deep-learning models used 9,421 training sequences and 3,319 test sequences.

Approximate results from the report (rounded; see `model_comparison.csv` in the project outputs for exact run values):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.985 | 0.85 | 1.00 | 0.92 | 1.00 |
| Decision Tree | 0.988 | 0.88 | 0.99 | 0.93 | 0.99 |
| Random Forest | 0.988 | 0.92 | 0.96 | 0.94 | 1.00 |
| XGBoost | 0.992 | 0.93 | 0.98 | 0.95 | 1.00 |
| LSTM | 0.983 | 0.83 | 1.00 | 0.91 | 1.00 |
| GRU | 0.985 | 0.87 | 1.00 | 0.93 | 1.00 |
| Transformer | 0.988 | 0.90 | 0.98 | 0.94 | 1.00 |

These results are approximate values transcribed from the project report’s comparison chart. The report attributes the strong performance to the clear degradation patterns in the simulator. Validation on real, noisy fleet data is needed before drawing conclusions about operational performance.

## Technology

The report lists these tools and libraries:

- Python
- NumPy and Pandas
- scikit-learn
- XGBoost
- PyTorch
- SHAP
- Matplotlib
- Google Colab (development environment)

## Running the project

The report does not specify the repository’s notebook or script names, dependency file, or exact commands. Add the repository-specific instructions here, for example:

1. Clone this repository.
2. Create and activate a Python environment.
3. Install dependencies from the repository’s dependency file (such as `requirements.txt`, if included).
4. Run the data-generation and training notebook or script.
5. Review the model comparison, health monitoring output, and SHAP explanations.

Replace this section with the exact commands and file names from the repository before publishing, so readers can run the project without guessing.

## Limitations

- The dataset is synthetic, so performance may be lower on real vehicle data with noise and different failure modes.
- The prediction horizon is fixed at 40 time steps.
- The 35% and 75% alert thresholds and health-score weights were selected manually and need tuning against real maintenance costs.
- The simulator models one general failure label, not separate battery, engine, brake, or other component failures.
- The report notes that probability fluctuations near the warning threshold can cause alert flickering; smoothing or requiring the threshold to persist across multiple steps could help.

## Future work

- Evaluate on real fleet OBD-II/CAN-bus data or public datasets such as NASA C-MAPSS.
- Estimate Remaining Useful Life (RUL).
- Build a live dashboard with streaming sensor data.
- Add component-specific failure models and anomaly detection.
- Tune alert thresholds using maintenance costs and deploy a compressed model on edge hardware.

## Project report

This README is based on the project report titled **AI/ML-Based Predictive Maintenance and Vehicle Health Monitoring System**.
