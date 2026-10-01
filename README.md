# Hotel Booking Cancellation Prediction

Predicting which hotel bookings will be cancelled, and profiling the guests who cancel, using Random Forest, Logistic Regression and K-Means clustering on ~120K reservations from two Portuguese hotels.

## Problem

Cancellations cause empty rooms, lost revenue and poor demand forecasts. This project asks:

1. Can booking information predict whether a reservation will be cancelled?
2. Among cancelled bookings, are there distinct guest "risk personas"?

## Key Results

Evaluated on a held-out, stratified 20% test set (17,536 bookings, 4,860 of them cancelled). Both models were tuned for F1 because only 27.7% of bookings are cancelled.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| **Random Forest** (tuned) | **79.43%** | **60.56%** | 73.87% | **66.56%** | **86.43%** |
| Logistic Regression (tuned) | 70.11% | 47.60% | **77.72%** | 59.04% | 81.53% |

- **Random Forest is the stronger model overall.** It catches about 74% of actual cancellations (3,590 of 4,860) and is right about 61% of the time when it raises a flag.
- **Logistic Regression flags more cancellations (higher recall) but with many more false alarms** (4,158 vs 2,338 false positives), so the better choice depends on the relative cost of a missed cancellation vs unnecessary outreach.
- **Lead time is the strongest signal.** The cancellation rate rises from 8.4% for bookings made within 7 days to 38.4% for bookings made more than 150 days ahead, and `lead_time` is the top Random Forest feature.
- **Logistic Regression coefficients** show Non Refund deposits, previous cancellations and Transient customers raising cancellation risk, and parking requests and Offline TA/TO bookings lowering it.
- **K-Means found four profiles among cancelled bookings** (see below), with modest cluster separation (silhouette 0.29).

## Dataset

- **Source:** Course-provided "Dataset C: Hotel Bookings" (COMP5310). The raw file is included in this repo.
- **Size:** 119,987 rows, 32 columns; City Hotel and Resort Hotel, July 2015 to August 2017.
- **Target:** `is_canceled` (0 = not cancelled, 1 = cancelled).
- **After cleaning:** 87,678 rows, 63,381 not cancelled (72.3%) and 24,297 cancelled (27.7%).

## Approach

**1. Data cleaning and EDA** (`Data_Cleaning.ipynb`)

- **Missing values:** dropped `company` (94% missing); filled `agent` with 0 (treated as a direct booking), `children` with 0, `meal` with "Undefined", `country` with "None", and the remaining small gaps with the mode.
- **Inconsistent values:** fixed spelling and whitespace variants (`CityHotel`, `Direc`, `bb`, `HB `), 955 reservation dates in mixed formats, and one negative ADR (set to 0).
- **Invalid rows:** removed 180 bookings with zero guests (119,987 → 119,807), then 32,129 exact duplicates (→ 87,678).
- **Feature engineering:** `total_stay_nights`, `total_guests`, `is_family`, `stay_category`, `room_type_changed`, and `is_direct_booking` (replacing `agent`).
- **Leakage and noise removal:** dropped `reservation_status` and `reservation_status_date` (they encode the outcome) and `country` (178 values).
- **EDA:** correlations, cancellation rate by lead-time quintile, ADR by cancellation status, cancellations by hotel and deposit type.

**2. Classification** (`Data_Modelling_and_Evaluation.ipynb`)

- 80/20 stratified split, `random_state=42`, `class_weight='balanced'` for both models, 3-fold stratified CV scored on F1.
- **Random Forest:** ordinal-encoded categoricals, 30 features. Grid search over `n_estimators`, `max_depth` and `min_samples_split` on a 25,000-row training subsample, then refit on the full 70,142-row training set. Best: `n_estimators=200`, `max_depth=20`, `min_samples_split=5` (CV F1 0.635). Fully grown trees reached a training F1 of about 0.997 with lower CV F1, so depth and split size were constrained.
- **Logistic Regression:** one-hot encoded (62 features), numeric features standardised with a scaler fitted on training data only. Grid search over `C` and `penalty` on the full training set. Best: `C=10`, `penalty='l1'` (CV F1 0.592).

**3. Segmentation with K-Means**

On the 24,297 cancelled bookings only: ordinal encoding, 99th-percentile capping of ADR, lead time and days on the waiting list, then standardisation (27 features). `k` was chosen from 2 to 8 using inertia and silhouette score (best at k = 4, silhouette 0.2943). K-means++ and random initialisation, with `n_init` of 10, 20 and 50, all gave identical results.

| Persona | Bookings | Avg lead time (days) | Avg ADR | Special requests | Prior cancellations | Direct bookings |
|---|---:|---:|---:|---:|---:|---:|
| Standard Risk | 17,607 (72.5%) | 105 | 113.5 | 0.58 | 0.04 | 0.00 |
| Premium Bookers | 3,058 (12.6%) | 113 | 165.7 | 0.57 | 0.02 | 0.01 |
| Last-Minute Cancellers | 2,452 (10.1%) | 60 | 101.4 | 0.35 | 0.13 | 0.57 |
| Extreme Planners | 1,180 (4.9%) | 226 | 94.3 | 0.02 | 0.46 | 0.12 |

Persona names are interpretive labels based on each cluster's average profile.

## Limitations and Future Work

- **Some predictors may not be known at booking time.** `room_type_changed` (the second most important Random Forest feature), `required_car_parking_spaces` (the largest Logistic Regression coefficient), `booking_changes` and `assigned_room_type` can reflect events after the booking is made, so the reported scores may be optimistic for a true at-booking-time model. Re-running without them is the obvious next test.
- **Random, not time-based, split.** The data covers 2015 to 2017 and two hotels in Portugal, so results may not generalise to other periods or markets.
- **Deposit effects are counter-intuitive.** Non Refund bookings show a very high cancellation rate in this data, so deposit-policy conclusions should not be drawn from it without more context.
- **Duplicate removal** assumed identical rows are errors, but some may be genuine separate bookings (32,129 rows, about 27%, were removed).
- **Random Forest importance** (mean decrease in impurity) favours high-cardinality numeric features; permutation importance would be less biased.
- **Clustering** covers cancelled bookings only and separation is modest, so personas describe who cancels rather than predict cancellation.
- **Possible next steps:** gradient boosting, decision-threshold tuning against business costs, permutation importance, and a time-based validation split.

## Repository Structure

```text
Hotel-Booking-Cancellation-Prediction/
├── Data_Cleaning.ipynb                  # Cleaning, feature engineering, EDA
├── Data_Modelling_and_Evaluation.ipynb  # Random Forest, Logistic Regression, K-Means
├── Lab_08_Group_09_Report.pdf           # Full written report
└── README.md
```

## How to Run

```bash
git clone https://github.com/hemangijoshi12/Hotel-Booking-Cancellation-Prediction.git
cd Hotel-Booking-Cancellation-Prediction
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
jupyter notebook
```

1. Place the raw dataset in the repo root as `hotel_bookings.csv`.
2. Run `Data_Cleaning.ipynb`. It writes the cleaned data to `dataset_c.csv`.
3. Run `Data_Modelling_and_Evaluation.ipynb`, which reads `dataset_c.csv`.

**Tech stack:** Python, pandas, NumPy, scikit-learn, Matplotlib, Seaborn, Jupyter

## Report and Notes

- Full methodology and discussion: [`Report.pdf`](Report.pdf).
