# 🏨 Hotel Booking Cancellation Prediction

An end-to-end machine learning project for **predicting hotel booking cancellations and identifying customer booking segments** using historical hotel reservation data.

The project covers the complete data science workflow: **data cleaning, exploratory data analysis, feature engineering, leakage prevention, supervised classification, hyperparameter tuning, model evaluation, and unsupervised customer segmentation**.

---

## 📌 Problem Statement

Hotel booking cancellations can affect room availability, revenue forecasting, and operational planning.

This project investigates whether historical booking information can be used to predict whether a reservation will be cancelled and whether cancelled bookings can be further segmented into meaningful customer groups.

### Objectives

1. Clean and preprocess real-world hotel booking data.
2. Identify patterns associated with booking cancellations.
3. Engineer features that better represent booking behaviour.
4. Build and tune classification models for cancellation prediction.
5. Compare model performance using multiple evaluation metrics.
6. Use K-Means clustering to identify distinct booking/customer segments.

---

## 📊 Dataset

The project uses the **Hotel Booking Demand** dataset containing reservations from:

* **City Hotel**
* **Resort Hotel**

The original dataset contains **119,987 records and 32 attributes**. The target variable is:

```text
is_canceled
```

| Value | Meaning                   |
| ----- | ------------------------- |
| `0`   | Booking was not cancelled |
| `1`   | Booking was cancelled     |

The dataset contains information about:

* Hotel type
* Booking lead time
* Arrival dates
* Length of stay
* Number of guests
* Meal type
* Market segment
* Distribution channel
* Previous cancellations
* Previous bookings
* Room types
* Deposit type
* Customer type
* Average Daily Rate (ADR)
* Special requests
* Parking requirements

---

# 🔄 Project Pipeline

```text
Raw Hotel Booking Data
        │
        ▼
Data Quality Assessment
        │
        ▼
Missing Value Treatment
        │
        ▼
Categorical Value Normalisation
        │
        ▼
Date & Data Type Cleaning
        │
        ▼
Duplicate Removal
        │
        ▼
Feature Engineering
        │
        ▼
Leakage Prevention
        │
        ├───────────────┐
        ▼               ▼
Classification      Customer Segmentation
        │               │
        ▼               ▼
Random Forest       K-Means
Logistic Regression
        │
        ▼
Model Evaluation
```

---

# 🧹 Data Cleaning

The `Data_Cleaning.ipynb` notebook performs detailed data-quality analysis and preprocessing.

### Missing values

The original dataset contained missing values in several attributes:

| Feature                | Missing |
| ---------------------- | ------: |
| `children`             |     182 |
| `meal`                 |     138 |
| `country`              |     628 |
| `market_segment`       |     129 |
| `distribution_channel` |     164 |
| `agent`                |  16,541 |
| `company`              | 113,162 |
| `customer_type`        |     154 |

Different imputation strategies were applied depending on the meaning and scale of the missing data.

Examples:

* `children` → filled with `0`
* `agent` → filled with `0`, representing direct booking
* `country` → filled with `"None"`
* `meal` → filled with `"Undefined"`
* `market_segment`, `distribution_channel`, and `customer_type` → filled using their respective modes.

### Categorical normalisation

Categorical variables contained inconsistent formatting such as:

```text
City Hotel
CityHotel

groups
Groups

Direct
Direc

bb
BB
```

String normalisation was applied by stripping whitespace and standardising categorical representations before modelling.

---

# 🧠 Leakage Prevention

Two variables were removed because they contain information about the booking outcome and would not be appropriate predictors when making a cancellation prediction:

```text
reservation_status
reservation_status_date
```

The project also removed `country` because of its high cardinality and limited reliability as a booking-time predictor.

The `agent` variable was transformed into:

```text
is_direct_booking
```

where:

```text
1 → direct booking
0 → booking through an agent
```

The original `agent` column was then removed.

---

# 🔧 Feature Engineering

Several behavioural features were created to provide more useful representations of the bookings.

### `total_stay_nights`

```python
total_stay_nights =
    stays_in_weekend_nights + stays_in_week_nights
```

### `total_guests`

```python
total_guests =
    adults + children + babies
```

### `is_family`

A binary feature indicating whether children or babies are included in the booking.

### `stay_category`

Bookings are categorised as:

* `Mixed`
* `Weekend Only`
* `Weekday Only`

### `room_type_changed`

Indicates whether the assigned room differs from the originally reserved room.

These engineered features were designed to capture booking behaviour that is not directly represented by individual raw columns.

---

# 🗑️ Duplicate Removal

After cleaning, the dataset contained:

```text
119,807 rows × 31 columns
```

A total of:

```text
32,129 duplicate rows
```

were identified and removed.

The final modelling dataset contains:

```text
87,678 rows × 31 columns
```

with:

* **63,381 non-cancelled bookings — 72.3%**
* **24,297 cancelled bookings — 27.7%**

---

# 📈 Exploratory Data Analysis

The project explores several relationships between booking characteristics and cancellation behaviour.

### Analyses include

* Correlation analysis of numerical variables
* Cancellation probability across lead-time quantiles
* ADR distribution by cancellation status
* Cancellation counts by hotel type
* Cancellation rate by deposit type

For example, the notebook specifically analyses how **lead time** relates to cancellation probability and examines the relationship between **ADR and cancellation status**.

---

# 🤖 Supervised Machine Learning

Two classification approaches were developed:

1. **Random Forest**
2. **Logistic Regression**

Because the dataset has an approximately **72.3% / 27.7% class distribution**, F1-score was used as the primary tuning metric rather than relying solely on accuracy.

Both models use an **80/20 stratified train-test split** with `random_state=42`.

---

## 🌲 Random Forest

Categorical variables were transformed using `OrdinalEncoder`.

```python
OrdinalEncoder(
    handle_unknown='use_encoded_value',
    unknown_value=-1
)
```

The model used:

```python
class_weight='balanced'
max_features='sqrt'
random_state=42
n_jobs=-1
```

### Hyperparameter tuning

GridSearchCV was performed using a **25,000-row stratified subsample** of the training data and **3-fold StratifiedKFold cross-validation**.

Search space:

```python
n_estimators:
    [100, 200]

max_depth:
    [10, 20, None]

min_samples_split:
    [2, 5]
```

The search was optimised for:

```text
F1 Score
```

### Best parameters

```python
{
    'n_estimators': 200,
    'max_depth': 20,
    'min_samples_split': 5
}
```

The best cross-validation F1 score was approximately **0.6353**. The tuned model was subsequently refitted on the full training set.

### Test performance

| Metric    |      Score |
| --------- | ---------: |
| Accuracy  | **79.43%** |
| Precision | **60.56%** |
| Recall    | **73.87%** |
| F1 Score  | **66.56%** |
| ROC-AUC   | **86.43%** |

For the cancellation class specifically:

| Metric    | Score |
| --------- | ----: |
| Precision |   61% |
| Recall    |   74% |
| F1 Score  |   67% |

The model therefore identifies approximately **74% of cancelled bookings in the held-out test set**.

---

## 📉 Logistic Regression

For Logistic Regression, categorical variables were **one-hot encoded** using:

```python
pd.get_dummies(
    df_lr,
    columns=CAT_COLS,
    drop_first=True
)
```

This produced a feature matrix containing:

```text
87,678 rows × 62 features
```

Numerical features were standardised using `StandardScaler`, with the scaler fitted **only on the training data** to prevent data leakage.

### Hyperparameter tuning

GridSearchCV explored:

```python
C:
    [0.01, 0.1, 1, 10]

penalty:
    ['l1', 'l2']
```

using 3-fold StratifiedKFold cross-validation.

The model used:

```python
class_weight='balanced'
solver='liblinear'
max_iter=500
```

### Best parameters

```python
{
    'C': 10,
    'penalty': 'l1'
}
```

Best cross-validation F1:

```text
0.5919
```

### Test performance

| Metric    |      Score |
| --------- | ---------: |
| Accuracy  | **70.11%** |
| Precision | **47.60%** |
| Recall    | **77.72%** |
| F1 Score  | **59.04%** |
| ROC-AUC   | **81.53%** |

---

# 📊 Model Comparison

| Model               |   Accuracy |  Precision |     Recall |         F1 |    ROC-AUC |
| ------------------- | ---------: | ---------: | ---------: | ---------: | ---------: |
| Random Forest       | **79.43%** | **60.56%** |     73.87% | **66.56%** | **86.43%** |
| Logistic Regression |     70.11% |     47.60% | **77.72%** |     59.04% |     81.53% |

The results show different performance characteristics:

* Random Forest achieved higher accuracy, precision, F1, and ROC-AUC on the held-out test set.
* Logistic Regression achieved higher recall for cancelled bookings.
* Random Forest tuning also demonstrated the importance of controlling tree depth and minimum split size to reduce overfitting.

---

# 🔎 Hyperparameter Tuning & Overfitting

The Random Forest grid search provided an explicit comparison between model complexity and cross-validation performance.

For example, an unconstrained forest with:

```text
max_depth = None
min_samples_split = 2
```

achieved a training F1 of approximately **0.997**, while its cross-validation F1 was substantially lower.

The selected configuration:

```text
max_depth = 20
min_samples_split = 5
n_estimators = 200
```

provided a better balance between training performance and cross-validation performance.

This demonstrates the practical importance of hyperparameter tuning rather than simply selecting the most complex model.

---

# 👥 Customer Segmentation with K-Means

In addition to cancellation prediction, the project uses **K-Means clustering** to identify distinct booking segments.

The clustering workflow:

1. Focuses on cancelled bookings.
2. Removes variables that are unsuitable for clustering.
3. Encodes categorical variables using `OrdinalEncoder`.
4. Caps extreme values at the 99th percentile for selected numerical variables.
5. Standardises the feature matrix.
6. Evaluates different values of `k`.
7. Compares K-Means initialisation strategies.
8. Profiles the resulting clusters.

For outlier handling, the following features were capped at their 99th percentile:

```text
ADR
Lead Time
Days in Waiting List
```

The features were then standardised because K-Means relies on Euclidean distance.

---

## Selecting the Number of Clusters

Values of `k` from 2 to 8 were evaluated using inertia and silhouette score.

|     k |     Inertia | Silhouette |
| ----: | ----------: | ---------: |
|     2 |     568,110 |     0.2695 |
|     3 |     516,160 |     0.2833 |
| **4** | **484,444** | **0.2943** |
|     5 |     456,001 |     0.1268 |
|     6 |     436,235 |     0.1358 |
|     7 |     406,271 |     0.1461 |
|     8 |     394,663 |     0.1579 |

The analysis selected:

```text
k = 4
```

with:

```text
init = k-means++
n_init = 10
```

---

# 👤 Booking Personas

The final K-Means model produced four booking segments:

| Cluster | Segment                | Bookings | Share |
| ------: | ---------------------- | -------: | ----: |
|       0 | Extreme Planners       |    1,180 |  4.9% |
|       1 | Standard Risk          |   17,607 | 72.5% |
|       2 | Premium Bookers        |    3,058 | 12.6% |
|       3 | Last-Minute Cancellers |    2,452 | 10.1% |

Final clustering metrics:

```text
Silhouette Score : 0.2943
Inertia (WCSS)   : 484,444
```

These clusters provide an additional behavioural view of the booking data alongside the supervised cancellation prediction models.

---

# 🛠️ Technologies

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **Matplotlib**
* **Seaborn**
* **Jupyter Notebook**

### Machine Learning

* Random Forest
* Logistic Regression
* K-Means Clustering
* GridSearchCV
* Stratified K-Fold Cross-Validation
* Ordinal Encoding
* One-Hot Encoding
* Standard Scaling

---

# 📁 Repository Structure

```text
Hotel-Booking-Cancellation-Prediction/
│
├── Data_Cleaning.ipynb
├── Data_Modelling_and_Evaluation.ipynb
├── Lab_08_Group_09_Report.pdf
└── README.md
```

### Notebooks

**`Data_Cleaning.ipynb`**

Contains:

* Data quality assessment
* Missing-value analysis
* Categorical normalisation
* Data type correction
* Duplicate detection/removal
* Feature engineering
* Leakage prevention
* Exploratory data analysis

**`Data_Modelling_and_Evaluation.ipynb`**

Contains:

* Random Forest classification
* Logistic Regression classification
* Hyperparameter tuning
* Cross-validation
* Classification metrics
* Confusion matrices
* ROC-AUC evaluation
* K-Means clustering
* Cluster selection
* Cluster profiling

**`Lab_08_Group_09_Report.pdf`**

Contains the detailed project report and analysis.

---

# 🚀 Running the Project

### 1. Clone the repository

```bash
git clone https://github.com/hemangijoshi12/Hotel-Booking-Cancellation-Prediction.git

cd Hotel-Booking-Cancellation-Prediction
```

### 2. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### 3. Run the notebooks

Open Jupyter:

```bash
jupyter notebook
```

Run:

```text
Data_Cleaning.ipynb
        ↓
Data_Modelling_and_Evaluation.ipynb
```

The modelling notebook expects the cleaned dataset generated by the data-cleaning notebook.

---

# 📌 Key Takeaways

This project demonstrates an end-to-end approach to a real-world classification problem:

* Performed extensive data-quality analysis on **119K+ hotel booking records**.
* Removed **32K+ duplicate records**.
* Addressed missing values and inconsistent categorical representations.
* Created behaviour-oriented features such as `total_stay_nights`, `total_guests`, `is_family`, and `room_type_changed`.
* Explicitly removed potential **target leakage**.
* Used **stratified train-test splitting** to preserve class distribution.
* Tuned Random Forest and Logistic Regression using **3-fold cross-validation**.
* Evaluated models using **Accuracy, Precision, Recall, F1 and ROC-AUC**.
* Used K-Means to identify **four booking segments**.
* Combined predictive modelling with unsupervised segmentation to provide complementary views of hotel booking behaviour.

---

## 📄 Detailed Report

For the complete methodology, analysis and discussion, see:

**[Lab_08_Group_09_Report.pdf](Lab_08_Group_09_Report.pdf)**
