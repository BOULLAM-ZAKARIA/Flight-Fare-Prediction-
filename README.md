# Flight Fare Prediction (MakeMyTrip-Style Revenue Use Case)

This project builds a machine learning model to predict **flight ticket fares** across routes.  
The business goal is to support **pricing & conversion strategies** (e.g., triggering discounts/coupon offers for users likely to churn and not book).

---

## Business Problem
As a Data Scientist/ML Engineer at a travel platform (e.g., **MakeMyTrip**), we want to:
- Predict **expected fare** for a given flight search (route, time, airline, stops, duration, etc.)
- Use the prediction to support **dynamic offers** and improve **booking conversion** while protecting revenue

---

## Dataset
- Source: Public flight fare dataset (Excel format, Kaggle-style)
- Target variable: **Price**
- Key columns: Airline, Date_of_Journey, Source, Destination, Dep_Time, Arrival_Time, Duration, Total_Stops, Additional_Info, Route

---

## Project Workflow (Data Science Lifecycle)
### 1) Data Cleaning
- Removed missing values
- Parsed datetime columns and standardized formats

### 2) Exploratory Data Analysis (EDA)
- Price distribution and variance across airlines
- Relationship between duration/stops and price
- Correlation analysis

### 3) Feature Engineering (Core Work)
- Extracted time-based features:
  - `Day`, `Month`, `Year` from `Date_of_Journey`
  - `Dep_hour`, `Dep_min`, `Arr_hour`, `Arr_min`
- Converted `Duration` to **total minutes**
- Encoded categorical features:
  - One-hot for `Source`
  - Ordinal mapping for `Total_Stops`
  - Mean/target encoding for selected features (e.g., Airline, Destination)
- Dropped unused columns after featurization (e.g., Route, Additional_Info)

### 4) Outlier Handling
- Capped extreme `Price` values using median-based replacement (to reduce model sensitivity)

### 5) Feature Selection
- Ranked features using **Mutual Information Regression**

### 6) Modeling & Evaluation
Trained and compared:
- Random Forest Regressor
- Decision Tree Regressor
- XGBoost Regressor

Metrics:
- R², MAE, MSE, RMSE, MAPE

### 7) Hyperparameter Tuning
- Tuned Random Forest using **RandomizedSearchCV**
- Saved final model with **pickle** (deployment-ready artifact)

---

## Results
- Best model: **Random Forest (tuned)**  
- Evaluation metrics: - MAE: 0.81
                      - MAPE: 14.12 %

---

## Tech Stack
- Python, Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- XGBoost
- Pickle

---

## How to Run
1. Clone the repo
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
