# Logistics Delivery & Cost Prediction using Linear Regression

## Problem Statement
In the logistics and supply chain industry, accurate prediction of shipment delivery time and shipping cost is very important for efficient planning, customer satisfaction, and cost optimization.

Delays in delivery or inaccurate cost estimation can lead to poor logistics planning, increased operational expenses, and reduced customer trust.

The goal of this project is to build machine learning models that can predict:
- Shipment delivery time (`Transit_Days`)
- Shipment cost (`Cost`)

using historical logistics data.

---

# Project Objectives
- Predict how long a shipment will take to reach its destination
- Predict the cost of each shipment
- Analyze factors affecting delivery time and cost
- Build baseline predictive models using Linear Regression

---

# Dataset Overview
The dataset contains 2000 synthetic logistics shipment records simulating real-world supply chain operations across US locations.

### Features include:
- Shipment_ID
- Origin_Warehouse
- Destination
- Carrier
- Shipment_Date
- Delivery_Date
- Weight_kg
- Cost
- Status
- Distance_miles
- Transit_Days

### Dataset Characteristics:
- Missing values (2%)
- Outliers (3%)
- Seasonal patterns
- Carrier variability
- Real-world noise

---

# Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

# Exploratory Data Analysis (EDA)
The dataset was explored using:
- `.head()`
- `.info()`
- `.describe()`

Visual analysis (pairplots) was used to understand relationships between:
- Distance
- Weight
- Cost
- Transit time

This helped identify patterns, trends, and potential outliers.

---

# Data Cleaning & Feature Engineering

## Missing Values Handling
Missing values in the `Cost` column were filled using median imputation:

```python
logistics["Cost"].fillna(logistics["Cost"].median(), inplace=True)
