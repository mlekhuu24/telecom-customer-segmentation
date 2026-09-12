# telecom-customer-segmentation
Telecom Customer Segmentation is a machine learning project that uses K-Means clustering to group telecom customers based on their tenure, charges, data usage, call minutes, and support calls. It helps identify different customer groups for better analysis and decision-making
# Telecom Customer Segmentation using Python and Scikit-learn

## Project Overview

This project uses machine learning to segment telecom customers into different groups based on their usage patterns, charges, tenure, and support activity.

## Algorithm Used

* K-Means Clustering
* StandardScaler
* Silhouette Score

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Joblib

## Features Used

* Tenure in months
* Monthly charge
* Total charge
* Data usage in GB
* Call minutes
* Number of support calls

## How It Works

1. Load the telecom customer dataset.
2. Select relevant customer features.
3. Standardize the data using StandardScaler.
4. Apply K-Means clustering with 4 customer segments.
5. Evaluate the clustering using the Silhouette Score.
6. Save the trained model and segmented customer data.
7. Generate a visualization of the customer segments.

## Project Files

* `train_model.py` – Trains the K-Means model.
* `predict.py` – Predicts the segment of a new customer.
* `telecom_customers.csv` – Input dataset.
* `telecom_segmentation_model.pkl` – Trained machine learning model.
* `segmented_customers.csv` – Customer data with assigned segments.
* `customer_segments.png` – Segment visualization.
* `requirements.txt` – Required Python libraries.

## Installation

```bash
pip install -r requirements.txt
```

## Run the Project

Train the model:

```bash
python train_model.py
```

Predict a new customer's segment:

```bash
python predict.py
```

## Output

The project generates:

* Customer segments
* Silhouette Score
* Trained `.pkl` model
* Segmented customer CSV file
* Customer segmentation visualization

## Applications

Customer segmentation can help telecom companies understand customer behavior, identify high-value customers, analyze usage patterns, and improve targeted services and marketing strategies.

## Note

The dataset included in this project is synthetic and intended for educational purposes.
