# Retail Sales Forecasting & Analytics Dashboard

## Overview

This project focuses on building a machine learning model to forecast retail sales and analyzing the results using an interactive Power BI dashboard.

A regression model was developed using historical sales data to predict future sales. The predictions were then used to perform data analysis and visualize key business insights such as sales trends, promotional impact, and model performance.

The goal is to combine machine learning and data analytics to support data-driven decision making.

## Machine Learning Approach

- Performed data preprocessing and feature engineering (Date → Year, Month, Day, DayOfWeek)
- Merged store-level information with transactional data
- Handled missing values and categorical variables
- Trained a LightGBM regression model to predict sales
- Evaluated model performance using RMSE
- Achieved validation RMSE of ~1183 with ~88% accuracy

## Analysis & Dashboard

The predicted results were exported and used to build an interactive Power BI dashboard.

The dashboard provides:

- Sales trend analysis (Actual vs Predicted)
- Monthly sales patterns
- Impact of promotions on sales
- Model performance visualization using error distribution
- Interactive filters for dynamic exploration

This enables both business insight generation and model evaluation in a single view.

## Key Features

- Machine learning-based sales forecasting
- Feature engineering and data preprocessing
- Model evaluation using RMSE
- Interactive Power BI dashboard
- Sales trend and seasonal analysis
- Promotion impact insights
- Error distribution analysis
