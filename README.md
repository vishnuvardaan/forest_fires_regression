# Algerian Forest Fires Prediction - Machine Learning Project

![Python](https://img.shields.io/badge/python-v3.7+-blue.svg)
![Contributions welcome](https://img.shields.io/badge/contributions-welcome-orange.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## 📋 Project Overview

This project implements machine learning models to predict forest fires in Algeria using meteorological and Fire Weather Index (FWI) data. The analysis focuses on two regions in Algeria: Bejaia (northeast) and Sidi Bel-abbes (northwest), covering the fire season from June to September 2012.

## 🎯 Objectives

- Predict forest fire occurrence using weather and environmental data
- Compare performance of different regression techniques (Ridge and Lasso)
- Analyze the most important features contributing to forest fire prediction
- Provide insights for forest fire prevention and early warning systems

## 📊 Dataset Information

### Dataset Overview
- **Total Instances**: 244 (122 instances per region)
- **Time Period**: June 2012 to September 2012
- **Regions**: Bejaia and Sidi Bel-abbes, Algeria
- **Classes**: Fire (138 instances) and Not Fire (106 instances)
- **Features**: 11 input attributes + 1 output class

### Features Description

#### Weather Data
- **Date**: Day/Month/Year (DD/MM/YYYY)
- **Temperature**: Maximum temperature at noon (22-42°C)
- **RH**: Relative Humidity (21-90%)
- **Ws**: Wind Speed (6-29 km/h)
- **Rain**: Total daily rainfall (0-16.8 mm)

#### Fire Weather Index (FWI) Components
- **FFMC**: Fine Fuel Moisture Code (28.6-92.5)
- **DMC**: Duff Moisture Code (1.1-65.9)
- **DC**: Drought Code (7-220.4)
- **ISI**: Initial Spread Index (0-18.5)
- **BUI**: Buildup Index (1.1-68)
- **FWI**: Fire Weather Index (0-31.1)

#### Target Variable
- **Classes**: Binary classification (Fire/Not Fire)

## 🛠️ Technologies Used

- **Python 3.7+**
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computations
- **Matplotlib** - Data visualization
- **Seaborn** - Statistical data visualization
- **Scikit-learn** - Machine learning algorithms
- **Jupyter Notebook** - Interactive development environment

## 📁 Project Structure

```
forest_fires_regression/
│
├── Algerian_forest_fires_dataset_UPDATE.csv    # Raw dataset
├── Algerian_forest_fires_cleaned_dataset.csv   # Preprocessed dataset
├── Model Training.ipynb                         # Main model training notebook
├── Ridge, Lasso Regression.ipynb              # Regularization techniques analysis
└── README.md                                   # Project documentation
```

## 🚀 Getting Started

### Prerequisites

Make sure you have Python 3.7+ installed. You can install the required packages using:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/forest_fires_regression.git
cd forest_fires_regression
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Launch Jupyter Notebook:
```bash
jupyter notebook
```

4. Open and run the notebooks:
   - `Model Training.ipynb` - For main analysis and model training
   - `Ridge, Lasso Regression.ipynb` - For regularization techniques

## 📈 Analysis & Models

### Exploratory Data Analysis
- Data cleaning and preprocessing
- Statistical analysis of weather patterns
- Correlation analysis between features
- Visualization of fire occurrence patterns

### Machine Learning Models

The following algorithms were implemented and evaluated:

#### 1. **Linear Regression** (Baseline)
- **R² Score**: 0.9848
- Standard linear regression without regularization

#### 2. **Ridge Regression** (L2 Regularization)
- **Ridge**: R² Score = 0.9843
- **RidgeCV** (Cross-Validated): R² Score = 0.9843

#### 3. **Lasso Regression** (L1 Regularization)
- **Lasso**: R² Score = 0.9492
- **LassoCV** (Cross-Validated): R² Score = 0.9821

#### 4. **ElasticNet Regression** (L1 + L2 Regularization)
- **ElasticNet**: R² Score = 0.8753
- **ElasticNetCV** (Cross-Validated): R² Score = 0.9814

### Model Performance Summary

| Algorithm | R² Score | Performance Rank |
|-----------|----------|------------------|
| Linear Regression | 0.9848 | 🥇 1st |
| RidgeCV | 0.9843 | 🥈 2nd |
| Ridge | 0.9843 | 🥈 2nd |
| LassoCV | 0.9821 | 🥉 3rd |
| ElasticNetCV | 0.9814 | 4th |
| Lasso | 0.9492 | 5th |
| ElasticNet | 0.8753 | 6th |

### Model Evaluation
- Cross-validation techniques for model selection
- Performance metrics (MSE, MAE, R²)
- Feature importance analysis
- Model comparison and hyperparameter tuning

## 🎯 Key Results

The project demonstrates:
- Effectiveness of different regression techniques for fire prediction
- Impact of regularization on model performance
- Most important meteorological factors for fire occurrence
- Regional differences in fire patterns between Bejaia and Sidi Bel-abbes

## 📊 Visualizations

The notebooks include various visualizations:
- Distribution plots of weather variables
- Correlation heatmaps
- Feature importance plots
- Model performance comparisons
- Regional fire occurrence patterns