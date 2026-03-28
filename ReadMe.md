# Boeing Data Science Challenge Problem ✈️

## Kerollos Lowandy

**Repository:** [data-science-car-industry-challenge-problem](https://github.com/Kerollosl/data-science-car-industry-challenge-problem)

## 📋 Overview

A machine learning solution for the Boeing Data Science Challenge, predicting aircraft component failures and maintenance needs using regression and classification models. The project analyzes training data to build predictive models for the automotive/aerospace industry.

## 🎯 Challenge Objectives

- Predict component failure probabilities
- Classify maintenance priority levels
- Optimize predictive maintenance scheduling
- Compare multiple ML model performances

## 🧠 Model Selection

**Primary Model: Random Forest**
- Best performance across metrics
- Handles both regression and classification
- Robust to outliers and missing data
- Provides feature importance insights

**Alternative Models Tested:**
- Linear Regression
- Decision Trees
- Gradient Boosting
- Neural Networks (TensorFlow)

## 📊 Dataset

- **Training Data:** `Training_DataSet.csv` (9.4 MB)
- **Test Data:** `Test_Dataset.csv` (1.5 MB)
- **Features:** Multiple sensor readings and operational parameters
- **Target:** Component failure prediction

## 🚀 Quick Start

### Prerequisites

**Python Version:** 3.9.13

**Required Packages:**
```bash
pip install numpy==1.26.2 pandas==2.1.3 matplotlib==3.8.2 tensorflow==2.15.0 scikit-learn==1.3.2
```

### Running the Solution

**Option 1: Default (Random Forest)**
```bash
python main.py
```

**Option 2: Compare All Models**
1. Open `main.py`
2. Navigate to line 42
3. Follow commented instructions to enable model comparison
4. Run the script

## 📁 Repository Contents

### Python Scripts
- `main.py` - Main execution script with Random Forest implementation
- `functions.py` - Utility functions for data processing and model evaluation

### Data Files
- `Training_DataSet.csv` - Training dataset with labels
- `Test_Dataset.csv` - Test dataset for predictions

### Documentation
- `Boeing Data Science Challenge Problem Instructions.pdf` - Official challenge requirements
- `Boeing_Data_Science_Challenge_Problem.ipynb` - Jupyter notebook with complete analysis

### Output
- `Output_Data_Visuals/` - Generated visualizations and metrics

## 🛠️ Technical Stack

- **Machine Learning:** Scikit-learn, TensorFlow
- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib
- **Model Types:** Regression, Classification
- **Evaluation:** Cross-validation, metrics analysis

## 📝 Workflow

1. **Data Loading & Preprocessing**
   - Load CSV datasets
   - Handle missing values
   - Feature engineering
   - Data normalization

2. **Model Training**
   - Train Random Forest models
   - Hyperparameter tuning
   - Cross-validation
   - Performance evaluation

3. **Prediction & Output**
   - Generate test set predictions
   - Save results to CSV
   - Visualize performance metrics

4. **Model Comparison** (Optional)
   - Train alternative models
   - Compare metrics
   - Analyze trade-offs

## 📈 Model Performance

The Random Forest model was selected based on:
- **Accuracy:** Highest among tested models
- **Robustness:** Handles complex feature interactions
- **Interpretability:** Feature importance analysis
- **Speed:** Efficient training and prediction

## 🔧 Customization

### Change Model Type
```python
# In main.py, modify the model selection
from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier()
```

### Adjust Hyperparameters
```python
# Random Forest parameters
n_estimators = 100  # Number of trees
max_depth = 10      # Tree depth
min_samples_split = 2  # Minimum samples to split
```

## 📊 Visualizations

The project generates:
- Feature importance plots
- Model performance comparisons
- Prediction distribution charts
- Error analysis visualizations

## 🎓 Applications

- Predictive maintenance in aerospace
- Component failure forecasting
- Maintenance scheduling optimization
- Quality control in manufacturing
- Risk assessment for aircraft operations

## 👤 Author

**Kerollos Lowandy**
- GitHub: [@Kerollosl](https://github.com/Kerollosl)
- Email: klowandy@gmail.com

## 📄 License

This project is available for educational and research purposes.

---

**Last Updated:** March 27, 2026
