Breast Cancer Prediction using Machine Learning
Leveraging machine learning for early and accurate breast cancer detection.

Project Overview
Breast cancer is one of the most prevalent cancers worldwide, and early detection plays a crucial role in improving patient outcomes. This project utilizes machine learning techniques to classify breast tumors as malignant (M) or benign (B) based on diagnostic features.

The project focuses on:

Data preprocessing and exploration to ensure high-quality input data.
Training and evaluation of three machine learning models:
XGBoost (Extreme Gradient Boosting)
Random Forest
Multi-Layer Perceptron (MLP)
Performance analysis using accuracy, precision, recall, and F1 score.
Cross-validation for XGBoost to assess model robustness.
Visualization of results using confusion matrices and other graphical representations.
Dataset
The dataset used in this project is the Breast Cancer Wisconsin (Diagnostic) Dataset, available in breast-cancer.csv. It consists of:

569 samples with 32 features, including the diagnosis label.
Target variable: diagnosis (M = malignant, B = benign).
Features: Various numerical measurements of breast mass, such as radius, texture, perimeter, and area.
Installation
Ensure that Python is installed, then install the required dependencies using:

bash
Copy
Edit
pip install pandas numpy scikit-learn xgboost seaborn matplotlib
Usage
To run the project, follow these steps:

Clone the repository:

bash
Copy
Edit
git clone https://github.com/your-username/breast-cancer-prediction.git
cd breast-cancer-prediction
Launch the Jupyter Notebook:

bash
Copy
Edit
jupyter notebook Breast_Cancer_Prediction_Spotlight.ipynb
Execute the notebook cells sequentially to:

Load and preprocess the dataset.
Train and evaluate the machine learning models.
Visualize and interpret the results.
Models & Performance
This project evaluates the following models:

1. XGBoost (Extreme Gradient Boosting)
Accuracy: 94.74%
Precision: 0.95
Recall: 0.91
F1 Score: 0.93
2. Random Forest (Ensemble Learning)
Accuracy: 96.49%
Precision: 0.98
Recall: 0.93
F1 Score: 0.95
3. Multi-Layer Perceptron (MLP - Neural Network)
Accuracy: 97.37%
Precision: 0.95
Recall: 0.98
F1 Score: 0.97
Cross-Validation (XGBoost)
Mean Accuracy: 93.15%
Standard Deviation: 0.0256
Results
Each model’s performance is assessed using standard evaluation metrics, including accuracy, precision, recall, and F1-score. Confusion matrices are utilized to visualize classification outcomes, ensuring a clear understanding of model effectiveness.

Contributing
Contributions to this project are welcome. If you have ideas for improving the models, feature selection, or visualization techniques, feel free to fork the repository and submit a pull request.

License
This project is licensed under the MIT License, allowing for modification and distribution with appropriate credit.

