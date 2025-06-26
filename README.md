# Rock_Vs_Mine_Prediction

This project implements a machine learning pipeline to classify sonar signals as either rock or mine using the Logistic Regression model. The dataset consists of sonar frequency data with 60 features, and the model is trained to accurately distinguish between the two classes.

# Dataset    
Source: https://www.kaggle.com/datasets/uciml/sonar-all-data  
File used: sonar.csv   
 Attributes: 60 continuous features + 1 label (R for Rock, M for Mine)

# Tech Stack             
  Python    
  Jupyter Notebook   
  NumPy, Pandas  
  Scikit-learn

# Work Flow
1. Load Dataset with Pandas
2. Data Exploration & Visualization
3. Train-Test Split
4. Train Model using Logistic Regression
5. Evaluate Accuracy on Test Data
6. Predict for New Input

 # Evaluation
Model Used: Logistic Regression  
Metric: Accuracy Score  
Train/Test Split: 90% training, 10% testing (with stratified sampling)

#  How to Run 
1.Clone this repository:  
git clone https://github.com/your-username/rock-vs-mine-prediction.git    
cd rock-vs-mine-prediction    
2.Install dependencies:
pip install -r requirements.txt
3.Open and run the notebook:  
jupyter notebook Rock_Vs_Mine_Prediction.ipynb

# Result
input_data = (0.02, 0.04, ..., 0.03)  # 60 values    
input_np = np.asarray(input_data).reshape(1, -1)  
prediction = model.predict(input_np)  
print("Prediction:", "Mine" if prediction[0] == 'M' else "Rock")


