# lasso-regression-insurance-ml
# Insurance Data Prediction using Lasso Regression (Machine Learning)

## 📌 Project Overview

This project demonstrates the use of **Lasso Regression (L1 Regularization)** on an **insurance dataset** to build a supervised machine learning prediction model. The same dataset used for Linear Regression was reused here to understand the impact of **regularization**, **feature selection**, and **model generalization**.

The project covers the complete ML pipeline including preprocessing, feature engineering, model training, evaluation, and comparison with basic Linear Regression.

---

## 🛠️ Tools & Libraries Used

* **Python**
* **Pandas** – data manipulation and analysis
* **NumPy** – numerical operations
* **Seaborn & Matplotlib** – data visualization
* **Scikit-learn** – preprocessing, modeling, evaluation
* **JupyterLab** – development environment

---

## 📂 Project Workflow

### 1️⃣ Data Loading & Exploration

* Loaded insurance dataset using **Pandas**
* Checked data structure, missing values, and data types
* Performed exploratory data analysis (EDA)

---

### 2️⃣ Feature & Target Separation

* Defined independent features (**X**) and target variable (**y – insurance charges**)

```python
X = data.drop(columns=["charges"])
y = data["charges"]
```

---

### 3️⃣ Feature Engineering

#### 🔹 One-Hot Encoding

* Applied One-Hot Encoding to categorical variables such as **region**
* Used `drop_first=True` to avoid multicollinearity

```python
X = pd.get_dummies(X, columns=["region"], drop_first=True)
```

#### 🔹 Binary Encoding

* Converted binary categorical features (e.g., sex, smoker) into numeric values (0/1)

#### 🔹 Interaction Features

* Created interaction features to capture combined effects

```python
X["age_smoker"] = X["age"] * X["smoker"]
```

---

### 4️⃣ Feature Scaling

* Applied **Standardization** since Lasso Regression is sensitive to feature scale
* Ensured numeric features (e.g., salary in lakhs) were on the same scale

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

---

### 5️⃣ Train-Test Split

* Split data into training and testing sets

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
```

---

### 6️⃣ Model Training – Lasso Regression

* Trained the model using **Lasso Regression**

```python
from sklearn.linear_model import Lasso
lasso_model = Lasso(alpha=0.5)
lasso_model.fit(X_train, y_train)
```

---

### 7️⃣ Hyperparameter Tuning (LassoCV)

* Used **LassoCV** to automatically select the best alpha value using cross-validation

```python
from sklearn.linear_model import LassoCV
lasso_cv = LassoCV(alphas=[0.001, 0.1, 1, 2, 5, 10], cv=5)
lasso_cv.fit(X_train, y_train)
```

---

### 8️⃣ Prediction

* Predicted insurance charges on test data

```python
y_pred = lasso_cv.predict(X_test)
```

---

### 9️⃣ Model Evaluation

#### 📊 Mean Squared Error (MSE)

* Measured prediction error

```python
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)
```

#### 📊 R² Score

* Evaluated how well the model explains variance

```python
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
```

---

### 🔟 Feature Selection using Lasso

* Observed that Lasso automatically reduces less important feature coefficients to **zero**
* Helped in identifying the most influential features in insurance cost prediction

---

## ⚖️ Underfitting & Overfitting Analysis

* Compared training and testing performance
* Lasso helped reduce overfitting compared to standard Linear Regression

---

## ✅ Key Learnings

* Importance of **regularization** in ML models
* How **L1 penalty** performs automatic feature selection
* Role of **scaling** in Lasso Regression
* Difference between **Linear Regression vs Lasso Regression**

---

## 🚀 Conclusion

Lasso Regression improved model generalization and reduced overfitting by penalizing large coefficients. This project strengthened my understanding of **regularized regression techniques** and their real-world applications using insurance data.

---

## 📌 Future Improvements

* Compare with **Ridge Regression** and **Elastic Net**
* Perform **cross-validation analysis**
* Visualize feature importance using coefficients

---

⭐ If you find this project useful, feel free to star the repository!
