# ===============================
# 1. Import Libraries
# ===============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score
from sklearn.decomposition import PCA

# ===============================
# 2. Load Dataset
# ===============================
# Example: You can replace with your dataset
df = pd.read_csv("student_data_c.csv")

print(df.head())

# ===============================
# 3. Data Preprocessing
# ===============================

# Check missing values
print(df.isnull().sum())

# Fill missing values (simple approach)
df.fillna(df.mean(), inplace=True)

# Feature selection (example)
X = df[['study_hours', 'attendance', 'internal_marks', 'previous_score']]
y_reg = df['final_score']     # Regression target
y_clf = df['pass_fail']       # Classification target (0/1)

# Normalization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ===============================
# 4. Exploratory Data Analysis (EDA)
# ===============================

# Histogram
plt.hist(df['final_score'], bins=10)
plt.title("Distribution of Final Scores")
plt.show()

# Boxplot
sns.boxplot(x=df['final_score'])
plt.title("Boxplot for Outliers")
plt.show()

# Scatter plot
sns.scatterplot(x='study_hours', y='final_score', data=df)
plt.show()

# Correlation Heatmap
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()

# ===============================
# 5. Feature Selection (PCA)
# ===============================
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("Explained Variance:", pca.explained_variance_ratio_)

# ===============================
# 6. Train-Test Split
# ===============================
X_train, X_test, y_train_reg, y_test_reg = train_test_split(
    X_scaled, y_reg, test_size=0.2, random_state=42)

_, _, y_train_clf, y_test_clf = train_test_split(
    X_scaled, y_clf, test_size=0.2, random_state=42)

# ===============================
# 7. Regression Model
# ===============================
reg_model = LinearRegression()
reg_model.fit(X_train, y_train_reg)

y_pred_reg = reg_model.predict(X_test)

print("Regression Results:")
print("MSE:", mean_squared_error(y_test_reg, y_pred_reg))
print("R2 Score:", r2_score(y_test_reg, y_pred_reg))

# ===============================
# 8. Classification Model
# ===============================
clf_model = LogisticRegression()
clf_model.fit(X_train, y_train_clf)

y_pred_clf = clf_model.predict(X_test)

print("\nClassification Results:")
print("Accuracy:", accuracy_score(y_test_clf, y_pred_clf))
print("Precision:", precision_score(y_test_clf, y_pred_clf))
print("Recall:", recall_score(y_test_clf, y_pred_clf))

# ===============================
# 9. Prediction on New Data
# ===============================
new_student = np.array([[5, 80, 70, 65]])  # Example values
new_student_scaled = scaler.transform(new_student)

predicted_score = reg_model.predict(new_student_scaled)
predicted_pass = clf_model.predict(new_student_scaled)

print("\nNew Student Prediction:")
print("Predicted Score:", predicted_score)
print("Pass/Fail:", predicted_pass)