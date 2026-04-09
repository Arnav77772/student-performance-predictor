# ============================================
# Student Performance Predictor
# Complete ML Project
# ============================================

# Step 1: Libraries import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.metrics import (mean_squared_error, r2_score,
                             accuracy_score, precision_score,
                             recall_score, classification_report)

# ============================================
# Step 2: Dataset Load karna
# ============================================
df = pd.read_csv('student_data.csv')

print("=" * 50)
print("DATASET OVERVIEW")
print("=" * 50)
print("\nPehli 5 rows:")
print(df.head())
print("\nDataset size:", df.shape)
print("\nBasic Statistics:")
print(df.describe())
print("\nMissing Values:")
print(df.isnull().sum())

# ============================================
# Step 3: EDA - Exploratory Data Analysis
# ============================================
print("\n" + "=" * 50)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 50)

# Exam Score Distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['exam_score'], bins=30, kde=True, color='blue')
plt.title('Exam Score Distribution')
plt.xlabel('Exam Score')
plt.ylabel('Number of Students')
plt.tight_layout()
plt.show()

# Scatter Plots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
sns.scatterplot(data=df, x='study_hours', y='exam_score', ax=axes[0,0], color='blue')
axes[0,0].set_title('Study Hours vs Exam Score')
sns.scatterplot(data=df, x='attendance_percent', y='exam_score', ax=axes[0,1], color='green')
axes[0,1].set_title('Attendance vs Exam Score')
sns.scatterplot(data=df, x='internal_marks', y='exam_score', ax=axes[1,0], color='red')
axes[1,0].set_title('Internal Marks vs Exam Score')
sns.scatterplot(data=df, x='previous_marks', y='exam_score', ax=axes[1,1], color='purple')
axes[1,1].set_title('Previous Marks vs Exam Score')
plt.tight_layout()
plt.show()

# Correlation Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()

# Box Plots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
sns.boxplot(data=df, y='study_hours', ax=axes[0,0], color='blue')
axes[0,0].set_title('Study Hours')
sns.boxplot(data=df, y='attendance_percent', ax=axes[0,1], color='green')
axes[0,1].set_title('Attendance %')
sns.boxplot(data=df, y='internal_marks', ax=axes[1,0], color='red')
axes[1,0].set_title('Internal Marks')
sns.boxplot(data=df, y='previous_marks', ax=axes[1,1], color='purple')
axes[1,1].set_title('Previous Marks')
plt.tight_layout()
plt.show()

# ============================================
# Step 4: Preprocessing
# ============================================
print("\n" + "=" * 50)
print("DATA PREPROCESSING")
print("=" * 50)

X = df[['study_hours', 'attendance_percent', 'internal_marks', 'previous_marks']]
y = df['exam_score']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training data size: {X_train_scaled.shape}")
print(f"Testing data size: {X_test_scaled.shape}")
print("Preprocessing done! ✅")
