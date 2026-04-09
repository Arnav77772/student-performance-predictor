# ============================================
# Student Performance Predictor - Dashboard
# ============================================

import streamlit as st
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
# Page Config
# ============================================
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Student Performance Predictor")
st.markdown("---")

# ============================================
# Step 1: Dataset Load
# ============================================
@st.cache_data
def load_data():
    df = pd.read_csv('student_data.csv')
    return df

df = load_data()

# ============================================
# Sidebar
# ============================================
st.sidebar.title("📊 Navigation")
section = st.sidebar.radio("Go to:", [
    "📁 Dataset Overview",
    "📈 EDA - Graphs",
    "🔍 Feature Selection & PCA",
    "🤖 ML Models",
    "🎯 Live Prediction"
])

# ============================================
# Section 1: Dataset Overview
# ============================================
if section == "📁 Dataset Overview":
    st.header("📁 Dataset Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Students", df.shape[0])
    col2.metric("Total Features", df.shape[1])
    col3.metric("Pass Rate", f"{(df['pass_fail'].mean()*100):.1f}%")

    st.subheader("Pehli 5 Rows:")
    st.dataframe(df.head())

    st.subheader("Basic Statistics:")
    st.dataframe(df.describe())

    st.subheader("Missing Values:")
    missing = df.isnull().sum().reset_index()
    missing.columns = ['Column', 'Missing Values']
    st.dataframe(missing)

# ============================================
# Section 2: EDA
# ============================================
elif section == "📈 EDA - Graphs":
    st.header("📈 Exploratory Data Analysis")

    # Exam Score Distribution
    st.subheader("Exam Score Distribution")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.histplot(df['exam_score'], bins=30, kde=True, color='blue', ax=ax)
    ax.set_xlabel("Exam Score")
    ax.set_ylabel("Number of Students")
    st.pyplot(fig)

    # Scatter Plots
    st.subheader("Feature vs Exam Score")
    feature = st.selectbox("Feature chunno:", [
        'study_hours', 'attendance_percent',
        'internal_marks', 'previous_marks'
    ])
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.scatterplot(data=df, x=feature, y='exam_score', ax=ax, color='blue')
    ax.set_title(f'{feature} vs Exam Score')
    st.pyplot(fig)

    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
    st.pyplot(fig)

    # Box Plots
    st.subheader("Box Plots - Outliers")
    col = st.selectbox("Column chunno:", [
        'study_hours', 'attendance_percent',
        'internal_marks', 'previous_marks', 'exam_score'
    ])
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=df, y=col, ax=ax, color='lightblue')
    st.pyplot(fig)

# ============================================
# Section 3: Feature Selection & PCA
# ============================================
elif section == "🔍 Feature Selection & PCA":
    st.header("🔍 Feature Selection & PCA")

    X = df[['study_hours', 'attendance_percent',
            'internal_marks', 'previous_marks']]
    y = df['exam_score']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Forward Selection
    st.subheader("Forward Selection")
    lr_base = LinearRegression()
    forward = SequentialFeatureSelector(
        lr_base, n_features_to_select=2, direction='forward')
    forward.fit(X_scaled, y)
    forward_features = X.columns[forward.get_support()].tolist()
    st.success(f"Selected Features: {forward_features}")

    # Backward Selection
    st.subheader("Backward Selection")
    backward = SequentialFeatureSelector(
        lr_base, n_features_to_select=2, direction='backward')
    backward.fit(X_scaled, y)
    backward_features = X.columns[backward.get_support()].tolist()
    st.success(f"Selected Features: {backward_features}")

    # PCA
    st.subheader("PCA Visualization")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    st.info(f"Explained Variance: {pca.explained_variance_ratio_.round(2)}")
    st.info(f"Total Variance Covered: {sum(pca.explained_variance_ratio_):.2f}")

    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1],
                        c=df['exam_score'], cmap='coolwarm', alpha=0.6)
    plt.colorbar(scatter, ax=ax, label='Exam Score')
    ax.set_xlabel('First Principal Component')
    ax.set_ylabel('Second Principal Component')
    ax.set_title('PCA - Student Data')
    st.pyplot(fig)

# ============================================
# Section 4: ML Models
# ============================================
elif section == "🤖 ML Models":
    st.header("🤖 Machine Learning Models")

    X = df[['study_hours', 'attendance_percent',
            'internal_marks', 'previous_marks']]
    y_reg = df['exam_score']
    y_clf = df['pass_fail']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_reg, test_size=0.2, random_state=42)
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X, y_clf, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    scaler2 = StandardScaler()
    X_train_cs = scaler2.fit_transform(X_train_c)
    X_test_cs = scaler2.transform(X_test_c)

    # Regression
    st.subheader("📉 Regression Results")
    col1, col2 = st.columns(2)

    lr = LinearRegression()
    lr.fit(X_train_s, y_train)
    lr_pred = lr.predict(X_test_s)

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train_s, y_train)
    rf_pred = rf.predict(X_test_s)

    with col1:
        st.markdown("**Linear Regression**")
        st.metric("R2 Score", f"{r2_score(y_test, lr_pred):.2f}")
        st.metric("MSE", f"{mean_squared_error(y_test, lr_pred):.2f}")

    with col2:
        st.markdown("**Random Forest**")
        st.metric("R2 Score", f"{r2_score(y_test, rf_pred):.2f}")
        st.metric("MSE", f"{mean_squared_error(y_test, rf_pred):.2f}")

    # Actual vs Predicted
    st.subheader("Actual vs Predicted")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(y_test, rf_pred, color='blue', alpha=0.5)
    ax.plot([0, 100], [0, 100], color='red', linewidth=2)
    ax.set_xlabel('Actual Score')
    ax.set_ylabel('Predicted Score')
    ax.set_title('Actual vs Predicted (Random Forest)')
    st.pyplot(fig)

    # Classification
    st.subheader("📊 Classification Results (Pass/Fail)")
    clf = LogisticRegression()
    clf.fit(X_train_cs, y_train_c)
    clf_pred = clf.predict(X_test_cs)

    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{accuracy_score(y_test_c, clf_pred):.2f}")
    col2.metric("Precision", f"{precision_score(y_test_c, clf_pred):.2f}")
    col3.metric("Recall", f"{recall_score(y_test_c, clf_pred):.2f}")

    st.text(classification_report(y_test_c, clf_pred))

# ============================================
# Section 5: Live Prediction
# ============================================
elif section == "🎯 Live Prediction":
    st.header("🎯 Live Student Prediction")
    st.markdown("Student ki details dalo — AI predict karega!")

    col1, col2 = st.columns(2)

    with col1:
        study_hours = st.slider("Study Hours per day", 1.0, 12.0, 6.0)
        attendance = st.slider("Attendance %", 30.0, 100.0, 75.0)

    with col2:
        internal_marks = st.slider("Internal Marks", 20.0, 100.0, 65.0)
        previous_marks = st.slider("Previous Marks", 20.0, 100.0, 60.0)

    if st.button("🔮 Predict!"):
        X = df[['study_hours', 'attendance_percent',
                'internal_marks', 'previous_marks']]
        y = df['exam_score']

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_scaled, y)

        input_data = scaler.transform([[
            study_hours, attendance,
            internal_marks, previous_marks
        ]])
        prediction = rf.predict(input_data)[0]
        result = "✅ PASS" if prediction >= 40 else "❌ FAIL"

        st.markdown("---")
        col1, col2 = st.columns(2)
        col1.metric("Predicted Exam Score", f"{prediction:.1f}")
        col2.metric("Result", result)

        if prediction >= 70:
            st.success("🌟 Excellent Performance Expected!")
        elif prediction >= 40:
            st.info("👍 Average Performance Expected")
        else:
            st.error("⚠️ Student needs extra support!")