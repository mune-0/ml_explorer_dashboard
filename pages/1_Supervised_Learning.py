import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import plotly.express as px

#Page title
st.title("📊 Supervised Learning")
st.markdown("## Classification and Regression Demos")

# Sidebar for parameters
st.sidebar.header("Parameters")
test_size = st.sidebar.slider(
    "Test Set Size (%)",
    min_value=10,
    max_value=50,
    value=20,
    step=5
)
random_state = st.sidebar.number_input(
    "Random State",
    min_value=0,
    max_value=100,
    value=42
)

# =========================================
# CLASSIFICATION SECTION
# =========================================
st.subheader("🌸 Classification: Iris Flowers")

# Load Iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
df['species'] = df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

# Show dataset preview
col1, col2 = st.columns(2)

with col1:
    st.write("**Dataset Preview:**")
    st.dataframe(df.head(10))
    st.write(f"📊 Shape: {df.shape[0]} samples, {df.shape[1]-2} features")

with col2:
    # Create scatter plot
    fig = px.scatter(
        df,
        x='sepal length (cm)',
        y='sepal width (cm)',
        color='species',
        title='Iris Dataset - Sepal Dimensions',
        color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#45B7D1']
    )
    st.plotly_chart(fig, use_container_width=True)

# Train model
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=test_size/100,
    random_state=int(random_state)
)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Display metrics
st.subheader("🎯 Model Performance")

metric_col1, metric_col2, metric_col3 = st.columns(3)

with metric_col1:
    st.metric("Accuracy", f"{accuracy:.2%}")

with metric_col2:
    st.metric("Train Size", len(X_train))

with metric_col3:
    st.metric("Test Size", len(X_test))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
fig_cm = px.imshow(
    cm,
    text_auto=True,
    labels=dict(x="Predicted Class", y="actual Class"),
    x=['Setosa', 'Versicolor', 'Virginica'],
    y=['Setosa', 'Versicolor', 'Virginica'],
    title='Confusion Matrix',
    color_continuous_scale='Blues'
)

st.plotly_chart(fig_cm, use_container_width=True)

# Educational tip
st.info("💡 **Try this:** Change the test size slider to see how it affects accuracy!")
