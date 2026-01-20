import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, r2_score
import plotly.express as px
import plotly.graph_objects as go

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="ML Explorer Dashboard - Supervised Learning",
    page_icon="🤖",
    layout="wide"
)

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
    st.plotly_chart(fig, width='stretch')

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

st.plotly_chart(fig_cm, width='stretch')

# Educational tip
st.info("💡 **Try this:** Change the test size slider to see how it affects accuracy!")

# ================================================
# REGRESSION SECTION
# ================================================

st.markdown("---")
st.subheader("🏠 Regression: Diabetes Progression")

# Load Diabetes progression dataset

diabetes = load_diabetes()
diabetes_df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
diabetes_df['progression'] = diabetes.target

st.write("**Dataset Info:**")
st.write("Predicting disease progression one year after baseline")
st.write(f"📊 {diabetes_df.shape[0]} samples, {diabetes_df.shape[1]-1} features")

# Train regression model
X_diabetes = diabetes.data
y_diabetes = diabetes.target

X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(
    X_diabetes, y_diabetes,
    test_size=test_size/100,
    random_state=int(random_state)
)

# Linear Regression
reg_model = LinearRegression()
reg_model.fit(X_train_d, y_train_d)
y_pred_d = reg_model.predict(X_test_d)

#Calculate metrics
rmse = np.sqrt(mean_squared_error(y_test_d, y_pred_d))
r2 = r2_score(y_test_d, y_pred_d)

# Predictions vs Actual scatter plot
fig_reg = go.Figure()

# Add scatter points
fig_reg.add_trace(go.Scatter(
    x=y_test_d,
    y=y_pred_d,
    mode='markers',
    name='Predictions',
    marker=dict(
        size=6,
        color=np.abs(y_test_d - y_pred_d),
        colorscale='Viridis',
        showscale=True,
        colorbar=dict(title="Error")
    ),
    text=[f'Actual: {a:.1f}<br>Predicted: {p:.1f}'
         for a, p in zip(y_test_d, y_pred_d)],
    hovertemplate='%{text}<extra></extra>'
))

# Add perfect prediction line
max_val = max(y_test_d.max(), y_pred_d.max())
min_val = min(y_test_d.min(), y_pred_d.min())
fig_reg.add_trace(go.Scatter(
    x=[min_val, max_val],
    y=[min_val, max_val],
    mode='lines',
    name='Perfect Predictions',
    line=dict(color='red', dash='dash', width=2)
))

fig_reg.update_layout(
    title='Predictions vs Actual Values',
    xaxis_title='Actual Progression',
    yaxis_title='Predicted Progression',
    hovermode='closest',
    legend=dict(
        orientation="h",  # Horizontal
        yanchor="bottom",
        y=1.02,          # Just above plot
        xanchor="center",
        x=0.5            # Centered
    ),
    margin=dict(t=100)   # Space for legend
)

st.plotly_chart(fig_reg, width='stretch')

# Regression metrics
st.subheader("📈 Regression Metrics")

reg_col1, reg_col2, reg_col3 = st.columns(3)

with reg_col1:
    st.metric("RMSE", f"{rmse:.2f}")
    st.caption("(Lower is better)")

with reg_col2:
    st.metric("R² Score", f"{r2:.3f}")
    st.caption("(Closer to 1.0 is better)")

with reg_col3:
    st.metric("Test samples", len(X_test_d))


st.info("""
💡 **Understanding the metrics:**
- **RMSE** (Root Mean Square Error): Average prediction error
- **R²** (R-squared): How well the model explains the variance (1.0 = perfect)
- Points near the red line = good predictions
""")
