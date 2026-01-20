# ML Explorer Dashboard

Interactive web application demonstrating three machine learning paradigms through hands-on visualizations.

## ✨ Features

### Home Page
- Clean, professional landing page
- Overview of three ML paradigms
- Easy navigation

### 📊 Supervised Learning
- **Iris Flower Classification**
  - Logistic Regression model
  - Interactive train/test split (10-50%)
  - Real-time accuracy metrics
  - Confusion matrix visualization
  - Scatter plot with species coloring
  
- **California Housing Regression**
  - Linear Regression model
  - Predictions vs actual scatter plot
  - RMSE and R² metrics
  - Error visualization with color coding

### 🔍 Unsupervised Learning
- **K-Means Clustering**
  - Adjustable cluster count (k=2 to k=10)
  - Dynamic sample size (100-1000)
  - Silhouette score quality metric
  - Cluster center visualization
  
- **PCA Analysis**
  - 4D → 2D dimensionality reduction
  - Explained variance by component
  - Interactive variance bar chart

### 🎮 Reinforcement Learning
- **Q-Learning Grid World**
  - 5×5 navigation environment
  - Configurable learning parameters (α, γ, ε)
  - Training with progress visualization
  - Rewards over time chart
  - Q-values heatmap showing learned policy
  - Reset functionality

## 🛠️ Tech Stack

- **Python 3.8+**
- **Streamlit** - Web framework for rapid prototyping
- **scikit-learn** - Machine learning algorithms
- **Plotly** - Interactive visualizations
- **NumPy** - Numerical computing
- **Pandas** - Data manipulation

## 📦 Installation

```bash
# Fork and Clone Repository
git clone <your-forked-repo-url>
cd ml_explorer_dashboard

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Usage
```bash
# Activate virtual environment
source venv/bin/activate

# Run application
streamlit run app.py
```

Open your browser to http://localhost:8501

## 📁 Project Structure


```
ml_explorer_dashboard/
├── app.py                          # Home page
├── requirements.txt                # Python dependencies
├── README.md                       # Documentation
├── .gitignore                      # Git ignore rules
└── pages/
    ├── 1_Supervised_Learning.py   # Classification & Regression
    ├── 2_Unsupervised_Learning.py # Clustering & PCA
    └── 3_Reinforcement_Learning.py# Q-learning
```

## 📚 What I Learned

### Technical Skills
- Streamlit web framework
- scikit-learn ML library
- Plotly interactive visualizations
- Git version control
- Python virtual environments

### ML Concepts
- Supervised learning (classification vs regression)
- Model evaluation (accuracy, RMSE, R²)
- Unsupervised learning (clustering, PCA)
- Reinforcement learning (Q-learning, Bellman equation)
- Train/test split importance

## 🎯 Key Features

- **Interactive Controls:** Sliders and buttons that update visualizations in real-time
- **Educational Content:** Explanations and tips throughout
- **Professional Design:** Clean, consistent UI/UX
- **Well-Organized Code:** Modular structure with clear comments
- **Git Best Practices:** Atomic commits with descriptive messages

## 📈 Development Timeline

- **Day 0:** Project setup and dependencies
- **Day 1:** Home page + Supervised learning
- **Day 2:** Unsupervised learning
- **Day 3:** Reinforcement learning
- **Total:** 15+ meaningful commits

## 🔮 Future Enhancements

Potential additions:
- Additional datasets (wine, diabetes, MNIST)
- More algorithms (SVM, Random Forest, Neural Networks)
- Model comparison features
- Data upload functionality
- Dark mode toggle
- Downloadable reports

## 👤 Author 
Josue Munezero - [LinkedIn](https://www.linkedin.com/in/josue-munezero/)

## 📄 License
MIT License
