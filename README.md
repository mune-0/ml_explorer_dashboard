# ML Explorer Dashboard

Interactive web application demonstrating three machine learning paradigms.

## Status
✅ Day 1 Complete: Home page + Supervised Learning  
🚧 Day 2 In Progress: Unsupervised Learning  
⏳ Day 3 Planned: Reinforcement Learning  

## Features

### ✅ Completed
- Home page with ML paradigm overview
- Supervised Learning:
  - Iris flower classification (Logistic Regression)
  - California housing regression (Linear Regression)
  - Interactive train/test split
  - Real-time accuracy and error metrics
  - Confusion matrix visualization

### 🚧 In Progress
- Unsupervised Learning (K-Means, PCA)

### ⏳ Planned
- Reinforcement Learning (Q-learning)

## Tech Stack
- Python 3.8+
- Streamlit - Web framework
- scikit-learn - ML algorithms
- Plotly - Interactive visualizations
- NumPy & Pandas - Data manipulation

## Installation

```bash
# Clone repository
git clone <your-repo-url>
cd ml_explorer_dashboard

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

## Usage
```bash
# Activate virtual environment
source venv/bin/activate

# Run application
streamlit run app.py
```
Open your browser to http://localhost:8501

## Project Structure
ml_explorer_dashboard/
├── app.py                          # Home page
├── requirements.txt                # Python dependencies
├── README.md                       # This file
└── pages/
    └── 1_Supervised_Learning.py   # Classification & Regression ✅

## Author 
Josue Munezero

## License
MIT License
