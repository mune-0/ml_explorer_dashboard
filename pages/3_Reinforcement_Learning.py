import streamlit as st
import numpy as np
import plotly.graph_objects as go

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="ML Explorer Dashboard - Reinforcement Learning",
    page_icon="🤖",
    layout="wide"
)

# Page title
st.title("🎮 Reinforcement Learning")
st.markdown("## Q-Learning Grid World")

# ====================================
# SESSION STATE INITIALIZATION
# ====================================

# Initialize Q-table in session state
if 'q_table' not in st.session_state:
    st.session_state.q_table = np.zeros((5, 5, 4)) # 5x5 grid, 4 actions
    st.session_state.episodes_run = 0
    st.session_state.rewards_history = []

# ====================================
# PARAMETERS
# ====================================

# Sidebar parameters
st.sidebar.header("Q-Learning Parameters")
learning_rate = st.sidebar.slider("Learning Rate (α)", 0.1, 1.0, 0.8, 0.1)
discount = st.sidebar.slider("Discount Factor (γ)", 0.1, 1.0, 0.95, 0.05)
epsilon = st.sidebar.slider("Exploration Rate (ε)", 0.0, 1.0, 0.1, 0.05)

episodes_to_train = st.sidebar.number_input(
    "Episodes to train",
    min_value=1,
    max_value=1000,
    value=100
)

# ===================================
# GRID WORLD CONSTANTS
# ===================================

GRID_SIZE = 5
START = (0, 0)
GOAL = (4, 4)
ACTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # right, down, left, up
ACTION_NAMES = ['→', '↓', '←', '↑']

# ==================================
# Q-LEARNING ALGORITHM
# ==================================

def train_episode(q_table, learning_rate, discount, epsilon):
    """
    Run one episode of Q-learning.

    The agent starts at START and tries to reach GOAL by:
    1. Choosing actions (epsilon-greedy)
    2. Receiving rewards
    3. Updating Q-values using Bellman equation

    Parameters:
    -----------
    q_table : numpy.ndarray
        3D array of Q-values (rows × cols × actions)
    learning_rate : float
        How much to update Q-values (0-1)
    discount : float
        How much to value future rewards (0-1)
    epsilon : float
        Probability of random action (0-1)

    Returns:
    --------
    episode_reward : float
        Total reward accumulated in this episode
    """

    state = START
    episode_reward = 0
    steps = 0

    while state != GOAL and steps < 50:
        # Epsilon-greedy action selection
        if np.random.random() < epsilon:
            action = np.random.randint(4) # Explore
        else:
            action = np.argmax(q_table[state[0], state[1], :]) # Exploit

        # Take action
        new_row = max(0, min(GRID_SIZE-1, state[0] + ACTIONS[action][0]))
        new_col = max(0, min(GRID_SIZE-1, state[1] + ACTIONS[action][1]))
        new_state = (new_row, new_col)

        # Calculate reward
        if new_state == GOAL:
            reward = 100
        else:
            reward = -1
        episode_reward += reward

        # Q-learning update (Bellman equation)
        old_value = q_table[state[0], state[1], action]

        if new_state == GOAL:
            next_max = 0 # No future value at terminal state
        else:
            next_max = np.max(q_table[new_state[0], new_state[1], :])

        new_value = old_value + learning_rate * (
            reward + discount * next_max - old_value
        )
        q_table[state[0], state[1], action] = new_value

        state = new_state
        steps += 1

    return episode_reward

# ========================================
# INFO SECTION
# ========================================

st.info("""
🎯 **Goal:** Train an agent to navigate from START (0,0) to GOAL (4,4)

**How it works:**
- Agent learns through trial and error
- Receives +100 reward for reaching goal, -1 for each step
- Q-values represent "quality" of taking each action in each state
- Higher Q-value = better action
""")

# ========================================
# TRAINING SECTION
# ========================================

col1, col2 = st.columns([1, 2])

with col1:
    if st.button("🚀 Train Agent", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()

        rewards = []
        for i in range(episodes_to_train):
            reward = train_episode(
                st.session_state.q_table,
                learning_rate,
                discount,
                epsilon
            )
            rewards.append(reward)
            st.session_state.rewards_history.append(reward)

            # Update progress
            progress_bar.progress((i + 1) / episodes_to_train)
            status_text.text(f"Training... {i+1}/{episodes_to_train}")

        st.session_state.episodes_run += episodes_to_train
        progress_bar.empty()
        status_text.empty()
        st.success(f"✅ Trained {episodes_to_train} episodes!")

    if st.button("🔄 Reset Agent"):
        st.session_state.q_table = np.zeros((5, 5, 4))
        st.session_state.episodes_run = 0
        st.session_state.rewards_history = []
        st.success("Agent reset!")

with col2:
    st.metric("Total Episodes Trained", st.session_state.episodes_run)

# ========================================
# Q-VALUES VISUALIZATION
# ========================================

st.subheader("🗺️ Learned Q-Values Heatmap")

# Get max Q-value for each state
q_values = np.max(st.session_state.q_table, axis=2)

if st.session_state.episodes_run > 0:
    q_values[GOAL[0], GOAL[1]] = 100

# Create heatmap
fig_q = go.Figure(data=go.Heatmap(
    z=q_values,
    colorscale='Viridis',
    text=np.round(q_values, 1),
    texttemplate='%{text}',
    textfont={"size": 10},
    colorbar=dict(title="Q-Value")
))

# Add START and GOAL annotations
fig_q.add_annotation(
    x=0, y=0,
    text="START",
    showarrow=False,
    font=dict(color='white', size=14, family='Arial Black'),
    yshift=15
)
fig_q.add_annotation(
    x=4, y=4,
    text="GOAL",
    showarrow=False,
    font=dict(color='white', size=14, family='Arial Black'),
    yshift=15
)

fig_q.update_layout(
    title='Q-Values Heatmap (brighter = better)',
    xaxis_title='Column',
    yaxis_title='Row',
    height=500
)

st.plotly_chart(fig_q, use_container_width=True)

st.caption("""
💡 **Interpretation:**
- Brighter colors = higher Q-values = better states
- As agent learns, path from START to GOAL becomes brighter
- Dark areas are rarely visited or lead to poor outcomes
""")

# ========================================
# LEARNING PROGRESS VISUALIZATION
# ========================================

# Visualizations
if st.session_state.episodes_run > 0:
    st.subheader("📈 Learning Progress")

    # Learning curve
    fig_progress = go.Figure()
    fig_progress.add_trace(go.Scatter(
        y=st.session_state.rewards_history,
        mode='lines',
        name='Episode Reward',
        line=dict(color='#2c5aa0', width=1)
    ))

    # Add moving average if enough data
    if len(st.session_state.rewards_history) > 10:
        window = 10
        moving_avg = np.convolve(
            st.session_state.rewards_history,
            np.ones(window)/window,
            mode='valid'
        )
        fig_progress.add_trace(go.Scatter(
            y=moving_avg,
            mode='lines',
            name=f'{window}-Episode Average',
            line=dict(color='#ff6b6b', width=2)
        ))

        fig_progress.update_layout(
            title='Rewards Over Time',
            xaxis_title='Episode',
            yaxis_title='Total Reward',
            hovermode='x unified',
            height=400
        )
        st.plotly_chart(fig_progress, use_container_width=True)

# ========================================
# EDUCATIONAL SECTION
# ========================================

st.markdown("---")
st.markdown("""
### 💡 Understanding Reinforcement Learning

**What's happening:**
1. Agent starts at (0,0) with no knowledge
2. Takes random actions, gets rewards
3. Updates Q-values based on results
4. Over time, learns optimal path

**Key Concepts:**
- **Learning Rate (α):** How much to update Q-values
  - Higher = faster learning but less stable
  - Lower = slower but more stable

- **Discount Factor (γ):** How much to value future rewards
  - Higher = more long-term thinking
  - Lower = focus on immediate rewards

- **Exploration Rate (ε):** Probability of random action
  - Higher = more exploration
  - Lower = more exploitation of known good actions

**Experiments to try:**
1. **High learning rate (α=0.9):**
   - Reset agent
   - Train 100 episodes
   - Notice: Learns fast but unstable

2. **Low learning rate (α=0.3):**
   - Reset agent
   - Train 100 episodes
   - Notice: Learns slowly but steady

3. **More training:**
   - Train 500 more episodes
   - Watch path become very bright

**Real-world applications:**
- Game AI (chess, Go, video games)
- Robotics (navigation, manipulation)
- Resource management (traffic lights, power grids)
- Trading strategies (stock market)
- Recommendation systems
""")
