import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import io
import base64
from utils.animations import create_gradient_descent_animation, plot_why_subtract

# Configure the page
st.set_page_config(
    page_title="Gradient Descent Visualizer",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Main title
st.markdown('<h1 class="main-header">🎯 Gradient Descent Visualizer</h1>', unsafe_allow_html=True)

# Sidebar for controls
st.sidebar.title("⚙️ Configuration")
st.sidebar.markdown("Adjust the parameters to see how gradient descent works:")

# Function selection
function_choice = st.sidebar.selectbox(
    "Choose Cost Function:",
    ["Quadratic: f(x) = (x-3)² + 2", 
     "Double Well: f(x) = x⁴ - 8x² + 3x + 10",
     "Complex: f(x) = sin(x) + 0.1*(x-2)²"]
)

# Parameter controls
col1, col2, col3 = st.sidebar.columns(3)
with col1:
    start_x = st.slider("Start X", -2.0, 8.0, 6.0, 0.1)
with col2:
    learning_rate = st.slider("Learning Rate (η)", 0.01, 0.8, 0.3, 0.01)
with col3:
    num_steps = st.slider("Number of Steps", 3, 20, 8)

# Algorithm selection
algorithm = st.sidebar.selectbox(
    "Gradient Descent Variant:",
    ["Basic Gradient Descent", "Momentum", "Nesterov", "Adam"]
)

# Add momentum parameter if selected
if algorithm in ["Momentum", "Nesterov"]:
    momentum = st.sidebar.slider("Momentum (β)", 0.1, 0.9, 0.9, 0.1)

# Information section in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 Learning Points:")
st.sidebar.markdown("""
- **Gradient** points uphill (direction of steepest ascent)
- We **subtract** gradient to go **downhill**
- **Learning rate** controls step size
- Too large η → overshooting
- Too small η → slow convergence
""")

# Main content area
tab1, tab2, tab3, tab4 = st.tabs(["🎯 Live Animation", "📊 Optimization Path", "❓ Why Subtract?", "📖 Explanation"])

with tab1:
    st.markdown('<h2 class="sub-header">Live Gradient Descent Animation</h2>', unsafe_allow_html=True)
    
    # Create animation
    fig, animation = create_gradient_descent_animation(
        function_choice, start_x, learning_rate, num_steps, algorithm
    )
    
    # Display animation
    if animation:
        # Convert animation to HTML5 video
        video_html = animation.to_jshtml()
        
        # Display using components
        st.components.v1.html(video_html, height=600)
        
        # Download button for animation
        buf = io.BytesIO()
        animation.save(buf, format='gif', writer='pillow', fps=2)
        buf.seek(0)
        
        st.download_button(
            label="📥 Download Animation as GIF",
            data=buf,
            file_name="gradient_descent.gif",
            mime="image/gif"
        )
    else:
        st.error("Could not generate animation. Please check the parameters.")

with tab2:
    st.markdown('<h2 class="sub-header">Optimization Path Analysis</h2>', unsafe_allow_html=True)
    
    # Create static plot showing the entire optimization path
    fig_static = create_static_optimization_plot(function_choice, start_x, learning_rate, num_steps, algorithm)
    st.pyplot(fig_static)
    
    # Convergence analysis
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 📈 Convergence Metrics")
        # Calculate and display convergence metrics
        metrics = calculate_convergence_metrics(function_choice, start_x, learning_rate, num_steps)
        st.metric("Final Cost", f"{metrics['final_cost']:.4f}")
        st.metric("Total Improvement", f"{metrics['improvement']:.4f}")
        st.metric("Convergence Rate", f"{metrics['convergence_rate']:.4f}")
    
    with col2:
        st.markdown("### 🎯 Parameter History")
        # Show parameter values at each step
        history = get_parameter_history(function_choice, start_x, learning_rate, num_steps)
        st.dataframe(history)

with tab3:
    st.markdown('<h2 class="sub-header">Why Do We Subtract The Gradient?</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Create the "why subtract" visualization
        fig_why = plot_why_subtract(start_x, learning_rate)
        st.pyplot(fig_why)
    
    with col2:
        st.markdown("""
        <div class="info-box">
        <h3>🎯 Key Insight</h3>
        <p><strong>Gradient = Direction of STEEPEST ASCENT</strong></p>
        <p>To <strong>MINIMIZE</strong> the function, we need to go in the <strong>OPPOSITE</strong> direction!</p>
        <br>
        <p><strong>Update Rule:</strong></p>
        <p><code>x_new = x_old - η × ∇f(x)</code></p>
        <br>
        <p>• <span style="color: green">SUBTRACT</span> → Move DOWNHILL ✅</p>
        <p>• <span style="color: red">ADD</span> → Move UPHILL ❌</p>
        </div>
        """, unsafe_allow_html=True)

with tab4:
    st.markdown('<h2 class="sub-header">Gradient Descent Explained</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📖 What is Gradient Descent?
        
        Gradient Descent is an **optimization algorithm** used to find the minimum of a function. 
        In machine learning, we use it to minimize the **cost function** and find the best parameters for our model.
        
        ### 🎯 The Core Idea
        
        1. **Start** at a random point on the cost function
        2. **Calculate** the gradient (slope) at that point
        3. **Move** in the opposite direction of the gradient
        4. **Repeat** until you reach the minimum
        
        ### ⚙️ Key Components
        
        - **Cost Function**: Measures how wrong our predictions are
        - **Gradient**: Direction of steepest ascent
        - **Learning Rate**: Size of each step
        - **Parameters**: Values we're optimizing
        """)
    
    with col2:
        st.markdown("""
        ### 📝 Mathematical Foundation
        
        **Update Rule:**
        ```
        θ = θ - η * ∇J(θ)
        ```
        
        Where:
        - `θ` = Parameters
        - `η` = Learning rate
        - `∇J(θ)` = Gradient of cost function
        
        ### 🚀 Variants
        
        - **Batch GD**: Uses entire dataset
        - **SGD**: Uses one random example
        - **Mini-batch**: Uses small batches
        - **Momentum**: Adds velocity term
        - **Adam**: Adaptive learning rates
        
        ### 💡 Pro Tips
        
        - Normalize your features
        - Monitor learning curves
        - Use learning rate scheduling
        - Try different optimizers
        """)
    
    # Interactive formula explanation
    st.markdown("---")
    st.markdown("### 🧮 Interactive Formula Explorer")
    
    current_x = st.slider("Current x value", -2.0, 8.0, 5.0, 0.1, key="formula_x")
    current_lr = st.slider("Current learning rate", 0.01, 1.0, 0.3, 0.01, key="formula_lr")
    
    # Calculate and display the formula step by step
    gradient = calculate_gradient(function_choice, current_x)
    new_x = current_x - current_lr * gradient
    
    st.latex(f"\\text{{Gradient }}\\nabla f(x) = {gradient:.3f}")
    st.latex(f"x_{{new}} = x_{{old}} - \\eta \\times \\nabla f(x)")
    st.latex(f"x_{{new}} = {current_x:.3f} - {current_lr:.3f} \\times {gradient:.3f} = {new_x:.3f}")

# Footer
st.markdown("---")
st.markdown(
    "**Built with ❤️ using Streamlit | "
    "Gradient Descent Visualizer for Machine Learning Education**"
)

# Utility functions (these would be in a separate file in production)
def create_static_optimization_plot(function_choice, start_x, learning_rate, num_steps, algorithm):
    """Create a static plot showing the optimization path"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Define the function based on selection
    if function_choice == "Quadratic: f(x) = (x-3)² + 2":
        f = lambda x: (x - 3)**2 + 2
        x_vals = np.linspace(-1, 7, 400)
    elif function_choice == "Double Well: f(x) = x⁴ - 8x² + 3x + 10":
        f = lambda x: x**4 - 8*x**2 + 3*x + 10
        x_vals = np.linspace(-3, 4, 400)
    else:  # Complex function
        f = lambda x: np.sin(x) + 0.1*(x-2)**2
        x_vals = np.linspace(-1, 5, 400)
    
    y_vals = f(x_vals)
    ax.plot(x_vals, y_vals, 'b-', linewidth=2, label='Cost Function')
    
    # Simulate gradient descent
    current_x = start_x
    x_history = [current_x]
    y_history = [f(current_x)]
    
    for step in range(num_steps):
        gradient = calculate_gradient(function_choice, current_x)
        current_x = current_x - learning_rate * gradient
        x_history.append(current_x)
        y_history.append(f(current_x))
    
    ax.plot(x_history, y_history, 'ro-', linewidth=2, markersize=6, label='Optimization Path')
    ax.set_xlabel('Parameter (x)')
    ax.set_ylabel('Cost f(x)')
    ax.set_title(f'Gradient Descent Path - {algorithm}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig

def calculate_convergence_metrics(function_choice, start_x, learning_rate, num_steps):
    """Calculate convergence metrics"""
    f = get_function(function_choice)
    current_x = start_x
    initial_cost = f(current_x)
    
    for step in range(num_steps):
        gradient = calculate_gradient(function_choice, current_x)
        current_x = current_x - learning_rate * gradient
    
    final_cost = f(current_x)
    
    return {
        'final_cost': final_cost,
        'improvement': initial_cost - final_cost,
        'convergence_rate': (initial_cost - final_cost) / initial_cost if initial_cost != 0 else 0
    }

def get_parameter_history(function_choice, start_x, learning_rate, num_steps):
    """Get parameter values at each step"""
    import pandas as pd
    
    f = get_function(function_choice)
    current_x = start_x
    history = []
    
    for step in range(num_steps + 1):
        cost = f(current_x)
        gradient = calculate_gradient(function_choice, current_x) if step < num_steps else 0
        history.append({
            'Step': step,
            'Parameter (x)': round(current_x, 4),
            'Cost f(x)': round(cost, 4),
            'Gradient': round(gradient, 4)
        })
        
        if step < num_steps:
            current_x = current_x - learning_rate * gradient
    
    return pd.DataFrame(history)

def get_function(function_choice):
    """Get the function based on selection"""
    if function_choice == "Quadratic: f(x) = (x-3)² + 2":
        return lambda x: (x - 3)**2 + 2
    elif function_choice == "Double Well: f(x) = x⁴ - 8x² + 3x + 10":
        return lambda x: x**4 - 8*x**2 + 3*x + 10
    else:  # Complex function
        return lambda x: np.sin(x) + 0.1*(x-2)**2

def calculate_gradient(function_choice, x, h=1e-5):
    """Calculate gradient numerically"""
    f = get_function(function_choice)
    return (f(x + h) - f(x - h)) / (2 * h)
