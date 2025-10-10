# requirements.txt
"""
streamlit==1.28.0
numpy==1.24.0
matplotlib==3.7.0
plotly==5.15.0
scipy==1.11.0
pandas==2.0.0
altair==5.0.0
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import pandas as pd
import altair as alt
from scipy.optimize import minimize

class EnhancedGradientDescentVisualizer:
    def __init__(self):
        self.functions = {
            "Quadratic (1D)": self.quadratic_1d,
            "Quadratic (2D)": self.quadratic_2d,
            "Rosenbrock (2D)": self.rosenbrock,
            "Himmelblau (2D)": self.himmelblau,
            "Ackley (2D)": self.ackley,
            "Rastrigin (2D)": self.rastrigin
        }
        
        self.gradients = {
            "Quadratic (1D)": self.quadratic_1d_grad,
            "Quadratic (2D)": self.quadratic_2d_grad,
            "Rosenbrock (2D)": self.rosenbrock_grad,
            "Himmelblau (2D)": self.himmelblau_grad,
            "Ackley (2D)": self.ackley_grad,
            "Rastrigin (2D)": self.rastrigin_grad
        }
        
        self.global_minima = {
            "Quadratic (1D)": [0.0],
            "Quadratic (2D)": [0.0, 0.0],
            "Rosenbrock (2D)": [1.0, 1.0],
            "Himmelblau (2D)": [[3.0, 2.0], [-2.8, 3.1], [-3.8, -3.3], [3.6, -1.8]],
            "Ackley (2D)": [0.0, 0.0],
            "Rastrigin (2D)": [0.0, 0.0]
        }
    
    def quadratic_1d(self, x):
        return x**2
    
    def quadratic_1d_grad(self, x):
        return 2*x
    
    def quadratic_2d(self, x, y):
        return x**2 + y**2
    
    def quadratic_2d_grad(self, x, y):
        return np.array([2*x, 2*y])
    
    def rosenbrock(self, x, y):
        return (1 - x)**2 + 100 * (y - x**2)**2
    
    def rosenbrock_grad(self, x, y):
        dx = -2*(1 - x) - 400*x*(y - x**2)
        dy = 200*(y - x**2)
        return np.array([dx, dy])
    
    def himmelblau(self, x, y):
        return (x**2 + y - 11)**2 + (x + y**2 - 7)**2
    
    def himmelblau_grad(self, x, y):
        dx = 4*x*(x**2 + y - 11) + 2*(x + y**2 - 7)
        dy = 2*(x**2 + y - 11) + 4*y*(x + y**2 - 7)
        return np.array([dx, dy])
    
    def ackley(self, x, y):
        return -20 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2))) - \
               np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))) + np.e + 20
    
    def ackley_grad(self, x, y):
        # Simplified gradient for Ackley function
        r = np.sqrt(x**2 + y**2)
        if r == 0:
            return np.array([0, 0])
        dx = (0.2 * x / r) * np.exp(-0.2 * r) + 2 * np.pi * np.sin(2 * np.pi * x) * np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y)))
        dy = (0.2 * y / r) * np.exp(-0.2 * r) + 2 * np.pi * np.sin(2 * np.pi * y) * np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y)))
        return np.array([dx, dy])
    
    def rastrigin(self, x, y):
        A = 10
        return A * 2 + (x**2 - A * np.cos(2 * np.pi * x)) + (y**2 - A * np.cos(2 * np.pi * y))
    
    def rastrigin_grad(self, x, y):
        dx = 2*x + 2 * np.pi * 10 * np.sin(2 * np.pi * x)
        dy = 2*y + 2 * np.pi * 10 * np.sin(2 * np.pi * y)
        return np.array([dx, dy])

def main():
    st.set_page_config(page_title="Enhanced LR Visualizer", layout="wide")
    
    st.title("🎯 Enhanced Learning Rate Visualization Tool")
    st.markdown("""
    **Interactive exploration of gradient descent optimization with advanced visualizations**
    """)
    
    # Initialize visualizer
    visualizer = EnhancedGradientDescentVisualizer()
    
    # Sidebar controls
    st.sidebar.header("🎛️ Control Panel")
    
    # Function selection with descriptions
    function_descriptions = {
        "Quadratic (1D)": "Simple convex function - ideal for learning basics",
        "Quadratic (2D)": "Basic bowl-shaped function",
        "Rosenbrock (2D)": "Banana-shaped valley - tests optimizer efficiency",
        "Himmelblau (2D)": "Multiple local minima - complex optimization landscape",
        "Ackley (2D)": "Many local minima - challenging for optimization",
        "Rastrigin (2D)": "Highly multimodal - many local minima"
    }
    
    function_type = st.sidebar.selectbox(
        "Select Function:",
        list(visualizer.functions.keys()),
        help="Choose the optimization landscape"
    )
    
    # Show function description
    st.sidebar.info(f"**{function_type}**: {function_descriptions[function_type]}")
    
    # Advanced settings expander
    with st.sidebar.expander("⚙️ Advanced Settings", expanded=True):
        learning_rate = st.slider(
            "Learning Rate (η):",
            min_value=0.001,
            max_value=2.0,
            value=0.1,
            step=0.001,
            help="Step size for gradient descent"
        )
        
        iterations = st.slider(
            "Number of Iterations:",
            min_value=10,
            max_value=1000,
            value=100,
            step=10,
            help="Maximum number of optimization steps"
        )
        
        # Optimizer selection
        optimizer_type = st.selectbox(
            "Optimizer:",
            ["Vanilla GD", "Momentum", "Adam", "RMSprop"],
            help="Choose optimization algorithm"
        )
        
        if optimizer_type == "Momentum":
            momentum = st.slider("Momentum (β):", 0.0, 0.99, 0.9, 0.01)
        elif optimizer_type == "Adam":
            beta1 = st.slider("β₁:", 0.8, 0.999, 0.9, 0.001)
            beta2 = st.slider("β₂:", 0.8, 0.999, 0.999, 0.001)
        elif optimizer_type == "RMSprop":
            rho = st.slider("Decay Rate (ρ):", 0.8, 0.999, 0.9, 0.001)
    
    # Initial point based on function type
    st.sidebar.subheader("🎯 Initial Position")
    if function_type == "Quadratic (1D)":
        initial_x = st.sidebar.slider("Initial x:", -5.0, 5.0, 4.0, 0.1)
        initial_point = np.array([initial_x])
    else:
        col1, col2 = st.sidebar.columns(2)
        with col1:
            initial_x = st.sidebar.slider("Initial x:", -5.0, 5.0, 3.0, 0.1)
        with col2:
            initial_y = st.sidebar.slider("Initial y:", -5.0, 5.0, 3.0, 0.1)
        initial_point = np.array([initial_x, initial_y])
    
    # Visualization options
    with st.sidebar.expander("📊 Visualization Options"):
        show_3d = st.checkbox("Show 3D Surface", True)
        show_contour = st.checkbox("Show Contour Plot", True)
        show_animation = st.checkbox("Animate Optimization", False)
        show_comparison = st.checkbox("Compare Learning Rates", False)
    
    # Main content area
    if st.sidebar.button("🚀 Run Optimization", type="primary"):
        with st.spinner("Running optimization..."):
            # Run optimization based on selected method
            if optimizer_type == "Vanilla GD":
                points, losses = vanilla_gradient_descent(
                    visualizer, function_type, initial_point, learning_rate, iterations
                )
            elif optimizer_type == "Momentum":
                points, losses = momentum_gradient_descent(
                    visualizer, function_type, initial_point, learning_rate, iterations, momentum
                )
            elif optimizer_type == "Adam":
                points, losses = adam_optimizer(
                    visualizer, function_type, initial_point, learning_rate, iterations, beta1, beta2
                )
            elif optimizer_type == "RMSprop":
                points, losses = rmsprop_optimizer(
                    visualizer, function_type, initial_point, learning_rate, iterations, rho
                )
            
            # Create comprehensive visualizations
            create_enhanced_visualizations(
                visualizer, function_type, points, losses, learning_rate, 
                optimizer_type, show_3d, show_contour, show_animation
            )
            
            # Learning rate comparison if enabled
            if show_comparison:
                create_lr_comparison(visualizer, function_type, initial_point, iterations, optimizer_type)

def vanilla_gradient_descent(visualizer, function_name, initial_point, learning_rate, iterations):
    return run_optimization(visualizer, function_name, initial_point, learning_rate, iterations)

def momentum_gradient_descent(visualizer, function_name, initial_point, learning_rate, iterations, momentum):
    points = [initial_point.copy()]
    losses = [visualizer.calculate_loss(function_name, initial_point)]
    
    current_point = initial_point.copy()
    velocity = np.zeros_like(initial_point)
    
    for i in range(iterations):
        grad = visualizer.gradients[function_name](*current_point)
        velocity = momentum * velocity + learning_rate * grad
        current_point = current_point - velocity
        points.append(current_point.copy())
        losses.append(visualizer.calculate_loss(function_name, current_point))
        
        if np.any(np.isnan(current_point)) or np.any(np.abs(current_point) > 1e10):
            break
            
    return np.array(points), np.array(losses)

def adam_optimizer(visualizer, function_name, initial_point, learning_rate, iterations, beta1, beta2):
    points = [initial_point.copy()]
    losses = [visualizer.calculate_loss(function_name, initial_point)]
    
    current_point = initial_point.copy()
    m = np.zeros_like(initial_point)
    v = np.zeros_like(initial_point)
    epsilon = 1e-8
    
    for t in range(1, iterations + 1):
        grad = visualizer.gradients[function_name](*current_point)
        
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)
        
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        
        current_point = current_point - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
        points.append(current_point.copy())
        losses.append(visualizer.calculate_loss(function_name, current_point))
        
        if np.any(np.isnan(current_point)) or np.any(np.abs(current_point) > 1e10):
            break
            
    return np.array(points), np.array(losses)

def rmsprop_optimizer(visualizer, function_name, initial_point, learning_rate, iterations, rho):
    points = [initial_point.copy()]
    losses = [visualizer.calculate_loss(function_name, initial_point)]
    
    current_point = initial_point.copy()
    cache = np.zeros_like(initial_point)
    epsilon = 1e-8
    
    for i in range(iterations):
        grad = visualizer.gradients[function_name](*current_point)
        
        cache = rho * cache + (1 - rho) * (grad ** 2)
        current_point = current_point - learning_rate * grad / (np.sqrt(cache) + epsilon)
        
        points.append(current_point.copy())
        losses.append(visualizer.calculate_loss(function_name, current_point))
        
        if np.any(np.isnan(current_point)) or np.any(np.abs(current_point) > 1e10):
            break
            
    return np.array(points), np.array(losses)

def run_optimization(visualizer, function_name, initial_point, learning_rate, iterations):
    points = [initial_point.copy()]
    losses = [visualizer.calculate_loss(function_name, initial_point)]
    
    current_point = initial_point.copy()
    
    for i in range(iterations):
        grad = visualizer.gradients[function_name](*current_point)
        current_point = current_point - learning_rate * grad
        points.append(current_point.copy())
        losses.append(visualizer.calculate_loss(function_name, current_point))
        
        if np.any(np.isnan(current_point)) or np.any(np.abs(current_point) > 1e10):
            break
            
    return np.array(points), np.array(losses)

def create_enhanced_visualizations(visualizer, function_type, points, losses, learning_rate, optimizer_type, show_3d, show_contour, show_animation):
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎯 Optimization Path", "📉 Loss Analysis", "🌊 3D Landscape", "📊 Performance Metrics", "🎓 Educational Insights"])
    
    with tab1:
        create_optimization_path_tab(visualizer, function_type, points, losses, show_contour, show_animation)
    
    with tab2:
        create_loss_analysis_tab(losses, learning_rate, optimizer_type)
    
    with tab3:
        if show_3d and function_type != "Quadratic (1D)":
            create_3d_landscape_tab(visualizer, function_type, points)
        else:
            st.info("3D visualization available for 2D functions only")
    
    with tab4:
        create_performance_metrics_tab(visualizer, function_type, points, losses, optimizer_type)
    
    with tab5:
        create_educational_insights_tab(learning_rate, losses, optimizer_type)

def create_optimization_path_tab(visualizer, function_type, points, losses, show_contour, show_animation):
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if function_type == "Quadratic (1D)":
            fig = create_enhanced_1d_plot(visualizer, function_type, points, losses)
        else:
            fig = create_enhanced_2d_plot(visualizer, function_type, points, show_contour)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Real-time progress indicator
        st.subheader("🔄 Optimization Progress")
        
        progress_data = {
            'Iteration': range(len(points)),
            'Loss': losses,
            'Step Size': [np.linalg.norm(points[i] - points[i-1]) if i > 0 else 0 for i in range(len(points))]
        }
        df = pd.DataFrame(progress_data)
        
        # Current status
        current_iter = len(points) - 1
        current_loss = losses[-1]
        total_improvement = losses[0] - losses[-1]
        
        st.metric("Current Iteration", current_iter)
        st.metric("Current Loss", f"{current_loss:.6f}")
        st.metric("Total Improvement", f"{total_improvement:.6f}")
        
        # Step size chart
        if len(points) > 1:
            chart = alt.Chart(df).mark_line().encode(
                x='Iteration',
                y='Step Size',
                tooltip=['Iteration', 'Step Size']
            ).properties(title="Step Size Over Time")
            st.altair_chart(chart, use_container_width=True)

def create_enhanced_1d_plot(visualizer, function_type, points, losses):
    x_vals = np.linspace(-5, 5, 200)
    y_vals = visualizer.quadratic_1d(x_vals)
    
    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.1,
                       subplot_titles=('Optimization Path', 'Gradient Magnitude'))
    
    # Function curve and path
    fig.add_trace(go.Scatter(
        x=x_vals, y=y_vals, mode='lines', name='f(x) = x²',
        line=dict(color='lightblue', width=3), opacity=0.7
    ), row=1, col=1)
    
    path_losses = visualizer.quadratic_1d(points.flatten())
    fig.add_trace(go.Scatter(
        x=points.flatten(), y=path_losses, mode='markers+lines',
        name='Optimization Path', line=dict(color='red', width=4),
        marker=dict(size=8, color=np.arange(len(points)), 
                   colorscale='Viridis', showscale=True,
                   colorbar=dict(title="Iteration"))
    ), row=1, col=1)
    
    # Gradient magnitude
    gradients = [abs(visualizer.quadratic_1d_grad(x)) for x in points.flatten()]
    fig.add_trace(go.Scatter(
        x=np.arange(len(gradients)), y=gradients,
        mode='lines+markers', name='Gradient Magnitude',
        line=dict(color='orange', width=3)
    ), row=2, col=1)
    
    fig.update_layout(height=600, showlegend=True)
    fig.update_yaxes(title_text="f(x)", row=1, col=1)
    fig.update_yaxes(title_text="|∇f(x)|", row=2, col=1)
    fig.update_xaxes(title_text="x", row=1, col=1)
    fig.update_xaxes(title_text="Iteration", row=2, col=1)
    
    return fig

def create_enhanced_2d_plot(visualizer, function_type, points, show_contour):
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = visualizer.functions[function_type](X, Y)
    
    fig = go.Figure()
    
    if show_contour:
        # Enhanced contour plot
        fig.add_trace(go.Contour(
            x=x, y=y, z=Z, colorscale='Viridis',
            contours=dict(showlabels=True),
            opacity=0.7, name='Loss Surface'
        ))
    else:
        # Heatmap style
        fig.add_trace(go.Heatmap(
            x=x, y=y, z=Z, colorscale='Viridis',
            name='Loss Surface'
        ))
    
    # Enhanced optimization path with arrows
    for i in range(len(points)-1):
        fig.add_trace(go.Scatter(
            x=[points[i, 0], points[i+1, 0]],
            y=[points[i, 1], points[i+1, 1]],
            mode='lines',
            line=dict(color='red', width=3),
            showlegend=False
        ))
        
        # Add direction arrows
        if i % 5 == 0:  # Add arrows every 5 steps to avoid clutter
            fig.add_annotation(
                x=points[i, 0], y=points[i, 1],
                ax=points[i+1, 0], ay=points[i+1, 1],
                xref="x", yref="y",
                axref="x", ayref="y",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="red"
            )
    
    # Start and end points
    fig.add_trace(go.Scatter(
        x=[points[0, 0]], y=[points[0, 1]],
        mode='markers', name='Start',
        marker=dict(size=15, color='green', symbol='star', line=dict(width=2, color='darkgreen'))
    ))
    
    fig.add_trace(go.Scatter(
        x=[points[-1, 0]], y=[points[-1, 1]],
        mode='markers', name='End',
        marker=dict(size=15, color='orange', symbol='x', line=dict(width=2, color='darkorange'))
    ))
    
    # Global minima if known
    minima = visualizer.global_minima.get(function_type, [])
    if minima and isinstance(minima[0], list):
        min_x = [m[0] for m in minima]
        min_y = [m[1] for m in minima]
        fig.add_trace(go.Scatter(
            x=min_x, y=min_y, mode='markers',
            name='Global Minima',
            marker=dict(size=10, color='yellow', symbol='diamond', line=dict(width=2, color='gold'))
        ))
    
    fig.update_layout(
        title=f"Optimization Path - {function_type}",
        xaxis_title="x", yaxis_title="y",
        height=600, showlegend=True
    )
    
    return fig

def create_loss_analysis_tab(losses, learning_rate, optimizer_type):
    col1, col2 = st.columns(2)
    
    with col1:
        # Main loss curve
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(
            x=np.arange(len(losses)), y=losses,
            mode='lines+markers', name='Loss',
            line=dict(color='purple', width=3),
            marker=dict(size=4)
        ))
        
        fig_loss.update_layout(
            title="Loss vs Iterations",
            xaxis_title="Iteration",
            yaxis_title="Loss",
            yaxis_type="log" if np.max(losses) > 100 else "linear",
            height=400
        )
        st.plotly_chart(fig_loss, use_container_width=True)
    
    with col2:
        # Convergence analysis
        st.subheader("📈 Convergence Analysis")
        
        if len(losses) > 1:
            improvements = -np.diff(losses)
            final_improvement = improvements[-1] if len(improvements) > 0 else 0
            
            fig_conv = go.Figure()
            fig_conv.add_trace(go.Scatter(
                x=np.arange(1, len(losses)), y=improvements,
                mode='lines+markers', name='Improvement per Step',
                line=dict(color='green', width=2)
            ))
            
            fig_conv.update_layout(
                title="Improvement per Iteration",
                xaxis_title="Iteration",
                yaxis_title="Loss Improvement",
                height=400
            )
            st.plotly_chart(fig_conv, use_container_width=True)
            
            # Convergence metrics
            avg_improvement = np.mean(improvements)
            convergence_ratio = improvements[-1] / improvements[0] if improvements[0] != 0 else 0
            
            st.metric("Average Improvement/Step", f"{avg_improvement:.2e}")
            st.metric("Final Improvement", f"{final_improvement:.2e}")
            st.metric("Convergence Ratio", f"{convergence_ratio:.2f}")

def create_3d_landscape_tab(visualizer, function_type, points):
    st.subheader("🌊 3D Optimization Landscape")
    
    x = np.linspace(-5, 5, 50)
    y = np.linspace(-5, 5, 50)
    X, Y = np.meshgrid(x, y)
    Z = visualizer.functions[function_type](X, Y)
    
    # Create 3D surface
    fig_3d = go.Figure()
    
    fig_3d.add_trace(go.Surface(
        x=X, y=Y, z=Z,
        colorscale='Viridis',
        opacity=0.8,
        name='Loss Surface',
        contours=dict(z=dict(show=True, project_z=True))
    ))
    
    # Add optimization path in 3D
    path_z = [visualizer.calculate_loss(function_type, p) for p in points]
    fig_3d.add_trace(go.Scatter3d(
        x=points[:, 0], y=points[:, 1], z=path_z,
        mode='markers+lines',
        marker=dict(size=4, color='red', opacity=1),
        line=dict(color='red', width=5),
        name='Optimization Path'
    ))
    
    fig_3d.update_layout(
        title=f"3D Landscape - {function_type}",
        scene=dict(
            xaxis_title="x",
            yaxis_title="y",
            zaxis_title="f(x,y)",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        height=600
    )
    
    st.plotly_chart(fig_3d, use_container_width=True)

def create_performance_metrics_tab(visualizer, function_type, points, losses, optimizer_type):
    st.subheader("📊 Performance Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Iterations", len(points) - 1)
        st.metric("Final Loss", f"{losses[-1]:.6f}")
        st.metric("Initial Loss", f"{losses[0]:.6f}")
    
    with col2:
        total_improvement = losses[0] - losses[-1]
        improvement_per_iter = total_improvement / (len(points) - 1) if len(points) > 1 else 0
        st.metric("Total Improvement", f"{total_improvement:.6f}")
        st.metric("Improvement/Iteration", f"{improvement_per_iter:.6f}")
    
    with col3:
        final_grad_norm = np.linalg.norm(visualizer.gradients[function_type](*points[-1]))
        total_distance = np.sum([np.linalg.norm(points[i] - points[i-1]) for i in range(1, len(points))])
        st.metric("Final Gradient Norm", f"{final_grad_norm:.6f}")
        st.metric("Total Path Length", f"{total_distance:.2f}")
    
    # Efficiency analysis
    st.subheader("⏱️ Efficiency Metrics")
    
    efficiency_data = {
        'Metric': ['Convergence Speed', 'Stability', 'Final Accuracy', 'Overall Efficiency'],
        'Score': [min(100, (len(points) / 100) * 100), 
                 min(100, (1 - np.std(losses[-10:]) / np.mean(losses[-10:])) * 100) if len(losses) >= 10 else 50,
                 min(100, (1 / (losses[-1] + 1e-8)) * 10),
                 min(100, (total_improvement / len(points)) * 1000)]
    }
    
    df_eff = pd.DataFrame(efficiency_data)
    
    fig_gauge = go.Figure()
    for i, metric in enumerate(efficiency_data['Metric']):
        fig_gauge.add_trace(go.Indicator(
            mode = "gauge+number",
            value = efficiency_data['Score'][i],
            title = {'text': metric},
            domain = {'row': i//2, 'column': i%2},
            gauge = {'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [{'range': [0, 50], 'color': "lightgray"},
                             {'range': [50, 80], 'color': "gray"},
                             {'range': [80, 100], 'color': "lightgreen"}]}
        ))
    
    fig_gauge.update_layout(
        grid = {'rows': 2, 'columns': 2, 'pattern': "independent"},
        height=400
    )
    
    st.plotly_chart(fig_gauge, use_container_width=True)

def create_educational_insights_tab(learning_rate, losses, optimizer_type):
    st.subheader("🎓 Learning Rate Insights")
    
    # Learning rate analysis
    if learning_rate > 0.5:
        st.error("""
        **⚠️ Learning Rate Too High** 
        - **Symptoms**: Oscillations, divergence, unstable convergence
        - **Impact**: Overshoots minimum, may never converge
        - **Solution**: Reduce learning rate to 0.1 or lower
        """)
        st.progress(0.9, text="Risk: Very High")
        
    elif learning_rate < 0.01:
        st.warning("""
        **🐌 Learning Rate Too Low**
        - **Symptoms**: Very slow convergence, gets stuck easily
        - **Impact**: Wasted computation, may not reach minimum
        - **Solution**: Increase learning rate to 0.01-0.1 range
        """)
        st.progress(0.3, text="Risk: Moderate")
    else:
        st.success("""
        **✅ Good Learning Rate Range**
        - **Benefits**: Stable convergence, efficient progress
        - **Characteristics**: Smooth loss decrease, consistent steps
        - **Optimal Range**: 0.01 to 0.1 for most problems
        """)
        st.progress(0.1, text="Risk: Low")
    
    # Convergence patterns
    st.subheader("🔍 Convergence Pattern Analysis")
    
    if len(losses) > 10:
        recent_improvement = losses[-10] - losses[-1]
        if recent_improvement < 1e-6:
            st.info("**Convergence Status**: Likely converged - minimal recent improvement")
        elif np.any(np.diff(losses[-10:]) > 0):
            st.warning("**Convergence Status**: Oscillating - loss increases detected")
        else:
            st.success("**Convergence Status**: Actively converging - steady improvement")
    
    # Optimizer-specific insights
    st.subheader("🛠️ Optimizer Characteristics")
    
    optimizer_insights = {
        "Vanilla GD": "Simple but sensitive to learning rate. Good for understanding basics.",
        "Momentum": "Faster convergence, reduces oscillations. Good for ravines.",
        "Adam": "Adaptive learning rates. Robust for various problems.",
        "RMSprop": "Good for non-stationary objectives. Adapts learning rate per parameter."
    }
    
    st.info(f"**{optimizer_type}**: {optimizer_insights.get(optimizer_type, '')}")

def create_lr_comparison(visualizer, function_type, initial_point, iterations, optimizer_type):
    st.sidebar.subheader("🔬 Learning Rate Comparison")
    
    lr_values = st.sidebar.text_input("LR values to compare (comma separated)", "0.01, 0.1, 0.5")
    lr_list = [float(x.strip()) for x in lr_values.split(",")]
    
    if st.sidebar.button("Compare Learning Rates"):
        st.subheader("📊 Learning Rate Comparison")
        
        comparison_data = []
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        
        fig_compare = go.Figure()
        
        for i, lr in enumerate(lr_list):
            if optimizer_type == "Vanilla GD":
                points, losses = vanilla_gradient_descent(visualizer, function_type, initial_point, lr, iterations)
            elif optimizer_type == "Momentum":
                points, losses = momentum_gradient_descent(visualizer, function_type, initial_point, lr, iterations, 0.9)
            
            color = colors[i % len(colors)]
            fig_compare.add_trace(go.Scatter(
                x=np.arange(len(losses)), y=losses,
                mode='lines', name=f'LR = {lr}',
                line=dict(color=color, width=3)
            ))
            
            comparison_data.append({
                'Learning Rate': lr,
                'Final Loss': losses[-1],
                'Iterations to Converge': len(losses) - 1,
                'Total Improvement': losses[0] - losses[-1]
            })
        
        fig_compare.update_layout(
            title="Loss Curves for Different Learning Rates",
            xaxis_title="Iteration",
            yaxis_title="Loss",
            yaxis_type="log",
            height=500
        )
        
        st.plotly_chart(fig_compare, use_container_width=True)
        
        # Comparison table
        df_compare = pd.DataFrame(comparison_data)
        st.dataframe(df_compare.style.highlight_min(subset=['Final Loss'], color='lightgreen'), use_container_width=True)

if __name__ == "__main__":
    main()
