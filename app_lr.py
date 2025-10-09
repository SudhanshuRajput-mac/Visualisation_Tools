
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

class GradientDescentVisualizer:
    def __init__(self):
        self.functions = {
            "Quadratic (1D)": self.quadratic_1d,
            "Quadratic (2D)": self.quadratic_2d,
            "Rosenbrock (2D)": self.rosenbrock,
            "Himmelblau (2D)": self.himmelblau
        }
        
        self.gradients = {
            "Quadratic (1D)": self.quadratic_1d_grad,
            "Quadratic (2D)": self.quadratic_2d_grad,
            "Rosenbrock (2D)": self.rosenbrock_grad,
            "Himmelblau (2D)": self.himmelblau_grad
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
    
    def gradient_descent(self, function_name, initial_point, learning_rate, iterations):
        points = [initial_point.copy()]
        losses = [self.calculate_loss(function_name, initial_point)]
        
        current_point = initial_point.copy()
        
        for i in range(iterations):
            grad = self.gradients[function_name](*current_point)
            current_point = current_point - learning_rate * grad
            points.append(current_point.copy())
            losses.append(self.calculate_loss(function_name, current_point))
            
            # Early stopping if divergence detected
            if np.any(np.isnan(current_point)) or np.any(np.abs(current_point) > 1e10):
                break
                
        return np.array(points), np.array(losses)
    
    def calculate_loss(self, function_name, point):
        if len(point) == 1:
            return self.functions[function_name](point[0])
        else:
            return self.functions[function_name](point[0], point[1])

def main():
    st.set_page_config(page_title="Learning Rate Visualizer", layout="wide")
    
    st.title("🎯 Learning Rate Visualization Tool")
    st.markdown("""
    Understand how the learning rate affects gradient descent optimization through interactive visualization.
    Adjust parameters and watch how the optimization path changes in real-time!
    """)
    
    # Initialize visualizer
    visualizer = GradientDescentVisualizer()
    
    # Sidebar controls
    st.sidebar.header("🎛️ Control Panel")
    
    function_type = st.sidebar.selectbox(
        "Select Function:",
        list(visualizer.functions.keys()),
        help="Choose the optimization landscape"
    )
    
    learning_rate = st.sidebar.slider(
        "Learning Rate (η):",
        min_value=0.001,
        max_value=1.0,
        value=0.1,
        step=0.001,
        help="Step size for gradient descent"
    )
    
    iterations = st.sidebar.slider(
        "Number of Iterations:",
        min_value=10,
        max_value=500,
        value=100,
        step=10,
        help="Maximum number of optimization steps"
    )
    
    # Initial point based on function type
    if function_type == "Quadratic (1D)":
        initial_x = st.sidebar.slider("Initial x:", -5.0, 5.0, 4.0, 0.1)
        initial_point = np.array([initial_x])
    else:
        col1, col2 = st.sidebar.columns(2)
        with col1:
            initial_x = st.slider("Initial x:", -5.0, 5.0, 3.0, 0.1)
        with col2:
            initial_y = st.slider("Initial y:", -5.0, 5.0, 3.0, 0.1)
        initial_point = np.array([initial_x, initial_y])
    
    # Add momentum option
    use_momentum = st.sidebar.checkbox("Use Momentum", value=False)
    if use_momentum:
        momentum = st.sidebar.slider("Momentum (β):", 0.0, 0.99, 0.9, 0.01)
    else:
        momentum = 0.0
    
    # Run optimization
    if st.sidebar.button("Run Gradient Descent", type="primary"):
        with st.spinner("Optimizing..."):
            # Modified gradient descent with momentum
            points, losses = momentum_gradient_descent(
                visualizer, function_type, initial_point, learning_rate, iterations, momentum
            )
            
            # Create visualizations
            create_visualizations(visualizer, function_type, points, losses, learning_rate, momentum)

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
        
        # Early stopping if divergence detected
        if np.any(np.isnan(current_point)) or np.any(np.abs(current_point) > 1e10):
            break
            
    return np.array(points), np.array(losses)

def create_visualizations(visualizer, function_type, points, losses, learning_rate, momentum):
    # Create two columns for plots
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔄 Optimization Path")
        
        if function_type == "Quadratic (1D)":
            fig_1d = create_1d_plot(visualizer, function_type, points, losses)
            st.plotly_chart(fig_1d, use_container_width=True)
        else:
            fig_2d = create_2d_plot(visualizer, function_type, points)
            st.plotly_chart(fig_2d, use_container_width=True)
    
    with col2:
        st.subheader("📉 Loss Curve")
        fig_loss = create_loss_plot(losses, learning_rate, momentum)
        st.plotly_chart(fig_loss, use_container_width=True)
    
    # Convergence analysis
    st.subheader("📊 Convergence Analysis")
    col_metrics1, col_metrics2, col_metrics3 = st.columns(3)
    
    final_loss = losses[-1]
    convergence_rate = losses[0] - losses[-1]
    iterations_used = len(losses) - 1
    
    with col_metrics1:
        st.metric("Final Loss", f"{final_loss:.6f}")
    with col_metrics2:
        st.metric("Total Improvement", f"{convergence_rate:.6f}")
    with col_metrics3:
        st.metric("Iterations Used", iterations_used)
    
    # Learning rate analysis
    st.subheader("🎓 Learning Rate Insights")
    
    if learning_rate > 0.5:
        st.error("""
        **⚠️ Learning Rate Too High**: 
        - Likely to cause divergence or oscillation
        - Steps are too large, overshooting the minimum
        - Try reducing the learning rate below 0.1
        """)
    elif learning_rate < 0.01:
        st.warning("""
        **🐌 Learning Rate Too Low**: 
        - Very slow convergence
        - May get stuck in local minima
        - Requires many iterations
        - Try increasing the learning rate
        """)
    else:
        st.success("""
        **✅ Good Learning Rate Range**: 
        - Balanced convergence speed and stability
        - Smooth progression toward minimum
        - Efficient use of iterations
        """)
    
    if momentum > 0:
        st.info(f"""
        **🌀 Momentum Applied (β={momentum})**:
        - Accelerates convergence in relevant directions
        - Reduces oscillations in narrow valleys
        - Helps escape shallow local minima
        """)

def create_1d_plot(visualizer, function_type, points, losses):
    x_vals = np.linspace(-5, 5, 100)
    y_vals = visualizer.quadratic_1d(x_vals)
    
    fig = go.Figure()
    
    # Function curve
    fig.add_trace(go.Scatter(
        x=x_vals, y=y_vals,
        mode='lines',
        name='f(x) = x²',
        line=dict(color='blue', width=2)
    ))
    
    # Optimization path
    path_losses = visualizer.quadratic_1d(points.flatten())
    fig.add_trace(go.Scatter(
        x=points.flatten(), y=path_losses,
        mode='markers+lines',
        name='Optimization Path',
        line=dict(color='red', width=3),
        marker=dict(size=8, color=np.arange(len(points)), 
                   colorscale='Viridis', showscale=True,
                   colorbar=dict(title="Iteration"))
    ))
    
    fig.update_layout(
        title="1D Gradient Descent Optimization",
        xaxis_title="x",
        yaxis_title="f(x)",
        height=500
    )
    
    return fig

def create_2d_plot(visualizer, function_type, points):
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = visualizer.functions[function_type](X, Y)
    
    fig = go.Figure()
    
    # Contour plot
    fig.add_trace(go.Contour(
        x=x, y=y, z=Z,
        colorscale='Viridis',
        showscale=True,
        opacity=0.7,
        name='Loss Surface'
    ))
    
    # Optimization path
    fig.add_trace(go.Scatter(
        x=points[:, 0], y=points[:, 1],
        mode='markers+lines',
        name='Optimization Path',
        line=dict(color='red', width=4),
        marker=dict(
            size=8, 
            color=np.arange(len(points)),
            colorscale='Rainbow',
            showscale=False
        )
    ))
    
    # Start and end points
    fig.add_trace(go.Scatter(
        x=[points[0, 0]], y=[points[0, 1]],
        mode='markers',
        marker=dict(size=12, color='green', symbol='star'),
        name='Start'
    ))
    
    fig.add_trace(go.Scatter(
        x=[points[-1, 0]], y=[points[-1, 1]],
        mode='markers',
        marker=dict(size=12, color='orange', symbol='x'),
        name='End'
    ))
    
    fig.update_layout(
        title=f"2D Gradient Descent - {function_type}",
        xaxis_title="x",
        yaxis_title="y",
        height=500
    )
    
    return fig

def create_loss_plot(losses, learning_rate, momentum):
    iterations = np.arange(len(losses))
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=iterations, y=losses,
        mode='lines+markers',
        name='Loss',
        line=dict(color='purple', width=3),
        marker=dict(size=4)
    ))
    
    fig.update_layout(
        title="Loss vs Iterations",
        xaxis_title="Iteration",
        yaxis_title="Loss",
        yaxis_type="log" if np.max(losses) > 100 else "linear",
        height=500
    )
    
    # Add convergence rate annotation
    if len(losses) > 1:
        final_improvement = losses[-2] - losses[-1]
        fig.add_annotation(
            x=len(losses)-1, y=losses[-1],
            text=f"Final step: {final_improvement:.2e}",
            showarrow=True,
            arrowhead=2
        )
    
    return fig
# Additional extension ideas to add to the main app:

def add_adam_optimizer():
    """Add Adam optimizer option"""
    st.sidebar.subheader("Adam Optimizer")
    use_adam = st.sidebar.checkbox("Use Adam", value=False)
    if use_adam:
        beta1 = st.sidebar.slider("β₁", 0.8, 0.999, 0.9, 0.001)
        beta2 = st.sidebar.slider("β₂", 0.8, 0.999, 0.999, 0.001)
        epsilon = st.sidebar.number_input("ε", 1e-8, 1e-4, 1e-8, format="%.0e")
        return True, beta1, beta2, epsilon
    return False, 0, 0, 0

def create_3d_surface_plot(visualizer, function_type, points):
    """3D surface plot for advanced visualization"""
    x = np.linspace(-5, 5, 50)
    y = np.linspace(-5, 5, 50)
    X, Y = np.meshgrid(x, y)
    Z = visualizer.functions[function_type](X, Y)
    
    fig = go.Figure(data=[
        go.Surface(
            x=X, y=Y, z=Z,
            colorscale='Viridis',
            opacity=0.7,
            name='Loss Surface'
        ),
        go.Scatter3d(
            x=points[:, 0], y=points[:, 1], 
            z=[visualizer.calculate_loss(function_type, p) for p in points],
            mode='markers+lines',
            marker=dict(size=4, color='red'),
            line=dict(color='red', width=4),
            name='Optimization Path'
        )
    ])
    
    return fig

def add_comparison_mode():
    """Compare multiple learning rates simultaneously"""
    st.sidebar.subheader("Comparison Mode")
    compare_lr = st.sidebar.checkbox("Compare Multiple LR", value=False)
    if compare_lr:
        lr1 = st.sidebar.slider("LR 1", 0.001, 1.0, 0.01, 0.001)
        lr2 = st.sidebar.slider("LR 2", 0.001, 1.0, 0.1, 0.001)
        lr3 = st.sidebar.slider("LR 3", 0.001, 1.0, 0.5, 0.001)
        return True, [lr1, lr2, lr3]
    return False, []

if __name__ == "__main__":
    main()
