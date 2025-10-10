import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import io

# Configure page
st.set_page_config(
    page_title="Gradient Descent Failure Visualizer",
    page_icon="⚠️",
    layout="wide"
)

st.title("⚠️ Gradient Descent Failure Visualizer")
st.markdown("""
This tool shows **when Gradient Descent fails** due to:
- 🚀 High learning rate (overshoot / divergence)
- 🐢 Low learning rate (very slow)
- 🌀 Non-convex cost function (local minima)
""")

# Sidebar controls
st.sidebar.header("🔧 Controls")
function_type = st.sidebar.selectbox("Select Function Type", ["Convex (x²)", "Non-Convex (x⁴ - x²)"])
alpha = st.sidebar.slider("Learning Rate (α)", 0.001, 1.0, 0.1, 0.001)
iterations = st.sidebar.slider("Iterations", 10, 200, 50)
init_x = st.sidebar.slider("Initial X", -3.0, 3.0, -2.0, 0.1)

# Define loss functions
def convex_func(x): return x**2
def convex_grad(x): return 2*x

def nonconvex_func(x): return x**4 - x**2
def nonconvex_grad(x): return 4*x**3 - 2*x

# Select function
if "Non" in function_type:
    f, grad = nonconvex_func, nonconvex_grad
else:
    f, grad = convex_func, convex_grad

# Gradient Descent
x = init_x
x_values = [x]
for i in range(iterations):
    x = x - alpha * grad(x)
    x_values.append(x)

# Plot
fig, ax = plt.subplots(figsize=(8, 5))
x_range = np.linspace(-3, 3, 400)
y_range = f(x_range)
ax.plot(x_range, y_range, 'b', label='Loss Function')
point, = ax.plot([], [], 'ro', markersize=8)
path, = ax.plot([], [], 'r--', alpha=0.5)
ax.set_title("Gradient Descent Path")
ax.legend()
ax.set_xlabel("x (parameter)")
ax.set_ylabel("Loss f(x)")

# Animation
def animate(i):
    point.set_data(x_values[i], f(x_values[i]))
    path.set_data(x_values[:i], [f(val) for val in x_values[:i]])
    return point, path

anim = FuncAnimation(fig, animate, frames=len(x_values), interval=200, repeat=False)

# Convert animation to HTML
buf = io.BytesIO()
anim.save(buf, format='gif', fps=5)
st.image(buf.getvalue(), caption="Gradient Descent Animation", use_container_width=True)

# Explanation section
st.markdown("### 📊 Interpretation")
if alpha > 0.5:
    st.error("🚨 Learning rate too high! The steps are too large — the algorithm diverges.")
elif alpha < 0.01:
    st.warning("🐢 Learning rate too low — convergence is very slow.")
else:
    st.success("✅ Good learning rate — descent proceeds smoothly.")
    
if "Non" in function_type:
    st.info("🌀 Non-convex function — gradient descent might get stuck in a local minimum.")
