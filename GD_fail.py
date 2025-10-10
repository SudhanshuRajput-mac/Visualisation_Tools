import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import io
import base64

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
- ⛰️ Saddle points
""")

# Sidebar controls
st.sidebar.header("🔧 Controls")
function_type = st.sidebar.selectbox("Select Function Type", 
                                    ["Convex (x²)", "Non-Convex (x⁴ - x²)", "Saddle Point (x³)"])
alpha = st.sidebar.slider("Learning Rate (α)", 0.001, 2.0, 0.1, 0.001)
iterations = st.sidebar.slider("Iterations", 10, 200, 50)
init_x = st.sidebar.slider("Initial X", -3.0, 3.0, -2.0, 0.1)

# Define loss functions and their gradients
def convex_func(x): return x**2
def convex_grad(x): return 2*x

def nonconvex_func(x): return x**4 - x**2
def nonconvex_grad(x): return 4*x**3 - 2*x

def saddle_func(x): return x**3
def saddle_grad(x): return 3*x**2

# Select function based on choice
if function_type == "Non-Convex (x⁴ - x²)":
    f, grad = nonconvex_func, nonconvex_grad
    x_range = np.linspace(-1.5, 1.5, 400)
elif function_type == "Saddle Point (x³)":
    f, grad = saddle_func, saddle_grad
    x_range = np.linspace(-2, 2, 400)
else:
    f, grad = convex_func, convex_grad
    x_range = np.linspace(-3, 3, 400)

# Gradient Descent simulation
x = init_x
x_values = [x]
loss_values = [f(x)]

for i in range(iterations):
    gradient = grad(x)
    x_new = x - alpha * gradient
    x_values.append(x_new)
    loss_values.append(f(x_new))
    x = x_new

# Create the plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Plot 1: Function and descent path
y_range = f(x_range)
ax1.plot(x_range, y_range, 'b-', linewidth=2, label='Loss Function')
ax1.plot(x_values, loss_values, 'ro-', markersize=4, alpha=0.7, label='Gradient Descent Path')
ax1.set_xlabel("Parameter (x)")
ax1.set_ylabel("Loss f(x)")
ax1.set_title(f"Gradient Descent Path\nFunction: {function_type}")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Loss convergence
ax2.plot(loss_values, 'r-', linewidth=2)
ax2.set_xlabel("Iteration")
ax2.set_ylabel("Loss")
ax2.set_title("Loss Convergence")
ax2.set_yscale('log')
ax2.grid(True, alpha=0.3)

plt.tight_layout()

# Display the static plot
st.pyplot(fig)

# Create animation
fig_anim, ax_anim = plt.subplots(figsize=(8, 5))
ax_anim.plot(x_range, y_range, 'b-', linewidth=2, label='Loss Function')
point, = ax_anim.plot([], [], 'ro', markersize=10)
path, = ax_anim.plot([], [], 'r--', alpha=0.7, linewidth=1)
ax_anim.set_xlabel("Parameter (x)")
ax_anim.set_ylabel("Loss f(x)")
ax_anim.set_title(f"Gradient Descent Animation - {function_type}")
ax_anim.legend()
ax_anim.grid(True, alpha=0.3)

def animate(frame):
    if frame < len(x_values):
        point.set_data([x_values[frame]], [loss_values[frame]])
        path.set_data(x_values[:frame+1], loss_values[:frame+1])
    return point, path

anim = FuncAnimation(fig_anim, animate, frames=len(x_values), interval=300, blit=True, repeat=False)

# Save animation as GIF
try:
    buf = io.BytesIO()
    anim.save(buf, format='gif', writer='pillow', fps=3)
    buf.seek(0)
    
    # Display animation
    st.markdown("### 🎬 Animation")
    st.image(buf.getvalue(), caption="Gradient Descent Animation", use_container_width=True)
except Exception as e:
    st.warning(f"Animation could not be displayed: {e}")

# Analysis and interpretation
st.markdown("## 📊 Analysis")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📈 Final Results")
    st.metric("Final Parameter Value", f"{x_values[-1]:.4f}")
    st.metric("Final Loss", f"{loss_values[-1]:.4f}")
    st.metric("Total Iterations", len(x_values)-1)

with col2:
    st.markdown("### 🔍 Convergence Analysis")
    
    # Check for divergence
    if any(abs(x) > 100 for x in x_values) or any(abs(loss) > 1000 for loss in loss_values):
        st.error("🚨 **DIVERGENCE DETECTED!**")
        st.write("The algorithm is diverging due to excessively high learning rate.")
    
    # Check learning rate issues
    elif alpha > 1.0:
        st.error("🚀 **Learning Rate Too High**")
        st.write("Large oscillations or overshooting observed. Reduce learning rate.")
    
    elif alpha < 0.01:
        st.warning("🐢 **Learning Rate Too Low**")
        st.write("Convergence is very slow. Consider increasing learning rate.")
    
    else:
        st.success("✅ **Good Learning Rate**")
        st.write("Descent is proceeding smoothly toward minimum.")

# Function-specific warnings
st.markdown("### ⚠️ Function-Specific Issues")

if function_type == "Non-Convex (x⁴ - x²)":
    # Check if stuck in local minimum
    final_x = x_values[-1]
    if abs(final_x - 0) < 0.5 and abs(final_x) > 0.1:  # Near local minimum at x=0
        st.error("🌀 **STUCK IN LOCAL MINIMUM!**")
        st.write("Gradient descent converged to a local minimum instead of the global minimum.")
    else:
        st.info("🌀 Non-convex function - risk of local minima")
        st.write("This function has both local and global minima.")

elif function_type == "Saddle Point (x³)":
    if abs(x_values[-1]) < 0.1:  # Near saddle point at x=0
        st.error("⛰️ **STUCK AT SADDLE POINT!**")
        st.write("Gradient is zero at saddle point - algorithm cannot escape.")
    else:
        st.info("⛰️ Saddle point function - gradient becomes zero at x=0")

else:
    st.info("📉 Convex function - guaranteed convergence to global minimum with proper learning rate")

# Additional diagnostics
st.markdown("### 🔧 Technical Details")
col3, col4 = st.columns(2)

with col3:
    st.write("**Parameter History (last 10 values):**")
    st.write([f"{x:.3f}" for x in x_values[-10:]])

with col4:
    st.write("**Loss History (last 10 values):**")
    st.write([f"{loss:.6f}" for loss in loss_values[-10:]])

# Learning rate recommendations
st.markdown("### 💡 Recommendations")
if function_type == "Convex (x²)":
    st.write("For convex functions, try learning rates between 0.01 and 0.5")
elif function_type == "Non-Convex (x⁴ - x²)":
    st.write("For non-convex functions, consider adaptive learning rates or momentum")
elif function_type == "Saddle Point (x³)":
    st.write("For functions with saddle points, consider second-order methods or momentum")

st.markdown("---")
st.caption("Try different combinations of learning rates, initial positions, and functions to see various failure modes!")
