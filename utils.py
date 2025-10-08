import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def create_gradient_descent_animation(function_choice, start_x, learning_rate, num_steps, algorithm):
    """Create gradient descent animation"""
    
    # Define the function and gradient
    if function_choice == "Quadratic: f(x) = (x-3)² + 2":
        def f(x):
            return (x - 3)**2 + 2
        x_vals = np.linspace(-1, 7, 400)
        
    elif function_choice == "Double Well: f(x) = x⁴ - 8x² + 3x + 10":
        def f(x):
            return x**4 - 8*x**2 + 3*x + 10
        x_vals = np.linspace(-3, 4, 400)
        
    else:  # Complex function
        def f(x):
            return np.sin(x) + 0.1*(x-2)**2
        x_vals = np.linspace(-1, 5, 400)
    
    y_vals = f(x_vals)
    
    # Set up the figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Initialize variables for animation
    current_x = start_x
    points_history = []
    
    def animate(frame):
        ax1.clear()
        ax2.clear()
        
        # Plot the function on both subplots
        for ax in [ax1, ax2]:
            ax.plot(x_vals, y_vals, 'b-', linewidth=2, label='Cost Function')
            ax.set_xlabel('Parameter (x)')
            ax.set_ylabel('Cost f(x)')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(x_vals[0], x_vals[-1])
            ax.set_ylim(min(y_vals)-1, max(y_vals)+1)
        
        ax1.set_title(f'Gradient Descent - Step {frame + 1}')
        ax2.set_title('Gradient Explanation')
        
        if frame == 0:
            # Show initial point
            current_y = f(current_x)
            points_history.append((current_x, current_y))
            
            ax1.scatter(current_x, current_y, color='red', s=100, zorder=5)
            ax1.text(0.05, 0.95, f'x = {current_x:.2f}\nf(x) = {current_y:.2f}', 
                    transform=ax1.transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
        else:
            # Calculate gradient and update
            gradient = (f(current_x + 1e-5) - f(current_x - 1e-5)) / (2e-5)
            new_x = current_x - learning_rate * gradient
            current_y = f(current_x)
            new_y = f(new_x)
            
            # Store history
            points_history.append((current_x, current_y))
            
            # Plot all points in history
            history_x, history_y = zip(*points_history)
            ax1.plot(history_x, history_y, 'ro-', markersize=6, linewidth=2, alpha=0.7)
            ax1.scatter(current_x, current_y, color='red', s=100, zorder=5)
            
            # Show gradient information
            ax1.text(0.05, 0.95, 
                    f'x = {current_x:.2f}\n'
                    f'f(x) = {current_y:.2f}\n'
                    f'Gradient = {gradient:.2f}\n'
                    f'New x = {new_x:.2f}', 
                    transform=ax1.transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            # Second plot: Gradient explanation
            ax2.scatter(current_x, current_y, color='red', s=100, zorder=5)
            
            # Show gradient arrow
            grad_scale = min(abs(gradient), 1.5) * np.sign(gradient)
            ax2.arrow(current_x, current_y, grad_scale, 0, 
                     head_width=0.2, head_length=0.1, fc='red', ec='red', 
                     linewidth=3, label=f'Gradient = {gradient:.2f}')
            
            # Show update direction
            update_scale = -learning_rate * gradient
            ax2.arrow(current_x, current_y, update_scale, 0, 
                     head_width=0.2, head_length=0.1, fc='green', ec='green', 
                     linewidth=3, label='Update direction')
            
            ax2.legend()
            
            # Update for next iteration
            current_x = new_x
        
        return []
    
    # Create animation
    animation = FuncAnimation(fig, animate, frames=num_steps, interval=1000, repeat=False, blit=False)
    
    return fig, animation

def plot_why_subtract(start_x, learning_rate):
    """Create the 'why subtract gradient' visualization"""
    
    def f(x):
        return (x - 3)**2 + 2
    
    def df(x):
        return 2 * (x - 3)
    
    x_vals = np.linspace(0, 6, 100)
    y_vals = f(x_vals)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot the function
    ax.plot(x_vals, y_vals, 'b-', linewidth=3, alpha=0.7, label='Cost Function f(x)')
    
    # Starting point
    start_y = f(start_x)
    gradient = df(start_x)
    
    # Plot starting point
    ax.scatter(start_x, start_y, color='red', s=200, zorder=5, label='Current Position')
    
    # Show gradient direction (points uphill)
    ax.arrow(start_x, start_y, 0.8, 0, head_width=0.2, head_length=0.2, 
             fc='red', ec='red', linewidth=4, label=f'Gradient = {gradient:.2f} (Points UPHILL)')
    
    # Show what happens if we SUBTRACT gradient (correct - downhill)
    correct_x = start_x - learning_rate * gradient
    correct_y = f(correct_x)
    ax.arrow(start_x, start_y, -learning_rate * gradient, 0, head_width=0.2, head_length=0.2,
             fc='green', ec='green', linewidth=4, label='SUBTRACT Gradient (DOWNHILL - CORRECT)')
    ax.scatter(correct_x, correct_y, color='green', s=200, zorder=5)
    
    # Show what happens if we ADD gradient (wrong - uphill)
    wrong_x = start_x + learning_rate * gradient
    wrong_y = f(wrong_x)
    ax.arrow(start_x, start_y, learning_rate * gradient, 0, head_width=0.2, head_length=0.2,
             fc='orange', ec='orange', linewidth=4, linestyle='--', label='ADD Gradient (UPHILL - WRONG)')
    ax.scatter(wrong_x, wrong_y, color='orange', s=200, zorder=5)
    
    # Add annotations
    ax.annotate('Higher Cost!', (wrong_x, wrong_y), xytext=(wrong_x+0.3, wrong_y+1),
                arrowprops=dict(arrowstyle='->', color='orange'), fontsize=12, color='orange')
    
    ax.annotate('Lower Cost!', (correct_x, correct_y), xytext=(correct_x-1.5, correct_y+1),
                arrowprops=dict(arrowstyle='->', color='green'), fontsize=12, color='green')
    
    # Mathematical explanation
    equation_text = (
        "MATHEMATICAL REASON:\n\n"
        "Gradient ∇f(x) points in direction\nof STEEPEST ASCENT (uphill)\n\n"
        "To MINIMIZE the function, we go in\nthe OPPOSITE direction:\n\n"
        "UPDATE RULE:\n"
        f"x_new = x_old - η × ∇f(x)\n"
        f"       = {start_x:.1f} - {learning_rate} × {gradient:.1f}\n"
        f"       = {correct_x:.1f}"
    )
    
    ax.text(0.02, 0.98, equation_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    ax.set_xlabel('Parameter Value (x)')
    ax.set_ylabel('Cost f(x)')
    ax.set_title('WHY We Subtract The Gradient in Gradient Descent')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    return fig
