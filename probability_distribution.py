import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Probability Distribution Visualizer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .distribution-info {
        font-size: 0.9rem;
        line-height: 1.4;
    }
</style>
""", unsafe_allow_html=True)

class DistributionVisualizer:
    def __init__(self):
        self.distributions = {
            "Normal": self.normal_dist,
            "Uniform": self.uniform_dist,
            "Exponential": self.exponential_dist,
            "Binomial": self.binomial_dist,
            "Poisson": self.poisson_dist,
            "Gamma": self.gamma_dist,
            "Beta": self.beta_dist,
            "Chi-square": self.chi2_dist
        }
        
        self.distribution_info = {
            "Normal": "The normal distribution is symmetric and bell-shaped. It's characterized by mean (μ) and standard deviation (σ). Many natural phenomena follow this distribution.",
            "Uniform": "All outcomes are equally likely within a specified range [a, b]. The PDF is constant across the interval.",
            "Exponential": "Models the time between events in a Poisson process. Characterized by rate parameter λ (lambda). Memoryless property.",
            "Binomial": "Models the number of successes in n independent trials with probability p of success on each trial.",
            "Poisson": "Models the number of events occurring in a fixed interval of time/space with a known constant rate λ.",
            "Gamma": "Generalizes the exponential distribution. Useful for modeling waiting times, insurance claims, and rainfall amounts.",
            "Beta": "A flexible distribution defined on [0,1] interval. Useful for modeling probabilities and proportions.",
            "Chi-square": "Sum of squares of k independent standard normal variables. Used in hypothesis testing and confidence intervals."
        }
    
    def normal_dist(self):
        col1, col2 = st.columns(2)
        with col1:
            mu = st.slider("Mean (μ)", -5.0, 5.0, 0.0, 0.1, key="normal_mu")
        with col2:
            sigma = st.slider("Standard Deviation (σ)", 0.1, 5.0, 1.0, 0.1, key="normal_sigma")
        
        x = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)
        pdf = stats.norm.pdf(x, mu, sigma)
        cdf = stats.norm.cdf(x, mu, sigma)
        samples = stats.norm.rvs(mu, sigma, 1000)
        
        return x, pdf, cdf, samples, f"Normal(μ={mu}, σ={sigma})"
    
    def uniform_dist(self):
        col1, col2 = st.columns(2)
        with col1:
            a = st.slider("Lower bound (a)", -10.0, 5.0, 0.0, 0.1, key="uniform_a")
        with col2:
            b = st.slider("Upper bound (b)", a + 0.1, 10.0, 1.0, 0.1, key="uniform_b")
        
        x = np.linspace(a - 1, b + 1, 1000)
        pdf = stats.uniform.pdf(x, a, b - a)
        cdf = stats.uniform.cdf(x, a, b - a)
        samples = stats.uniform.rvs(a, b - a, 1000)
        
        return x, pdf, cdf, samples, f"Uniform(a={a}, b={b})"
    
    def exponential_dist(self):
        lam = st.slider("Rate parameter (λ)", 0.1, 5.0, 1.0, 0.1, key="exp_lambda")
        
        x = np.linspace(0, 10/lam, 1000)
        pdf = stats.expon.pdf(x, scale=1/lam)
        cdf = stats.expon.cdf(x, scale=1/lam)
        samples = stats.expon.rvs(scale=1/lam, size=1000)
        
        return x, pdf, cdf, samples, f"Exponential(λ={lam})"
    
    def binomial_dist(self):
        col1, col2 = st.columns(2)
        with col1:
            n = st.slider("Number of trials (n)", 1, 50, 10, 1, key="binom_n")
        with col2:
            p = st.slider("Probability of success (p)", 0.0, 1.0, 0.5, 0.01, key="binom_p")
        
        x = np.arange(0, n + 1)
        pdf = stats.binom.pmf(x, n, p)
        cdf = stats.binom.cdf(x, n, p)
        samples = stats.binom.rvs(n, p, size=1000)
        
        return x, pdf, cdf, samples, f"Binomial(n={n}, p={p})"
    
    def poisson_dist(self):
        lam = st.slider("Rate parameter (λ)", 0.1, 20.0, 5.0, 0.1, key="poisson_lambda")
        
        x = np.arange(0, int(3 * lam) + 5)
        pdf = stats.poisson.pmf(x, lam)
        cdf = stats.poisson.cdf(x, lam)
        samples = stats.poisson.rvs(lam, size=1000)
        
        return x, pdf, cdf, samples, f"Poisson(λ={lam})"
    
    def gamma_dist(self):
        col1, col2 = st.columns(2)
        with col1:
            alpha = st.slider("Shape (α)", 0.1, 10.0, 2.0, 0.1, key="gamma_alpha")
        with col2:
            beta = st.slider("Scale (β)", 0.1, 5.0, 1.0, 0.1, key="gamma_beta")
        
        x = np.linspace(0, stats.gamma.ppf(0.99, alpha, scale=beta), 1000)
        pdf = stats.gamma.pdf(x, alpha, scale=beta)
        cdf = stats.gamma.cdf(x, alpha, scale=beta)
        samples = stats.gamma.rvs(alpha, scale=beta, size=1000)
        
        return x, pdf, cdf, samples, f"Gamma(α={alpha}, β={beta})"
    
    def beta_dist(self):
        col1, col2 = st.columns(2)
        with col1:
            alpha = st.slider("Shape α", 0.1, 10.0, 2.0, 0.1, key="beta_alpha")
        with col2:
            beta = st.slider("Shape β", 0.1, 10.0, 2.0, 0.1, key="beta_beta")
        
        x = np.linspace(0, 1, 1000)
        pdf = stats.beta.pdf(x, alpha, beta)
        cdf = stats.beta.cdf(x, alpha, beta)
        samples = stats.beta.rvs(alpha, beta, size=1000)
        
        return x, pdf, cdf, samples, f"Beta(α={alpha}, β={beta})"
    
    def chi2_dist(self):
        df = st.slider("Degrees of freedom (k)", 1, 30, 5, 1, key="chi2_df")
        
        x = np.linspace(0, stats.chi2.ppf(0.99, df), 1000)
        pdf = stats.chi2.pdf(x, df)
        cdf = stats.chi2.cdf(x, df)
        samples = stats.chi2.rvs(df, size=1000)
        
        return x, pdf, cdf, samples, f"Chi-square(k={df})"
    
    def calculate_statistics(self, samples, dist_name):
        if dist_name in ["Binomial", "Poisson"]:
            return {
                "Mean": np.mean(samples),
                "Variance": np.var(samples),
                "Skewness": stats.skew(samples),
                "Kurtosis": stats.kurtosis(samples)
            }
        else:
            return {
                "Mean": np.mean(samples),
                "Variance": np.var(samples),
                "Skewness": stats.skew(samples),
                "Kurtosis": stats.kurtosis(samples)
            }

def main():
    st.markdown('<h1 class="main-header">📊 Probability Distribution Visualizer</h1>', unsafe_allow_html=True)
    
    visualizer = DistributionVisualizer()
    
    # Sidebar
    with st.sidebar:
        st.header("Distribution Settings")
        
        # Distribution selection
        selected_dist = st.selectbox(
            "Select Distribution",
            list(visualizer.distributions.keys())
        )
        
        # Multiple distributions option
        st.subheader("Comparison Settings")
        compare_mode = st.checkbox("Compare Multiple Distributions")
        
        if compare_mode:
            num_comparisons = st.slider("Number of distributions to compare", 2, 5, 2)
            comparison_dists = []
            comparison_params = []
            
            for i in range(num_comparisons):
                st.markdown(f"**Distribution {i+1}**")
                dist = st.selectbox(
                    f"Distribution type {i+1}",
                    list(visualizer.distributions.keys()),
                    key=f"comp_dist_{i}"
                )
                comparison_dists.append(dist)
        
        # Information box
        with st.expander("ℹ️ Distribution Information"):
            st.markdown(f'<div class="distribution-info">{visualizer.distribution_info[selected_dist]}</div>', unsafe_allow_html=True)
    
    # Main content
    if not compare_mode:
        # Single distribution view
        x, pdf, cdf, samples, dist_label = visualizer.distributions[selected_dist]()
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Probability Density Function (PDF)', 
                          'Cumulative Distribution Function (CDF)',
                          'Histogram with Theoretical PDF',
                          'Empirical vs Theoretical CDF'),
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        
        # PDF plot
        fig.add_trace(
            go.Scatter(x=x, y=pdf, mode='lines', name='PDF', 
                      line=dict(color='blue', width=2),
                      hovertemplate='x: %{x:.3f}<br>f(x): %{y:.3f}<extra></extra>'),
            row=1, col=1
        )
        
        # CDF plot
        fig.add_trace(
            go.Scatter(x=x, y=cdf, mode='lines', name='CDF',
                      line=dict(color='red', width=2),
                      hovertemplate='x: %{x:.3f}<br>F(x): %{y:.3f}<extra></extra>'),
            row=1, col=2
        )
        
        # Histogram with theoretical PDF
        fig.add_trace(
            go.Histogram(x=samples, histnorm='probability density', 
                        name='Empirical', nbinsx=50, opacity=0.7,
                        marker_color='lightblue'),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=x, y=pdf, mode='lines', name='Theoretical PDF',
                      line=dict(color='blue', width=2)),
            row=2, col=1
        )
        
        # Empirical vs Theoretical CDF
        sorted_samples = np.sort(samples)
        empirical_cdf = np.arange(1, len(sorted_samples) + 1) / len(sorted_samples)
        
        fig.add_trace(
            go.Scatter(x=sorted_samples, y=empirical_cdf, mode='lines', 
                      name='Empirical CDF', line=dict(color='orange', width=2)),
            row=2, col=2
        )
        
        fig.add_trace(
            go.Scatter(x=x, y=cdf, mode='lines', name='Theoretical CDF',
                      line=dict(color='red', width=2, dash='dash')),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            showlegend=True,
            title_text=f"{dist_label} Distribution",
            title_x=0.5
        )
        
        # Update axes
        fig.update_xaxes(title_text="x", row=1, col=1)
        fig.update_xaxes(title_text="x", row=1, col=2)
        fig.update_xaxes(title_text="x", row=2, col=1)
        fig.update_xaxes(title_text="x", row=2, col=2)
        
        fig.update_yaxes(title_text="f(x)", row=1, col=1)
        fig.update_yaxes(title_text="F(x)", row=1, col=2)
        fig.update_yaxes(title_text="Density", row=2, col=1)
        fig.update_yaxes(title_text="Probability", row=2, col=2)
        
        # Display plot
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistics
        st.subheader("Summary Statistics")
        stats_data = visualizer.calculate_statistics(samples, selected_dist)
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Mean", f"{stats_data['Mean']:.4f}")
        with col2:
            st.metric("Variance", f"{stats_data['Variance']:.4f}")
        with col3:
            st.metric("Skewness", f"{stats_data['Skewness']:.4f}")
        with col4:
            st.metric("Kurtosis", f"{stats_data['Kurtosis']:.4f}")
    
    else:
        # Comparison mode
        st.subheader("Distribution Comparison")
        
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        fig_comp = make_subplots(rows=1, cols=2, subplot_titles=('PDF Comparison', 'CDF Comparison'))
        
        for i in range(num_comparisons):
            # For simplicity in comparison mode, use default parameters
            # In a more advanced version, you could add parameter controls for each distribution
            if comparison_dists[i] == "Normal":
                x_comp, pdf_comp, cdf_comp, _, dist_label = visualizer.normal_dist()
            elif comparison_dists[i] == "Uniform":
                x_comp, pdf_comp, cdf_comp, _, dist_label = visualizer.uniform_dist()
            elif comparison_dists[i] == "Exponential":
                x_comp, pdf_comp, cdf_comp, _, dist_label = visualizer.exponential_dist()
            elif comparison_dists[i] == "Binomial":
                x_comp, pdf_comp, cdf_comp, _, dist_label = visualizer.binomial_dist()
            elif comparison_dists[i] == "Poisson":
                x_comp, pdf_comp, cdf_comp, _, dist_label = visualizer.poisson_dist()
            elif comparison_dists[i] == "Gamma":
                x_comp, pdf_comp, cdf_comp, _, dist_label = visualizer.gamma_dist()
            elif comparison_dists[i] == "Beta":
                x_comp, pdf_comp, cdf_comp, _, dist_label = visualizer.beta_dist()
            else:  # Chi-square
                x_comp, pdf_comp, cdf_comp, _, dist_label = visualizer.chi2_dist()
            
            fig_comp.add_trace(
                go.Scatter(x=x_comp, y=pdf_comp, mode='lines', 
                          name=dist_label, line=dict(color=colors[i % len(colors)], width=2)),
                row=1, col=1
            )
            
            fig_comp.add_trace(
                go.Scatter(x=x_comp, y=cdf_comp, mode='lines', 
                          name=dist_label, line=dict(color=colors[i % len(colors)], width=2),
                          showlegend=False),
                row=1, col=2
            )
        
        fig_comp.update_layout(height=500, title_text="Distribution Comparison", title_x=0.5)
        fig_comp.update_xaxes(title_text="x", row=1, col=1)
        fig_comp.update_xaxes(title_text="x", row=1, col=2)
        fig_comp.update_yaxes(title_text="f(x)", row=1, col=1)
        fig_comp.update_yaxes(title_text="F(x)", row=1, col=2)
        
        st.plotly_chart(fig_comp, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**Probability Distribution Visualizer** | "
        "Built with Streamlit, Plotly, and SciPy | "
        "Use the sliders to explore how parameters affect each distribution."
    )

if __name__ == "__main__":
    main()
