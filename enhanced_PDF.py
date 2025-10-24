import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats
import pandas as pd
import time
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Probability Distribution Visualizer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2c3e50;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
        margin: 2rem 0 1rem 0;
    }
    .info-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .distribution-info {
        font-size: 1rem;
        line-height: 1.6;
    }
    .stat-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem;
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

class EnhancedDistributionVisualizer:
    def __init__(self):
        self.distributions = {
            "Normal": self.normal_dist,
            "Uniform": self.uniform_dist,
            "Exponential": self.exponential_dist,
            "Binomial": self.binomial_dist,
            "Poisson": self.poisson_dist,
            "Gamma": self.gamma_dist,
            "Beta": self.beta_dist,
            "Chi-square": self.chi2_dist,
            "Student's t": self.t_dist,
            "Log-Normal": self.lognormal_dist
        }
        
        self.distribution_info = {
            "Normal": {
                "description": "The normal distribution is symmetric and bell-shaped. It's characterized by mean (μ) and standard deviation (σ). Many natural phenomena follow this distribution.",
                "parameters": ["μ (mean): Center of the distribution", "σ (standard deviation): Spread of the distribution"],
                "applications": ["Height/weight measurements", "Test scores", "Measurement errors"]
            },
            "Uniform": {
                "description": "All outcomes are equally likely within a specified range [a, b]. The PDF is constant across the interval.",
                "parameters": ["a (lower bound): Minimum value", "b (upper bound): Maximum value"],
                "applications": ["Random number generation", "Monte Carlo simulations", "Quality control"]
            },
            "Exponential": {
                "description": "Models the time between events in a Poisson process. Characterized by rate parameter λ (lambda). Memoryless property.",
                "parameters": ["λ (rate): Average number of events per unit time"],
                "applications": ["Waiting times", "Equipment failure times", "Radioactive decay"]
            },
            "Binomial": {
                "description": "Models the number of successes in n independent trials with probability p of success on each trial.",
                "parameters": ["n (trials): Number of independent trials", "p (probability): Probability of success in each trial"],
                "applications": ["Quality control", "Survey analysis", "Clinical trials"]
            },
            "Poisson": {
                "description": "Models the number of events occurring in a fixed interval of time/space with a known constant rate λ.",
                "parameters": ["λ (rate): Average number of events in the interval"],
                "applications": ["Call center arrivals", "Traffic flow", "Natural disaster occurrences"]
            },
            "Gamma": {
                "description": "Generalizes the exponential distribution. Useful for modeling waiting times, insurance claims, and rainfall amounts.",
                "parameters": ["α (shape): Shape parameter", "β (scale): Scale parameter"],
                "applications": ["Insurance claims", "Rainfall modeling", "Wait time analysis"]
            },
            "Beta": {
                "description": "A flexible distribution defined on [0,1] interval. Useful for modeling probabilities and proportions.",
                "parameters": ["α (shape): First shape parameter", "β (shape): Second shape parameter"],
                "applications": ["Bayesian analysis", "Project planning", "Proportion modeling"]
            },
            "Chi-square": {
                "description": "Sum of squares of k independent standard normal variables. Used in hypothesis testing and confidence intervals.",
                "parameters": ["k (degrees of freedom): Number of independent standard normal variables"],
                "applications": ["Hypothesis testing", "Goodness-of-fit tests", "Confidence intervals"]
            },
            "Student's t": {
                "description": "Similar to normal distribution but with heavier tails. Used when sample sizes are small and population variance is unknown.",
                "parameters": ["ν (degrees of freedom): Controls the tail thickness"],
                "applications": ["Small sample statistics", "Confidence intervals", "Hypothesis testing"]
            },
            "Log-Normal": {
                "description": "A distribution whose logarithm is normally distributed. Useful for modeling quantities that must be positive.",
                "parameters": ["μ (mean of log): Mean of the underlying normal", "σ (std of log): Std dev of the underlying normal"],
                "applications": ["Income distribution", "Stock prices", "Size of living tissue"]
            }
        }
        
        self.color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                             '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    def normal_dist(self):
        col1, col2, col3 = st.columns(3)
        with col1:
            mu = st.slider("Mean (μ)", -5.0, 5.0, 0.0, 0.1, key="normal_mu", 
                          help="Center of the distribution")
        with col2:
            sigma = st.slider("Standard Deviation (σ)", 0.1, 5.0, 1.0, 0.1, key="normal_sigma",
                             help="Spread of the distribution")
        with col3:
            show_sigma = st.checkbox("Show σ intervals", True, key="normal_sigma_int")
        
        x = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)
        pdf = stats.norm.pdf(x, mu, sigma)
        cdf = stats.norm.cdf(x, mu, sigma)
        samples = stats.norm.rvs(mu, sigma, 1000)
        
        return x, pdf, cdf, samples, f"Normal(μ={mu}, σ={sigma})", show_sigma
    
    def uniform_dist(self):
        col1, col2 = st.columns(2)
        with col1:
            a = st.slider("Lower bound (a)", -10.0, 5.0, 0.0, 0.1, key="uniform_a",
                         help="Minimum value of the distribution")
        with col2:
            b = st.slider("Upper bound (b)", a + 0.1, 10.0, 1.0, 0.1, key="uniform_b",
                         help="Maximum value of the distribution")
        
        x = np.linspace(a - 1, b + 1, 1000)
        pdf = stats.uniform.pdf(x, a, b - a)
        cdf = stats.uniform.cdf(x, a, b - a)
        samples = stats.uniform.rvs(a, b - a, 1000)
        
        return x, pdf, cdf, samples, f"Uniform(a={a}, b={b})", False
    
    def exponential_dist(self):
        col1, col2 = st.columns(2)
        with col1:
            lam = st.slider("Rate parameter (λ)", 0.1, 5.0, 1.0, 0.1, key="exp_lambda",
                           help="Average number of events per unit time")
        with col2:
            show_mean = st.checkbox("Show mean (1/λ)", True, key="exp_mean")
        
        x = np.linspace(0, 10/lam, 1000)
        pdf = stats.expon.pdf(x, scale=1/lam)
        cdf = stats.expon.cdf(x, scale=1/lam)
        samples = stats.expon.rvs(scale=1/lam, size=1000)
        
        return x, pdf, cdf, samples, f"Exponential(λ={lam})", show_mean
    
    def binomial_dist(self):
        col1, col2, col3 = st.columns(3)
        with col1:
            n = st.slider("Number of trials (n)", 1, 100, 20, 1, key="binom_n",
                         help="Total number of independent trials")
        with col2:
            p = st.slider("Probability of success (p)", 0.0, 1.0, 0.5, 0.01, key="binom_p",
                         help="Probability of success in each trial")
        with col3:
            show_pmf = st.checkbox("Show PMF points", True, key="binom_pmf")
        
        x = np.arange(0, n + 1)
        pdf = stats.binom.pmf(x, n, p)
        cdf = stats.binom.cdf(x, n, p)
        samples = stats.binom.rvs(n, p, size=1000)
        
        return x, pdf, cdf, samples, f"Binomial(n={n}, p={p})", show_pmf
    
    def poisson_dist(self):
        col1, col2 = st.columns(2)
        with col1:
            lam = st.slider("Rate parameter (λ)", 0.1, 30.0, 5.0, 0.1, key="poisson_lambda",
                           help="Average number of events in the interval")
        with col2:
            show_pmf = st.checkbox("Show PMF points", True, key="poisson_pmf")
        
        x = np.arange(0, int(3 * lam) + 10)
        pdf = stats.poisson.pmf(x, lam)
        cdf = stats.poisson.cdf(x, lam)
        samples = stats.poisson.rvs(lam, size=1000)
        
        return x, pdf, cdf, samples, f"Poisson(λ={lam})", show_pmf
    
    def gamma_dist(self):
        col1, col2, col3 = st.columns(3)
        with col1:
            alpha = st.slider("Shape (α)", 0.1, 10.0, 2.0, 0.1, key="gamma_alpha",
                             help="Shape parameter of the distribution")
        with col2:
            beta = st.slider("Scale (β)", 0.1, 5.0, 1.0, 0.1, key="gamma_beta",
                            help="Scale parameter of the distribution")
        with col3:
            show_mean = st.checkbox("Show mean (αβ)", True, key="gamma_mean")
        
        x = np.linspace(0, stats.gamma.ppf(0.99, alpha, scale=beta), 1000)
        pdf = stats.gamma.pdf(x, alpha, scale=beta)
        cdf = stats.gamma.cdf(x, alpha, scale=beta)
        samples = stats.gamma.rvs(alpha, scale=beta, size=1000)
        
        return x, pdf, cdf, samples, f"Gamma(α={alpha}, β={beta})", show_mean
    
    def beta_dist(self):
        col1, col2, col3 = st.columns(3)
        with col1:
            alpha = st.slider("Shape α", 0.1, 10.0, 2.0, 0.1, key="beta_alpha",
                             help="First shape parameter")
        with col2:
            beta = st.slider("Shape β", 0.1, 10.0, 2.0, 0.1, key="beta_beta",
                            help="Second shape parameter")
        with col3:
            show_mean = st.checkbox("Show mean", True, key="beta_mean")
        
        x = np.linspace(0, 1, 1000)
        pdf = stats.beta.pdf(x, alpha, beta)
        cdf = stats.beta.cdf(x, alpha, beta)
        samples = stats.beta.rvs(alpha, beta, size=1000)
        
        return x, pdf, cdf, samples, f"Beta(α={alpha}, β={beta})", show_mean
    
    def chi2_dist(self):
        col1, col2 = st.columns(2)
        with col1:
            df = st.slider("Degrees of freedom (k)", 1, 50, 5, 1, key="chi2_df",
                          help="Number of independent standard normal variables")
        with col2:
            show_mean = st.checkbox("Show mean (k)", True, key="chi2_mean")
        
        x = np.linspace(0, stats.chi2.ppf(0.99, df), 1000)
        pdf = stats.chi2.pdf(x, df)
        cdf = stats.chi2.cdf(x, df)
        samples = stats.chi2.rvs(df, size=1000)
        
        return x, pdf, cdf, samples, f"Chi-square(k={df})", show_mean
    
    def t_dist(self):
        col1, col2 = st.columns(2)
        with col1:
            df = st.slider("Degrees of freedom (ν)", 1, 50, 10, 1, key="t_df",
                          help="Controls the thickness of the tails")
        with col2:
            show_normal = st.checkbox("Compare with Normal", True, key="t_normal")
        
        x = np.linspace(-4, 4, 1000)
        pdf = stats.t.pdf(x, df)
        cdf = stats.t.cdf(x, df)
        samples = stats.t.rvs(df, size=1000)
        
        return x, pdf, cdf, samples, f"Student's t(ν={df})", show_normal
    
    def lognormal_dist(self):
        col1, col2, col3 = st.columns(3)
        with col1:
            mu = st.slider("Mean of log (μ)", -1.0, 2.0, 0.0, 0.1, key="lognorm_mu",
                          help="Mean of the underlying normal distribution")
        with col2:
            sigma = st.slider("Std of log (σ)", 0.1, 2.0, 1.0, 0.1, key="lognorm_sigma",
                             help="Standard deviation of the underlying normal distribution")
        with col3:
            show_stats = st.checkbox("Show distribution stats", True, key="lognorm_stats")
        
        x = np.linspace(0, stats.lognorm.ppf(0.99, sigma, scale=np.exp(mu)), 1000)
        pdf = stats.lognorm.pdf(x, sigma, scale=np.exp(mu))
        cdf = stats.lognorm.cdf(x, sigma, scale=np.exp(mu))
        samples = stats.lognorm.rvs(sigma, scale=np.exp(mu), size=1000)
        
        return x, pdf, cdf, samples, f"Log-Normal(μ={mu}, σ={sigma})", show_stats

    def calculate_statistics(self, samples, dist_name):
        stats_dict = {
            "Mean": np.mean(samples),
            "Variance": np.var(samples),
            "Standard Deviation": np.std(samples),
            "Skewness": stats.skew(samples),
            "Kurtosis": stats.kurtosis(samples),
            "Minimum": np.min(samples),
            "Maximum": np.max(samples),
            "Sample Size": len(samples)
        }
        return stats_dict

def main():
    st.markdown('<h1 class="main-header">🎯 Advanced Probability Distribution Visualizer</h1>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'last_update' not in st.session_state:
        st.session_state.last_update = datetime.now()
    
    visualizer = EnhancedDistributionVisualizer()
    
    # Sidebar with enhanced layout
    with st.sidebar:
        st.markdown('<div class="section-header">🎛️ Distribution Settings</div>', unsafe_allow_html=True)
        
        # Distribution selection with search
        selected_dist = st.selectbox(
            "Select Distribution",
            list(visualizer.distributions.keys()),
            help="Choose a probability distribution to visualize"
        )
        
        # Sample size control
        st.markdown("---")
        st.subheader("Sampling Settings")
        sample_size = st.slider("Sample Size", 100, 10000, 1000, 100,
                               help="Number of random samples to generate")
        
        # Animation and real-time updates
        st.subheader("Visualization Options")
        auto_update = st.checkbox("Auto-update on parameter change", True)
        show_animations = st.checkbox("Show animations", True)
        
        # Multiple distributions comparison
        st.markdown("---")
        st.markdown('<div class="section-header">📊 Comparison Mode</div>', unsafe_allow_html=True)
        compare_mode = st.checkbox("Enable Distribution Comparison")
        
        if compare_mode:
            num_comparisons = st.slider("Number of distributions", 2, 6, 3)
            comparison_dists = []
            
            for i in range(num_comparisons):
                dist = st.selectbox(
                    f"Distribution {i+1}",
                    list(visualizer.distributions.keys()),
                    key=f"comp_dist_{i}",
                    index=i if i < len(visualizer.distributions) else 0
                )
                comparison_dists.append(dist)
        
        # Enhanced information box
        st.markdown("---")
        with st.expander("📚 Distribution Information", expanded=True):
            info = visualizer.distribution_info[selected_dist]
            st.markdown(f'<div class="distribution-info">'
                       f'<h4>📖 Description</h4><p>{info["description"]}</p>'
                       f'<h4>⚙️ Parameters</h4><ul>{"".join([f"<li>{param}</li>" for param in info["parameters"]])}</ul>'
                       f'<h4>🎯 Applications</h4><ul>{"".join([f"<li>{app}</li>" for app in info["applications"]])}</ul>'
                       f'</div>', unsafe_allow_html=True)
    
    # Main content area
    if not compare_mode:
        # Single distribution view with enhanced features
        with st.spinner("Generating distribution..."):
            x, pdf, cdf, samples, dist_label, extra_param = visualizer.distributions[selected_dist]()
            samples = samples[:sample_size]  # Adjust sample size
        
        # Create enhanced subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                f'📈 Probability Density Function (PDF) - {dist_label}',
                f'📊 Cumulative Distribution Function (CDF) - {dist_label}',
                f'📋 Histogram with Theoretical PDF',
                f'🔄 Empirical vs Theoretical CDF'
            ),
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        # Enhanced PDF plot
        fig.add_trace(
            go.Scatter(
                x=x, y=pdf, mode='lines', name='PDF',
                line=dict(color=visualizer.color_palette[0], width=3),
                hovertemplate='<b>x</b>: %{x:.3f}<br><b>f(x)</b>: %{y:.3f}<extra></extra>',
                fill='tozeroy', fillcolor=f'{visualizer.color_palette[0]}20'
            ),
            row=1, col=1
        )
        
        # Enhanced CDF plot
        fig.add_trace(
            go.Scatter(
                x=x, y=cdf, mode='lines', name='CDF',
                line=dict(color=visualizer.color_palette[1], width=3),
                hovertemplate='<b>x</b>: %{x:.3f}<br><b>F(x)</b>: %{y:.3f}<extra></extra>'
            ),
            row=1, col=2
        )
        
        # Enhanced histogram with theoretical PDF
        fig.add_trace(
            go.Histogram(
                x=samples, histnorm='probability density', 
                name='Empirical Distribution', nbinsx=min(50, len(np.unique(samples))),
                opacity=0.7, marker_color=visualizer.color_palette[2],
                hovertemplate='<b>Range</b>: %{x}<br><b>Density</b>: %{y:.3f}<extra></extra>'
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=x, y=pdf, mode='lines', name='Theoretical PDF',
                line=dict(color=visualizer.color_palette[0], width=3)
            ),
            row=2, col=1
        )
        
        # Enhanced empirical vs theoretical CDF
        sorted_samples = np.sort(samples)
        empirical_cdf = np.arange(1, len(sorted_samples) + 1) / len(sorted_samples)
        
        fig.add_trace(
            go.Scatter(
                x=sorted_samples, y=empirical_cdf, mode='lines', 
                name='Empirical CDF', line=dict(color=visualizer.color_palette[3], width=3),
                hovertemplate='<b>x</b>: %{x:.3f}<br><b>F(x)</b>: %{y:.3f}<extra></extra>'
            ),
            row=2, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=x, y=cdf, mode='lines', name='Theoretical CDF',
                line=dict(color=visualizer.color_palette[1], width=3, dash='dash'),
                hovertemplate='<b>x</b>: %{x:.3f}<br><b>F(x)</b>: %{y:.3f}<extra></extra>'
            ),
            row=2, col=2
        )
        
        # Update layout with enhanced styling
        fig.update_layout(
            height=900,
            showlegend=True,
            title_text=f"🎯 {dist_label} Distribution Analysis",
            title_x=0.5,
            template="plotly_white",
            font=dict(size=12),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Update axes with better labels
        fig.update_xaxes(title_text="x", row=1, col=1)
        fig.update_xaxes(title_text="x", row=1, col=2)
        fig.update_xaxes(title_text="x", row=2, col=1)
        fig.update_xaxes(title_text="x", row=2, col=2)
        
        fig.update_yaxes(title_text="Probability Density f(x)", row=1, col=1)
        fig.update_yaxes(title_text="Cumulative Probability F(x)", row=1, col=2)
        fig.update_yaxes(title_text="Density", row=2, col=1)
        fig.update_yaxes(title_text="Probability", row=2, col=2)
        
        # Display the enhanced plot
        st.plotly_chart(fig, use_container_width=True)
        
        # Enhanced statistics section
        st.markdown('<div class="section-header">📊 Summary Statistics</div>', unsafe_allow_html=True)
        
        stats_data = visualizer.calculate_statistics(samples, selected_dist)
        
        # Create metric columns
        cols = st.columns(4)
        metric_keys = list(stats_data.keys())[:8]  # Show first 8 metrics
        
        for i, key in enumerate(metric_keys):
            with cols[i % 4]:
                value = stats_data[key]
                if key in ["Mean", "Variance", "Standard Deviation"]:
                    display_value = f"{value:.4f}"
                elif key in ["Skewness", "Kurtosis"]:
                    display_value = f"{value:.3f}"
                else:
                    display_value = f"{value:,.0f}" if isinstance(value, (int, np.integer)) else f"{value:.2f}"
                
                st.metric(
                    label=key,
                    value=display_value,
                    delta=None
                )
        
        # Data preview
        with st.expander("🔍 Sample Data Preview"):
            preview_df = pd.DataFrame(samples[:100], columns=['Sample Values'])
            st.dataframe(preview_df.style.format({'Sample Values': '{:.4f}'}), height=300)
            
            # Download option
            csv = preview_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Sample Data (CSV)",
                data=csv,
                file_name=f"{selected_dist.lower()}_samples.csv",
                mime="text/csv"
            )
    
    else:
        # Enhanced comparison mode
        st.markdown('<div class="section-header">📊 Distribution Comparison</div>', unsafe_allow_html=True)
        
        fig_comp = make_subplots(
            rows=1, cols=2,
            subplot_titles=('📈 PDF Comparison', '📊 CDF Comparison'),
            horizontal_spacing=0.1
        )
        
        for i, dist_name in enumerate(comparison_dists):
            # Get distribution data
            x_comp, pdf_comp, cdf_comp, _, dist_label, _ = visualizer.distributions[dist_name]()
            
            fig_comp.add_trace(
                go.Scatter(
                    x=x_comp, y=pdf_comp, mode='lines', 
                    name=dist_label, 
                    line=dict(color=visualizer.color_palette[i % len(visualizer.color_palette)], width=3),
                    hovertemplate=f'<b>{dist_label}</b><br>x: %{{x:.3f}}<br>f(x): %{{y:.3f}}<extra></extra>'
                ),
                row=1, col=1
            )
            
            fig_comp.add_trace(
                go.Scatter(
                    x=x_comp, y=cdf_comp, mode='lines', 
                    name=dist_label, 
                    line=dict(color=visualizer.color_palette[i % len(visualizer.color_palette)], width=3),
                    hovertemplate=f'<b>{dist_label}</b><br>x: %{{x:.3f}}<br>F(x): %{{y:.3f}}<extra></extra>',
                    showlegend=False
                ),
                row=1, col=2
            )
        
        fig_comp.update_layout(
            height=600,
            title_text="🎯 Distribution Comparison Analysis",
            title_x=0.5,
            template="plotly_white",
            showlegend=True
        )
        
        fig_comp.update_xaxes(title_text="x", row=1, col=1)
        fig_comp.update_xaxes(title_text="x", row=1, col=2)
        fig_comp.update_yaxes(title_text="Probability Density f(x)", row=1, col=1)
        fig_comp.update_yaxes(title_text="Cumulative Probability F(x)", row=1, col=2)
        
        st.plotly_chart(fig_comp, use_container_width=True)
        
        # Comparison statistics table
        st.subheader("📋 Comparison Statistics")
        comp_stats = []
        for dist_name in comparison_dists:
            _, _, _, samples, dist_label, _ = visualizer.distributions[dist_name]()
            stats_data = visualizer.calculate_statistics(samples, dist_name)
            stats_data['Distribution'] = dist_label
            comp_stats.append(stats_data)
        
        comp_df = pd.DataFrame(comp_stats)
        comp_df = comp_df[['Distribution'] + [col for col in comp_df.columns if col != 'Distribution']]
        st.dataframe(comp_df.style.format({col: '{:.4f}' for col in comp_df.select_dtypes(include=[np.number]).columns}))
    
    # Enhanced footer
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(
            "<div style='text-align: center;'>"
            "🎯 <b>Advanced Probability Distribution Visualizer</b> | "
            "Built with ❤️ using Streamlit, Plotly, and SciPy | "
            "Perfect for students, researchers, and data scientists"
            "</div>",
            unsafe_allow_html=True
        )

if __name__ == "__main__":
    main()
