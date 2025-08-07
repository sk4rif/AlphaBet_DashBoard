import streamlit as st
import datetime
import plotly.graph_objects as go
from helpers.mongo import pull_data

from .config import PAGE_TITLE, TIME_ZONE, REFRESH_INTERVAL, PLOT_COLUMNS, CURRENCIES
from .utils.data_fetcher import fetch_data, fetch_vol_surface
from .utils.plot_utils import create_base_fig
from .components.portfolio_charts import draw_portfolio, draw_chart
from .components.ev_charts import draw_ev, draw_question
from .components.optimization_charts import (
    draw_weight_evolution, draw_risk_metrics, draw_objective_metrics
)
from .components.surface_charts import draw_surface

# Streamlit setup
st.set_page_config(layout="wide", page_title=PAGE_TITLE)

# Comprehensive pitch black background styling
st.markdown("""
<style>
/* Force pitch black background on everything */
* {
    background-color: #000000 !important;
}

/* Main app background */
.stApp {
    background-color: #000000 !important;
}

/* Main content area */
.main, .main > div, .block-container {
    background-color: #000000 !important;
}

/* Sidebar */
.css-1d391kg, .css-1lcbmhc, section[data-testid="stSidebar"] {
    background-color: #000000 !important;
}

/* All divs and containers */
div, section, main {
    background-color: #000000 !important;
}

/* Tab containers and panels */
.stTabs, .stTabs > div, .stTabs [data-baseweb="tab-list"], .stTabs [data-baseweb="tab-panel"] {
    background-color: #000000 !important;
}

/* Column containers */
[data-testid="column"] {
    background-color: #000000 !important;
}

/* Vertical blocks */
[data-testid="stVerticalBlock"] {
    background-color: #000000 !important;
}

/* Header */
header[data-testid="stHeader"] {
    background-color: #000000 !important;
}

/* Ensure text remains visible */
.stMarkdown, .stText, h1, h2, h3, h4, h5, h6, p, span {
    color: #FFFFFF !important;
}

/* Input widgets - darker but visible */
.stSelectbox > div > div, .stDateInput > div > div, .stTextInput > div > div, .stNumberInput > div > div {
    background-color: #1a1a1a !important;
    color: #FFFFFF !important;
    border: 1px solid #333333 !important;
}

/* Buttons */
.stButton > button {
    background-color: #1a1a1a !important;
    color: #FFFFFF !important;
    border: 1px solid #333333 !important;
}

/* Override any white backgrounds */
[style*="background-color: white"], [style*="background-color: #ffffff"], [style*="background-color: #FFFFFF"] {
    background-color: #000000 !important;
}
</style>
""", unsafe_allow_html=True)

# Auto-refresh
if 'refresh_interval' not in st.session_state:
    st.session_state.refresh_interval = REFRESH_INTERVAL
st.empty().auto_refresh = True

def main():
    # Add logo to top left corner
    col1, col2 = st.columns([1, 7])
    with col1:
        try:
            st.image("dashboard/utils/AlphaBet_Logo.png", width=120)
        except FileNotFoundError:
            st.write("AlphaBet")
    
    now = datetime.datetime.now(TIME_ZONE)
    sd = st.sidebar.date_input('Start Date', value=now.date() - datetime.timedelta(days=7))
    ed = st.sidebar.date_input('End Date', value=now.date())
    if sd > ed:
        st.sidebar.error('Start <= End')
        sd = ed - datetime.timedelta(days=1)

    for label, days in [('24h',1),('7d',7),('30d',30),('90d',90)]:
        if st.sidebar.button(f'Last {label}'):
            sd, ed = (now - datetime.timedelta(days=days)).date(), now.date()

    df = fetch_data(sd, ed)
    if df.empty:
        st.warning('No data for selected range.')
        return

    st.markdown(f"### Data: {sd} to {ed}")
    tabs = st.tabs(['About', 'Main', 'Objective', 'Questions', 'Surface', 'Optimisation'])

    with tabs[0]:
        # Project Description Tab
        st.markdown("""
        # AlphaBet: Quantitative Cryptocurrency Derivatives Trading System
        
        ## System Overview
        
        AlphaBet is a production-grade algorithmic trading system designed for cryptocurrency derivatives and prediction markets, implementing advanced quantitative finance models with real-time execution capabilities. The system operates primarily on Polymarket, trading binary prediction contracts related to cryptocurrency price movements.
        
        ## Core Architecture
        
        ### 1. Market Data Infrastructure
        
        #### Multi-Exchange Data Aggregation
        - **Deribit**: Primary source for implied volatility data and options market structure
        - **Binance**: Spot and futures price feeds for underlying asset pricing
        - **Bybit**: Additional futures data for cross-validation and arbitrage detection
        - **Polymarket**: Prediction market contract data and order book information
        
        #### Real-Time Data Pipeline
        - **WebSocket Connections**: Asynchronous data streaming with automatic reconnection
        - **Data Validation**: Cross-exchange price verification and outlier detection
        - **Historical Data Persistence**: MongoDB storage for backtesting and model calibration
        - **Rate Limiting**: Intelligent API management to avoid exchange restrictions
        
        ### 2. Volatility Surface Construction
        
        #### SVI Model Implementation
        The system implements Gatheral's Stochastic Volatility Inspired (SVI) parameterization:
        
        ```
        σ²(k,τ) = a + b{ρ(k-m) + √[(k-m)² + σ²]}
        ```
        
        #### Technical Features
        - **Smile-by-Smile Fitting**: Sequential Least Squares Programming (SLSQP) optimization
        - **Arbitrage Prevention**: Calendar and butterfly arbitrage constraints
        - **Real-Time Updates**: Continuous recalibration from live Deribit IV data
        - **Surface Interpolation**: Cubic spline interpolation with bilinear surface construction
        
        ### 3. SJMCS: Stochastic Jump Monte Carlo Simulation
        
        #### Jump-Diffusion Model
        The core pricing engine implements a jump-diffusion process for cryptocurrency price evolution:
        
        ```
        dS(t) = μ(t)S(t)dt + σ(S,t)S(t)dW(t) + S(t-)θdN(t)
        ```
        
        #### Model Components
        - **μ(t)**: Risk-neutral drift extracted from futures curve interpolation
        - **σ(S,t)**: Local volatility interpolated from SVI surface in real-time
        - **λ**: Poisson jump intensity calibrated from historical cryptocurrency data
        - **θ**: Jump magnitude following log-normal distribution with mean reversion
        
        #### Advanced Simulation Features
        - **Log-Space Evolution**: Ensures numerical stability and positive price guarantee
        - **Dynamic Volatility Lookup**: Real-time surface interpolation during path evolution
        - **Variance Reduction Techniques**: Antithetic variates and control variates for improved convergence
        - **High-Performance Computing**: Numba JIT compilation enabling 100,000+ path simulations
        
        ### 4. Probability Estimation Engine
        
        #### Binary Contract Modeling
        The system specializes in binary prediction markets rather than traditional options:
        
        ```
        Probability = E[𝟙{S(T) > K}] where 𝟙 is the indicator function
        Expected Value = Probability × Payout - Contract Cost
        ```
        
        #### Advanced Probability Techniques
        - **Monte Carlo Integration**: Large-scale path simulation for accurate tail probability estimation
        - **Path-Dependent Contracts**: Handles barrier and Asian-style contract features
        - **Time-Dependent Parameters**: Dynamic jump intensity and volatility evolution
        - **Multi-Asset Correlation**: Cross-asset contract probability estimation using correlation matrices
        
        ### 5. Portfolio Optimization Framework
        
        #### Multi-Objective Optimization
        The optimization engine balances multiple competing objectives:
        
        ```
        Objective = Portfolio_EV - λ_risk × Portfolio_Risk - λ_concentration × HHI
        ```
        
        #### Components
        - **Expected Value Maximization**: Probability-weighted payoff optimization across all contracts
        - **Risk Minimization**: Volatility-based portfolio risk with full correlation matrix
        - **Concentration Penalty**: Herfindahl-Hirschman Index (HHI) preventing over-concentration
        - **Transaction Cost Integration**: Bid-ask spreads and market impact modeling
        
        #### Risk Metrics
        - **Portfolio Volatility**: σ_portfolio = √(w^T Σ w) using dynamic correlation matrices
        - **Concentration Risk**: HHI = Σ(w_i²) measuring portfolio diversification
        - **Theta Exposure**: Portfolio_Theta = Σ(w_i × θ_i) tracking time decay risk
        - **Greeks Aggregation**: Delta, gamma, and vega exposure across all positions
        
        #### Optimization Constraints
        - **Position Limits**: Maximum allocation per contract (typically 5-10% of portfolio)
        - **Liquidity Constraints**: Order book depth-based position sizing
        - **Sector Limits**: Maximum exposure per cryptocurrency or event category
        - **Kelly Criterion**: Fractional Kelly position sizing with uncertainty adjustments
        
        ### 6. Execution Engine
        
        #### Order Management System
        - **Polymarket Integration**: Direct API execution with sophisticated order routing
        - **Order Types**: Market, limit, and conditional orders with partial fill handling
        - **Slippage Modeling**: Real-time market impact estimation based on order book depth
        - **Fill Monitoring**: Comprehensive order status tracking and execution quality analysis
        
        #### Rebalancing Logic
        The system employs sophisticated rebalancing decisions:
        
        ```
        EV_Improvement = New_Portfolio_EV - Current_Portfolio_EV
        Transaction_Costs = Σ(Bid_Ask_Spreads + Market_Impact + Fees)
        Rebalance_Threshold = Transaction_Costs × (1 + λ_buffer)
        Decision Rule: Rebalance if EV_Improvement > Rebalance_Threshold
        ```
        
        ### 7. Risk Management System
        
        #### Real-Time Risk Monitoring
        - **Portfolio Greeks**: Aggregated delta, theta, gamma, and vega exposure tracking
        - **Value-at-Risk**: Monte Carlo simulation-based VaR calculation with multiple confidence levels
        - **Stress Testing**: Scenario analysis under extreme market conditions and volatility spikes
        - **Correlation Monitoring**: Dynamic correlation matrix updates and regime change detection
        
        #### Position-Level Risk Assessment
        - **Individual Contract Risk**: Volatility-adjusted position sizing based on time to expiry
        - **Portfolio Risk**: Full correlation matrix-based risk calculation with cross-asset effects
        - **Concentration Limits**: Real-time monitoring of single-contract and sector exposure
        - **Drawdown Controls**: Dynamic position sizing based on recent performance metrics
        
        ### 8. Performance Analytics
        
        #### Real-Time Dashboard
        - **Portfolio Metrics**: Live P&L, position details, and risk exposure visualization
        - **Risk Analytics**: 3D volatility surfaces, correlation heatmaps, and Greeks exposure charts
        - **Optimization History**: Weight evolution tracking and objective function convergence analysis
        - **Performance Attribution**: Contract-level and strategy-level return decomposition
        
        #### Key Performance Indicators
        - **Sharpe Ratio**: Risk-adjusted returns measurement with rolling window analysis
        - **Information Ratio**: Active return per unit of tracking error versus benchmark
        - **Maximum Drawdown**: Peak-to-trough loss analysis with recovery time tracking
        - **Hit Rate**: Percentage of profitable positions with statistical significance testing
        - **Expected Value Realization**: Actual versus predicted outcomes with model validation
        
        ### 9. Unique System Characteristics
        
        #### Prediction Market Specialization
        Unlike traditional options trading systems, AlphaBet is specifically designed for binary prediction markets:
        
        - **Event-Based Contracts**: "Will Bitcoin reach $X by date Y?" with binary outcomes
        - **Fixed Payout Structure**: Typically $1.00 for correct predictions, creating unique risk-return profiles
        - **Probability-Centric Approach**: Focus on event probability estimation rather than traditional Greeks
        - **Time-Sensitive Execution**: Contracts with specific expiration dates tied to real-world events
        
        #### Cryptocurrency Market Adaptation
        - **High Volatility Modeling**: Jump-diffusion models specifically calibrated for crypto price dynamics
        - **24/7 Operation**: Continuous trading and risk monitoring across global time zones
        - **Cross-Exchange Arbitrage**: Multi-venue price discovery and execution optimization
        - **Regulatory Compliance**: Designed for decentralized prediction market platforms
        
        #### Advanced Quantitative Features
        - **Multi-Asset Correlation**: Cross-cryptocurrency correlation modeling for portfolio construction
        - **Regime Detection**: Dynamic parameter adjustment based on market regime identification
        - **Tail Risk Management**: Specialized handling of extreme events common in cryptocurrency markets
        - **Model Validation**: Continuous backtesting and out-of-sample performance monitoring
        
        ## Conclusion
        
        This system represents a cutting-edge quantitative trading platform that combines traditional derivatives pricing theory with modern computational finance techniques, specifically optimized for the unique characteristics of cryptocurrency prediction markets and binary outcome contracts.
        """)
    
    with tabs[1]:
        # FYI Notice about data retention
        st.info("ℹ️ **FYI**: Data is currently wiped every week for maintenance purposes.")
        
        st.subheader('Asset Under Management')
        st.plotly_chart(draw_portfolio(df), use_container_width=True, key='portfolio_chart')
        # Show weighted averages in a single graph
        st.subheader('Weighted Averages')
        fig = create_base_fig('Weighted Averages')
        if 'weighted_probability' in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['weighted_probability'],
                name='Weighted Probability',
                mode='lines+markers'
            ))
        if 'weighted_best_ask' in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['weighted_best_ask'],
                name='Weighted Best Ask',
                mode='lines+markers'
            ))
        if 'weighted_difference' in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['weighted_difference'],
                name='Probability - Best Ask',
                mode='lines+markers',
                yaxis='y2'  # Use secondary y-axis
            ))
            # Add secondary y-axis
            fig.update_layout(
                yaxis2=dict(
                    title='Difference',
                    overlaying='y',
                    side='right'
                )
            )
        st.plotly_chart(fig, use_container_width=True, key='weighted_metrics')
        
        # Show remaining metrics
        cols = st.columns(2)
        remaining_cols = [col for col in PLOT_COLUMNS[:4] if col not in ['probability', 'best_ask']]
        for i, col in enumerate(remaining_cols):
            with cols[i%2]:
                st.subheader(col)
                st.plotly_chart(draw_chart(df, col, stack=True), use_container_width=True, key=f'main_metric_{col}')
        if st.checkbox('Show more metrics'):
            for i, col in enumerate(PLOT_COLUMNS[4:]):
                with cols[i%2]:
                    st.subheader(col)
                    st.plotly_chart(draw_chart(df, col, stack=(col not in ['Probability','best_ask'])), use_container_width=True, key=f'main_metric_{col}')



    with tabs[2]:
        ev_comp, obj_comp = draw_ev(df)
        st.subheader('Expected Value Comparison')
        st.plotly_chart(ev_comp, use_container_width=True, key='ev_comp_chart')
        st.subheader('Objective Comparison')
        st.plotly_chart(obj_comp, use_container_width=True, key='obj_comp_chart')

    with tabs[3]:
        # Get questions only from latest timestamp
        latest_ts = df.index.max()
        questions = sorted(df.loc[latest_ts]['question'].dropna().astype(str).unique())
        search = st.text_input('Search questions').strip().lower()
        if search:
            questions = [q for q in questions if search in q.lower()]
        total_pages = max(1, (len(questions)-1)//4 + 1)
        page = st.number_input('Page', min_value=1, max_value=total_pages, value=1)
        display_qs = questions[(page-1)*4 : page*4]
        cols = st.columns(2)
        for i, q in enumerate(display_qs):
            with cols[i%2]:
                st.subheader(q)
                st.plotly_chart(draw_question(df, q), use_container_width=True, key=f'question_{q}')

    with tabs[4]:
        curr = st.selectbox('Currency', options=CURRENCIES)
        taus, strikes, iv = fetch_vol_surface(curr)
        if taus.size and strikes.size and iv.size:
            st.plotly_chart(draw_surface(taus, strikes, iv, curr), use_container_width=True, key=f'surface_{curr}')
        else:
            st.error(f'No data for {curr}')

    with tabs[5]:
        try:
            opti_data = pull_data("AlphaBet", "OPTI", "RESULTS")
            if not opti_data or 'history' not in opti_data or 'questions' not in opti_data:
                st.warning('No optimization data available.')
                return
            st.subheader('Weight Evolution')
            st.plotly_chart(draw_weight_evolution(opti_data['history'], opti_data['questions']), use_container_width=True, key='weight_evolution')
            
            st.subheader('Risk Metrics Evolution')
            st.plotly_chart(draw_risk_metrics(opti_data['history']), use_container_width=True, key='risk_metrics')
            
            st.subheader('Objective Metrics Evolution')
            st.plotly_chart(draw_objective_metrics(opti_data['history']), use_container_width=True, key='objective_metrics')
            
            # Display optimization timestamp
            if 'timestamp' in opti_data:
                st.markdown(f"*Last optimization: {opti_data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}*")
        except Exception as e:
            st.error('Failed to load optimization data. Please try again later.')
            return

    if st.sidebar.button('🔄 Refresh'):
        st.experimental_rerun()

if __name__ == '__main__':
    main()
