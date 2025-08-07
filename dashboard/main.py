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
    tabs = st.tabs(['Main', 'Objective', 'Questions', 'Surface', 'Optimisation'])

    with tabs[0]:
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

    with tabs[1]:
        ev_comp, obj_comp = draw_ev(df)
        st.subheader('Expected Value Comparison')
        st.plotly_chart(ev_comp, use_container_width=True, key='ev_comp_chart')
        st.subheader('Objective Comparison')
        st.plotly_chart(obj_comp, use_container_width=True, key='obj_comp_chart')

    with tabs[2]:
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

    with tabs[3]:
        curr = st.selectbox('Currency', options=CURRENCIES)
        taus, strikes, iv = fetch_vol_surface(curr)
        if taus.size and strikes.size and iv.size:
            st.plotly_chart(draw_surface(taus, strikes, iv, curr), use_container_width=True, key=f'surface_{curr}')
        else:
            st.error(f'No data for {curr}')

    with tabs[4]:
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
