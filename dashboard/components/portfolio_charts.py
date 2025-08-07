import plotly.graph_objects as go
from ..utils.plot_utils import create_base_fig

def draw_portfolio(df):
    fig = create_base_fig(yaxis_title="Value")
    if df.empty or not {'cash', 'portfolio_value'}.issubset(df.columns):
        fig.add_annotation(text="Missing portfolio data", showarrow=False)
        return fig
    
    cash = df['cash'].dropna()
    assets = df['portfolio_value'].dropna()
    
    if not cash.empty:
        fig.add_trace(go.Scatter(x=cash.index, y=cash.values, mode='lines', name='cash', stackgroup='one'))
    if not assets.empty:
        fig.add_trace(go.Scatter(x=assets.index, y=assets.values, mode='lines', name='Assets', stackgroup='one'))
    if not cash.empty and not assets.empty:
        total = cash.reindex(df.index, fill_value=0) + assets.reindex(df.index, fill_value=0)
        fig.add_trace(go.Scatter(x=total.index, y=total.values, mode='lines', name='Total', line=dict(dash='dash', width=2)))
    return fig

def draw_chart(df, col, stack=False):
    fig = create_base_fig(yaxis_title=col)
    if df.empty or col not in df.columns or 'question' not in df.columns:
        fig.add_annotation(text=f"No data for '{col}'", showarrow=False)
        return fig
        
    pivot = df.pivot_table(index=df.index, columns='question', values=col, aggfunc='sum')
    mode = 'lines+markers' if col in ['probability', 'best_ask'] else 'lines'
    # Never stack probability or best_ask
    group = 'one' if stack and col not in ['probability', 'best_ask'] else None
    
    for q, series in pivot.items():
        data = series.dropna()
        if not data.empty:
            fig.add_trace(go.Scatter(x=data.index, y=data.values, mode=mode, name=q, stackgroup=group))
    return fig
