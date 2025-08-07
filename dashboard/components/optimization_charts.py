import plotly.graph_objects as go
from ..utils.plot_utils import create_base_fig

def draw_weight_evolution(history, questions):
    fig = create_base_fig('Weight')
    for i, q in enumerate(questions):
        fig.add_trace(go.Scatter(
            y=[w[i] for w in history['weights']],
            name=q,
            mode='lines'
        ))
    return fig

def draw_risk_metrics(history):
    fig = create_base_fig('Risk Metrics')
    # Add traces for metrics on primary y-axis
    for metric in ['maturity', 'concentration', 'probability', 'theta']:
        fig.add_trace(go.Scatter(
            y=history[metric],
            name=metric.capitalize(),
            mode='lines'
        ))
    # Add volatility on secondary y-axis
    fig.add_trace(go.Scatter(
        y=history['volatility'],
        name='Volatility',
        mode='lines',
        yaxis='y2'
    ))
    fig.update_layout(
        yaxis2=dict(title='Volatility', overlaying='y', side='right')
    )
    return fig

def draw_objective_metrics(history):
    fig = create_base_fig('EV')
    # Add EV and objective on primary y-axis
    for metric in ['ev', 'objective']:
        fig.add_trace(go.Scatter(
            y=history[metric],
            name=metric.upper() if metric == 'ev' else metric.capitalize(),
            mode='lines'
        ))
    # Add risk on secondary y-axis
    fig.add_trace(go.Scatter(
        y=history['risk'],
        name='Risk',
        mode='lines',
        yaxis='y2'
    ))
    fig.update_layout(
        yaxis2=dict(title='Risk', overlaying='y', side='right')
    )
    return fig
