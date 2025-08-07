import plotly.graph_objects as go

# Base Plotly Layout
BASE_LAYOUT = dict(
    template='plotly_dark', paper_bgcolor='black', plot_bgcolor='black',
    xaxis=dict(title='Timestamp'), legend=dict(bgcolor='rgba(0,0,0,0)', x=1.02, y=1)
)

def create_base_fig(yaxis_title=None):
    fig = go.Figure()
    layout = BASE_LAYOUT.copy()
    if yaxis_title:
        layout['yaxis'] = dict(title=yaxis_title)
    fig.update_layout(**layout)
    return fig
