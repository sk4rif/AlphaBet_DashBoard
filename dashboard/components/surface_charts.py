import plotly.graph_objects as go

def draw_surface(taus, strikes, iv, curr):
    fig = go.Figure(data=[go.Surface(x=strikes, y=taus, z=iv, showscale=True)])
    fig.update_layout(title=f"{curr} SVI Surface",
                     scene=dict(xaxis_title='Strike', yaxis_title='Tau', zaxis_title='IV'),
                     paper_bgcolor='black')
    return fig
