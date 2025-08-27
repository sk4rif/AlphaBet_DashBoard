import plotly.graph_objects as go
from ..utils.plot_utils import create_base_fig

def draw_ev(df):
    # EV Comparison
    ev_comp = create_base_fig(yaxis_title="EV")
    for col in ['EV_current', 'EV_target']:
        if col in df.columns:
            data = df[col].dropna()
            ev_comp.add_trace(go.Scatter(x=data.index, y=data.values, mode='lines+markers',
                                       name='Current' if col == 'EV_current' else 'Target'))
            
    # Objective Comparison
    obj_comp = create_base_fig(yaxis_title="Objective")
    for col in ['current_objective', 'target_objective']:
        if col in df.columns:
            data = df[col].dropna()
            obj_comp.add_trace(go.Scatter(x=data.index, y=data.values, mode='lines+markers',
                                        name='Current' if col == 'current_objective' else 'Target'))
    
    return ev_comp, obj_comp

def draw_question(df, q):
    fig = create_base_fig(yaxis_title="Value")
    qdf = df[df['question'].astype(str) == str(q)]
    if qdf.empty:
        fig.add_annotation(text=f"No data for {q}", showarrow=False)
        return fig
        
    for col in ['probability', 'best_bid']:
        if col in qdf.columns:
            data = qdf[col].dropna()
            fig.add_trace(go.Scatter(x=data.index, y=data.values, mode='lines+markers', name=col))
    return fig
