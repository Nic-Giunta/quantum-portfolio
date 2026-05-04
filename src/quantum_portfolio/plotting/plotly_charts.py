def plotly_weights(weights):
    try:
        import plotly.express as px
    except ImportError as exc:
        raise ImportError("Install quantum-portfolio[ml] for plotly support.") from exc
    return px.bar(x=weights.index, y=weights.values)
