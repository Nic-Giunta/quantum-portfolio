from pathlib import Path


def _markdown_weights_table(result) -> str:
    frame = result.to_frame().reset_index().rename(columns={"index": "asset"})
    columns = list(frame.columns)
    header = "| " + " | ".join(str(column) for column in columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = []
    for row in frame.itertuples(index=False):
        rows.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join([header, separator, *rows])


def markdown_report(result) -> str:
    return f"""# QuantumPortfolio Optimization Report

{result.summary()}

## Weights

{_markdown_weights_table(result)}

## Disclaimer

Research software only. Not financial, tax, legal, or investment advice.
"""
def save_markdown_report(result, path): Path(path).write_text(markdown_report(result), encoding="utf-8")
