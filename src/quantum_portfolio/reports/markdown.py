from pathlib import Path


def markdown_report(result) -> str:
    return f"""# QuantumPortfolio Optimization Report

{result.summary()}

## Weights

{result.to_frame().to_markdown()}

## Disclaimer

Research software only. Not financial, tax, legal, or investment advice.
"""
def save_markdown_report(result, path): Path(path).write_text(markdown_report(result), encoding="utf-8")
