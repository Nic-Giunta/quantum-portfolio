from pathlib import Path

from jinja2 import Environment, PackageLoader, select_autoescape


def html_report(result) -> str:
    env = Environment(loader=PackageLoader("quantum_portfolio.reports", "templates"), autoescape=select_autoescape())
    tpl = env.get_template("report.html.j2")
    return tpl.render(result=result, weights=result.to_frame().reset_index().rename(columns={"index":"asset"}).to_dict(orient="records"))
def save_html_report(result, path): Path(path).write_text(html_report(result), encoding="utf-8")
