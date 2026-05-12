"""
PaperPilot CLI — refine LaTeX paragraphs with a five-agent pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

import pipeline

app = typer.Typer(add_completion=False, help="PaperPilot: multi-agent LaTeX paragraph refinement.")
console = Console()


def _print_report(report) -> None:
    verdict_color = "green" if report.approved() else "yellow"
    verdict_label = Text(f"  {report.verdict.upper()}  ", style=f"bold white on {verdict_color}")

    lines: list[str] = [report.summary, ""]
    if report.issues:
        for iss in report.issues:
            lines.append(f"[bold][{iss.category.upper()}][/bold] \"{iss.location}\"")
            lines.append(f"  Problem    : {iss.description}")
            lines.append(f"  Suggestion : {iss.suggestion}")
            lines.append("")
    else:
        lines.append("[green]No issues found.[/green]")

    console.print()
    console.print(verdict_label, " ", end="")
    console.print(
        Panel("\n".join(lines).strip(), title="[bold]Reviewer Report[/bold]", border_style=verdict_color)
    )


def _print_paths(result) -> None:
    console.print()
    console.print(f"[bold green]✓[/bold green] [bold]output.tex[/bold]  {result.tex_path}")
    console.print(f"[bold green]✓[/bold green] [bold]output.bib[/bold]  {result.bib_path}")


@app.command()
def refine(
    input: Path        = typer.Option(...,  "--input",      help="Input .tex file (single paragraph or full section)."),
    output_dir: Path   = typer.Option(...,  "--output-dir", help="Directory for output.tex and output.bib."),
    section: str       = typer.Option(...,  "--section",    help="Section type: intro | method | experiment | conclusion | abstract."),
    topic: str         = typer.Option(...,  "--topic",      help="Paper topic, e.g. 'GFlowNets for discrete structure search'."),
    bib: Optional[Path] = typer.Option(None, "--bib",       help="Existing .bib to append citations to."),
    multi: bool        = typer.Option(False, "--multi",     help="Multi-paragraph mode: refine each paragraph independently."),
    verbose: bool      = typer.Option(False, "--verbose",   help="Print per-agent progress while running."),
    json_out: bool     = typer.Option(False, "--json",      help="Dump full result dict to stdout as JSON."),
) -> None:
    """Refine a LaTeX paragraph or section through five sequential agents."""

    if not input.exists():
        console.print(f"[red]Error:[/red] input file not found: {input}")
        raise typer.Exit(1)

    if bib and not bib.exists():
        console.print(f"[red]Error:[/red] bib file not found: {bib}")
        raise typer.Exit(1)

    try:
        if verbose:
            result = pipeline.run(
                input_path=input,
                output_dir=output_dir,
                section_type=section,
                topic=topic,
                bib_path=bib,
                multi=multi,
                verbose=True,
            )
        else:
            with console.status("[bold cyan]Running pipeline…[/bold cyan]", spinner="dots"):
                result = pipeline.run(
                    input_path=input,
                    output_dir=output_dir,
                    section_type=section,
                    topic=topic,
                    bib_path=bib,
                    multi=multi,
                    verbose=False,
                )
    except Exception as exc:
        console.print(f"[red]Pipeline failed:[/red] {exc}")
        raise typer.Exit(1)

    if json_out:
        print(json.dumps(result.to_dict(), indent=2))
        return

    _print_report(result.report)
    _print_paths(result)


if __name__ == "__main__":
    app()