""" command-line interface.

Usage
-----
    python -m src.cli "your natural-language command" [--show] [--model NAME]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from .agent import LLMAgent, RunContext, TOOL_SPECS, build_system_prompt, build_tool_registry
from .utils import load_config, make_run_dir, show_all


console = Console()


def _format_args(args: dict, max_len: int = 140) -> str:
    s = json.dumps(args, default=str, ensure_ascii=False)
    return s if len(s) <= max_len else s[: max_len - 1] + "..."


def _format_result(result: dict, max_keys: int = 6) -> str:
    if not isinstance(result, dict):
        return str(result)
    if "error" in result:
        return f"error: {result['error']}"
    parts = []
    for k, v in list(result.items())[:max_keys]:
        if isinstance(v, (list, tuple)):
            v = f"[{len(v)} items]" if len(v) > 4 else json.dumps(list(v), default=str)
        elif isinstance(v, dict):
            v = "{...}"
        parts.append(f"{k}={v}")
    if len(result) > max_keys:
        parts.append(f"(+{len(result) - max_keys} more)")
    return ", ".join(parts)


class StreamingDisplay:
    def __init__(self, console: Console):
        self.console = console
        self.body = Text(no_wrap=False)
        self.panel = Panel(self.body, title="Agent thinking...", border_style="green")
        # transient=False keeps the panel rendered after the run finishes so
        # the user can scroll back through the agent's reasoning.
        self.live = Live(
            self.panel,
            console=console,
            refresh_per_second=24,
            transient=False,
            vertical_overflow="visible",
        )
        self._narrated_this_turn = False

    def __enter__(self) -> "StreamingDisplay":
        self.live.__enter__()
        return self

    def finish(self) -> None:
        """Swap the panel title from the in-progress label to a final one."""
        self.panel.title = "Agent reasoning"
        self.live.refresh()

    def __exit__(self, *exc) -> None:
        self.live.__exit__(*exc)

    def on_event(self, kind: str, payload: dict) -> None:
        if kind == "step":
            i = payload.get("index", 0)
            if i > 0:
                self.body.append("\n")
            self.body.append(f"-- step {i + 1} --\n", style="dim")
            self._narrated_this_turn = False
        elif kind == "assistant_start":
            self.body.append("assistant ", style="bold cyan")
        elif kind == "token":
            self.body.append(payload.get("text", ""))
            self._narrated_this_turn = True
        elif kind == "assistant_end":
            if self._narrated_this_turn:
                self.body.append("\n")
            elif payload.get("has_tool_calls"):
                self.body.append("(emitted only tool calls)\n", style="dim")
            else:
                self.body.append("(empty turn -- conversation ended)\n", style="dim")
        elif kind == "tool_call":
            name = payload.get("name", "?")
            args = _format_args(payload.get("arguments", {}))
            self.body.append(f"-> {name}", style="yellow")
            self.body.append(f"({args})\n")
        elif kind == "tool_result":
            name = payload.get("name", "?")
            result = payload.get("result", {})
            err = isinstance(result, dict) and "error" in result
            self.body.append("  <- ", style="red" if err else "green")
            self.body.append(f"{name}: ", style="dim")
            self.body.append(f"{_format_result(result)}\n")
        elif kind == "nudge":
            reason = payload.get("reason", "?")
            self.body.append(f"[nudge: {reason}] retrying...\n", style="magenta")
        self.live.refresh()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="cognitive-array",
        description="LLM-driven radar agent (ECE 693 term project).",
    )
    parser.add_argument("query", type=str, help="Natural-language scenario command.")
    parser.add_argument("--show", action="store_true", help="Display plots interactively at the end.")
    parser.add_argument("--model", type=str, default=None, help="Override the Ollama model name.")
    parser.add_argument("--host", type=str, default=None, help="Override the Ollama host URL.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config.")
    parser.add_argument("--max-steps", type=int, default=8, help="Max agent tool-calling iterations.")
    parser.add_argument(
        "--ping",
        action="store_true",
        help="Check that the Ollama host and model are reachable, then exit.",
    )
    return parser.parse_args(argv)


def _resolve(cfg: dict, model: str | None, host: str | None) -> tuple[str, str, dict]:
    model_cfg = cfg.get("model", {})
    return (
        model or model_cfg.get("name", "gemma4:latest"),
        host or model_cfg.get("host", "http://localhost:11434"),
        model_cfg.get("options", {}),
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    model, host, options = _resolve(cfg, args.model, args.host)

    if args.ping:
        agent = LLMAgent(model=model, host=host, options=options)
        ok = agent.ping()
        console.print(f"[green]OK[/green]: {model} at {host}" if ok
                      else f"[red]FAIL[/red]: cannot reach {model} at {host}")
        return 0 if ok else 1

    run_dir = make_run_dir(cfg.get("output", {}).get("root", "output"))
    console.print(Panel.fit(f"[bold]Run directory[/bold]: {run_dir}", border_style="cyan"))

    arr = cfg["array"]
    out = cfg.get("output", {})
    ctx = RunContext(
        run_dir=run_dir,
        cfg=cfg,
        N=int(arr["n_elements"]),
        d=float(arr["spacing_wavelengths"]),
        angle_grid=np.arange(-90.0, 90.0 + 1e-9, float(out.get("angle_grid_deg", 0.1))),
        dpi=int(out.get("dpi", 150)),
    )
    tool_registry = build_tool_registry(ctx)

    agent = LLMAgent(
        model=model,
        host=host,
        options=options,
        system_prompt=build_system_prompt(cfg),
        tool_specs=TOOL_SPECS,
        tool_registry=tool_registry,
    )

    console.print(f"[bold]user[/bold] {args.query}")
    try:
        with StreamingDisplay(console) as display:
            transcript = agent.run(
                args.query,
                max_steps=args.max_steps,
                on_event=display.on_event,
            )
            display.finish()
    except Exception as exc:
        console.print(f"\n[red]Agent failed:[/red] {exc}")
        return 2

    # Persist the chain-of-thought transcript -- required deliverable (30% of grade).
    (run_dir / "logs" / "transcript.json").write_text(json.dumps(transcript, indent=2))
    (run_dir / "logs" / "query.txt").write_text(args.query + "\n")

    plots = sorted((run_dir / "plots").glob("*.png"))
    if plots:
        console.print(f"[dim]Generated {len(plots)} plot(s):[/dim] "
                      + ", ".join(p.name for p in plots))

    if args.show:
        show_all()

    return 0


if __name__ == "__main__":
    sys.exit(main())
