"""Command-line REPL for asking Small Language Graph questions."""

import argparse
from dataclasses import dataclass
from typing import Callable, Optional

from config import CONFIG
from inference.slg import SmallLanguageGraph


InputFn = Callable[[str], str]
OutputFn = Callable[[str], None]


@dataclass
class InteractiveSLGSession:
    """Small wrapper around SLG that keeps the graph loaded across questions."""

    experiment: str
    show_routing: bool = False
    router_method: Optional[str] = None
    prompt: str = "SLG> "

    def __post_init__(self) -> None:
        self.slg = SmallLanguageGraph(
            experts_location=self.experiment,
            experiment=self.experiment,
            router_method=self.router_method,
        )

    def answer(self, question: str) -> dict:
        return self.slg.ask_question(question)

    def run(
        self,
        input_fn: InputFn = input,
        output_fn: OutputFn = print,
    ) -> None:
        output_fn("Ask SLG a question. Type /exit, /quit, or press Ctrl-D to leave.")
        while True:
            try:
                question = input_fn(self.prompt)
            except EOFError:
                output_fn("")
                break
            except KeyboardInterrupt:
                output_fn("")
                break

            question = question.strip()
            if not question:
                continue
            if question.lower() in {"/exit", "/quit", "exit", "quit"}:
                break

            try:
                result = self.answer(question)
            except Exception as exc:
                output_fn(f"Error: {exc}")
                continue

            answer = result.get("answer") or ""
            output_fn(answer)
            if self.show_routing:
                router = result.get("router_method") or "unknown"
                selected = result.get("selected_expert") or "unknown"
                visited = ", ".join(result.get("visited_experts") or []) or "none"
                output_fn(f"[routing] router={router}; selected={selected}; visited={visited}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Interactively ask questions using the Small Language Graph."
    )
    parser.add_argument(
        "--experiment",
        default=CONFIG["experiment"],
        help="Experiment name containing SLG adapters and index artifacts.",
    )
    parser.add_argument(
        "--question",
        help="Ask one question and exit instead of starting the REPL.",
    )
    parser.add_argument(
        "--show-routing",
        action="store_true",
        help="Print selected and visited experts after each answer.",
    )
    parser.add_argument(
        "--router",
        choices=["cosine", "finetuned"],
        default=None,
        help="SLG router to use. Defaults to routing.method in config.yaml.",
    )
    parser.add_argument(
        "--prompt",
        default="SLG> ",
        help="Prompt text for interactive mode.",
    )
    return parser


def run_interactive_slg(
    experiment: str,
    question: Optional[str] = None,
    show_routing: bool = False,
    router_method: Optional[str] = None,
    prompt: str = "SLG> ",
) -> Optional[dict]:
    session = InteractiveSLGSession(
        experiment=experiment,
        show_routing=show_routing,
        router_method=router_method,
        prompt=prompt,
    )
    if question is None:
        session.run()
        return None

    result = session.answer(question)
    print(result.get("answer") or "")
    if show_routing:
        router = result.get("router_method") or "unknown"
        selected = result.get("selected_expert") or "unknown"
        visited = ", ".join(result.get("visited_experts") or []) or "none"
        print(f"[routing] router={router}; selected={selected}; visited={visited}")
    return result


def main() -> None:
    args = build_parser().parse_args()
    run_interactive_slg(
        experiment=args.experiment,
        question=args.question,
        show_routing=args.show_routing,
        router_method=args.router,
        prompt=args.prompt,
    )
