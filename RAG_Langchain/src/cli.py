import argparse
import logging
from pathlib import Path
from typing import Sequence

from .bootstrap import build_indexing_service, build_rag_service
from .config import Settings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="multimodal-rag",
        description="Parse, index, and query multimodal financial documents.",
    )
    parser.add_argument("--env-file", type=Path, default=None)
    parser.add_argument("--verbose", action="store_true")

    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="Parse and index documents")
    index_parser.add_argument("--pdf", nargs="*", type=Path, default=[])
    index_parser.add_argument("--markdown", nargs="*", type=Path, default=[])

    ask_parser = subparsers.add_parser("ask", help="Ask a question")
    ask_parser.add_argument("question", type=str)
    ask_parser.add_argument("--show-context", action="store_true")

    return parser


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s | %(name)s | %(message)s",
    )


def run_index(settings: Settings, pdfs: Sequence[Path], markdowns: Sequence[Path]) -> None:
    if not pdfs and not markdowns:
        pdfs = sorted(settings.input_dir.glob("*.pdf"))
        markdowns = sorted(settings.processed_dir.glob("*.md"))

    result = build_indexing_service(settings).index(
        pdf_paths=pdfs,
        markdown_paths=markdowns,
    )

    print(f"Indexed documents: {result.document_count}")
    print(f"Created chunks: {result.chunk_count}")
    print("Markdown files:")
    for path in result.markdown_files:
        print(f"- {path}")


def run_ask(settings: Settings, question: str, show_context: bool) -> None:
    result = build_rag_service(settings).ask(question)

    print("\nAnswer:\n")
    print(result.answer)

    if result.image_paths:
        print("\nRetrieved images:")
        for path in result.image_paths:
            print(f"- {path}")

    if show_context:
        print("\nRetrieved chunks:")
        for index, document in enumerate(result.retrieved_documents, start=1):
            print(f"\n--- Chunk {index} ---")
            print(document.page_content)


def main() -> None:
    args = build_parser().parse_args()
    configure_logging(args.verbose)
    settings = Settings.from_env(args.env_file)

    if args.command == "index":
        run_index(settings, args.pdf, args.markdown)
    elif args.command == "ask":
        run_ask(settings, args.question, args.show_context)


if __name__ == "__main__":
    main()
