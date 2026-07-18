import logging
from pathlib import Path
from typing import Iterable

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc.document import ImageRefMode

logger = logging.getLogger(__name__)


class DoclingPdfParser:
    """Convert PDF files to Markdown while exporting referenced images."""

    def __init__(self, output_dir: Path, images_scale: float = 3.0) -> None:
        self._output_dir = output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._converter = self._build_converter(images_scale)

    @staticmethod
    def _build_converter(images_scale: float) -> DocumentConverter:
        options = PdfPipelineOptions()
        options.do_formula_enrichment = True
        options.generate_picture_images = True
        options.images_scale = images_scale

        return DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=options),
            }
        )

    def parse(self, pdf_path: str | Path) -> Path:
        source = Path(pdf_path).expanduser().resolve()
        if not source.exists():
            raise FileNotFoundError(f"PDF file does not exist: {source}")
        if source.suffix.lower() != ".pdf":
            raise ValueError(f"Expected a PDF file, received: {source.name}")

        logger.info("Parsing PDF with Docling: %s", source)
        conversion_result = self._converter.convert(str(source))

        safe_stem = source.stem.replace(" ", "_")
        markdown_path = self._output_dir / f"{safe_stem}-with-image-refs.md"
        conversion_result.document.save_as_markdown(
            markdown_path,
            image_mode=ImageRefMode.REFERENCED,
            include_annotations=True,
        )

        logger.info("Markdown created: %s", markdown_path)
        return markdown_path

    def parse_many(self, pdf_paths: Iterable[str | Path]) -> list[Path]:
        return [self.parse(path) for path in pdf_paths]
