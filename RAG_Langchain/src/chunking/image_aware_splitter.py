import re
from pathlib import Path
from typing import List, Sequence

from langchain_core.documents import Document
from langchain_text_splitters.markdown import MarkdownHeaderTextSplitter


class OneImagePerChunkSplitter:
    """
    Splits Markdown so each image and its preceding description
    are stored in one chunk.

    Pattern:
    Description text → Figure caption → Image
    """

    def __init__(
        self,
        headers_to_split_on: List[tuple] | None = None,
        strip_headers: bool = False,
    ) -> None:
        self.headers_to_split_on = (
            headers_to_split_on
            or [("##", "Header 2")]
        )

        self.strip_headers = strip_headers

        self.base_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.headers_to_split_on,
            strip_headers=strip_headers,
        )

    def _split_by_image_blocks(
        self,
        text: str,
        metadata: dict,
    ) -> List[Document]:
        """
        Split text so each image chunk contains:

        - The description before the image.
        - The Figure caption.
        - The Markdown image tag.
        """
        chunks: List[Document] = []

        image_pattern = r"!\[Image\]\(([^)]+)\)"
        images = list(re.finditer(image_pattern, text))

        if not images:
            return [
                Document(
                    page_content=text.strip(),
                    metadata={
                        **metadata,
                        "chunk_type": "text_only",
                        "has_image": False,
                    },
                )
            ]

        last_position = 0

        for image_index, image_match in enumerate(
            images,
            start=1,
        ):
            image_start = image_match.start()
            image_end = image_match.end()

            text_before_image = text[
                last_position:image_start
            ].strip()

            image_tag = image_match.group(0)
            original_image_path = image_match.group(1).strip()

            resolved_image_path = self._resolve_image_path(
                image_path=original_image_path,
                metadata=metadata,
            )

            lines_before = text_before_image.splitlines()

            figure_caption = ""
            description_text = text_before_image

            if (
                lines_before
                and lines_before[-1]
                .strip()
                .lower()
                .startswith("figure")
            ):
                figure_caption = lines_before[-1].strip()

                description_text = "\n".join(
                    lines_before[:-1]
                ).strip()

            content_parts = [
                part
                for part in (
                    description_text,
                    figure_caption,
                    image_tag,
                )
                if part
            ]

            chunk_content = "\n\n".join(content_parts)

            chunks.append(
                Document(
                    page_content=chunk_content,
                    metadata={
                        **metadata,
                        "chunk_type": "image_with_description",
                        "has_image": True,
                        "image_index": image_index,
                        "figure_caption": figure_caption,

                        # Absolute or resolved image path.
                        "image_path": str(resolved_image_path),

                        # Original path written in Markdown.
                        "image_path_original": original_image_path,
                    },
                )
            )

            last_position = image_end

        remaining_text = text[last_position:].strip()

        if remaining_text:
            chunks.append(
                Document(
                    page_content=remaining_text,
                    metadata={
                        **metadata,
                        "chunk_type": "text_after_images",
                        "has_image": False,
                    },
                )
            )

        return chunks

    @staticmethod
    def _extract_header_text(
        text: str,
    ) -> tuple[str, str]:
        """
        Extract the Markdown header and its content.

        Returns:
            tuple containing:
            - header
            - content
        """
        lines = text.split("\n", 1)

        if lines and lines[0].startswith("##"):
            header = lines[0].strip()

            content = (
                lines[1].strip()
                if len(lines) > 1
                else ""
            )

            return header, content

        return "", text
    
    def split_documents(self, documents: Sequence[Document]) -> list[Document]: 
        chunks: list[Document] = [] 
        for document in documents: 
            chunks.extend(self.split_text(document.page_content)) 
        return chunks

    def split_text(
        self,
        text: str,
    ) -> List[Document]:
        """
        Split the Markdown by headers first.

        Sections containing images are then split so each image
        is stored with its preceding description.
        """
        header_chunks = self.base_splitter.split_text(text)

        final_chunks: List[Document] = []

        for chunk in header_chunks:
            section_text = chunk.page_content
            section_metadata = chunk.metadata

            if "![Image]" in section_text:
                header, content = self._extract_header_text(
                    section_text
                )

                image_chunks = self._split_by_image_blocks(
                    text=content,
                    metadata=section_metadata,
                )

                if header and image_chunks:
                    first_chunk = image_chunks[0]

                    image_chunks[0] = Document(
                        page_content=(
                            f"{header}\n\n"
                            f"{first_chunk.page_content}"
                        ),
                        metadata=first_chunk.metadata,
                    )

                final_chunks.extend(image_chunks)

            else:
                final_chunks.append(
                    Document(
                        page_content=section_text,
                        metadata={
                            **section_metadata,
                            "chunk_type": "text_only",
                            "has_image": False,
                        },
                    )
                )

        return final_chunks

    @staticmethod
    def _resolve_image_path(
        image_path: str,
        metadata: dict,
    ) -> Path:
        """
        Resolve an image path.

        If the image path is already absolute, return it directly.

        If it is relative, resolve it relative to the Markdown
        source file stored in metadata["source"].
        """
        path = Path(image_path)

        if path.is_absolute():
            return path

        source = metadata.get("source")

        if source:
            return (
                Path(source).parent / path
            ).resolve()

        return path.resolve()