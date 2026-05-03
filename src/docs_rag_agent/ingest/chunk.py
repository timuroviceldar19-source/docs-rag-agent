from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    source: str      # relative path string, e.g. "docs/en/docs/tutorial/index.md"
    heading: str     # the ## heading this chunk falls under, or "" if none
    index: int       # 0-based position of this chunk within the source file


def chunk_markdown(
    text: str,
    source: str,
    max_chars: int = 1500,
    overlap_chars: int = 150,
) -> list[Chunk]:
    """
    Split markdown text into chunks preserving heading context.

    Strategy:
    1. Split on lines that start with '## ' (level-2 headings).
       Each section = heading line + content below it until next ## heading.
    2. If a section exceeds max_chars, split further on double-newline paragraphs.
    3. If a paragraph still exceeds max_chars, hard-split at max_chars with
       overlap_chars of overlap.
    4. Return Chunk objects with source, heading, and 0-based index.
    """
    sections: list[tuple[str, str]] = []
    current_heading = ""
    current_lines: list[str] = []

    for line in text.splitlines():
        if line.startswith("## "):
            if current_lines:
                sections.append((current_heading, "\n".join(current_lines)))
            current_heading = line.removeprefix("## ").strip()
            current_lines = [line]
        else:
            current_lines.append(line)
    
    if current_lines:
        sections.append((current_heading, "\n".join(current_lines)))

    chunks: list[Chunk] = []
    chunk_index = 0

    for heading, section_text in sections:
        # Split section into paragraphs
        paragraphs = section_text.split("\n\n")
        
        for para in paragraphs:
            if not para.strip():
                continue
            
            if len(para) <= max_chars:
                if len(para.strip()) >= 20:
                    chunks.append(Chunk(
                        text=para,
                        source=source,
                        heading=heading,
                        index=chunk_index
                    ))
                    chunk_index += 1
            else:
                # Hard split
                start = 0
                while start < len(para):
                    end = min(start + max_chars, len(para))
                    piece = para[start:end]
                    
                    if len(piece.strip()) >= 20:
                        chunks.append(Chunk(
                            text=piece,
                            source=source,
                            heading=heading,
                            index=chunk_index
                        ))
                        chunk_index += 1
                    
                    start = end - overlap_chars
                    if start >= len(para) or end >= len(para):
                        break

    return chunks
