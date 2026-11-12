# /// script
# requires-python = ">=3.12"
# dependencies = ["docling==2.147.0"]
# ///
"""Turn data/downloads into Markdown under data/markdown.

Year folders stay the same. HTML filings become .md files, and manifest.json
becomes manifest.md.
"""

from __future__ import annotations

import json
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, HTMLFormatOption

DOWNLOADS_DIR = Path(__file__).resolve().parent / "downloads"
OUTPUT_DIR = Path(__file__).resolve().parent / "markdown"


def filing_markdown(converter: DocumentConverter, source: Path) -> str:
    result = converter.convert(source)
    if result.status.name != "SUCCESS":
        raise RuntimeError(f"{source.name} converted with status {result.status.name}")
    return result.document.export_to_markdown()


def manifest_markdown(payload: dict) -> str:
    lines = [
        "# Download manifest",
        "",
        f"- Source: {payload.get('source', '')}",
        f"- Generated: {payload.get('generated_at_utc', '')}",
        f"- Form: {payload.get('form', '')}",
        f"- Filings: {payload.get('downloaded_count', '')}",
        "",
        "| Ticker | Filing date | Report date | Accession | Markdown file |",
        "| --- | --- | --- | --- | --- |",
    ]
    for filing in payload.get("filings", []):
        local = Path(str(filing.get("local_path", ""))).with_suffix(".md").as_posix()
        lines.append(
            "| {ticker} | {filing_date} | {report_date} | {accession_number} | {local} |".format(
                ticker=filing.get("ticker", ""),
                filing_date=filing.get("filing_date", ""),
                report_date=filing.get("report_date", ""),
                accession_number=filing.get("accession_number", ""),
                local=local,
            )
        )
    lines.append("")
    return "\n".join(lines)


def convert_downloads() -> int:
    if not DOWNLOADS_DIR.is_dir():
        raise FileNotFoundError(f"Missing downloads folder: {DOWNLOADS_DIR}")

    converter = DocumentConverter(
        allowed_formats=[InputFormat.HTML],
        format_options={InputFormat.HTML: HTMLFormatOption()},
    )
    written = 0
    for source in sorted(DOWNLOADS_DIR.rglob("*")):
        if not source.is_file():
            continue
        relative = source.relative_to(DOWNLOADS_DIR)
        if source.suffix.lower() in {".htm", ".html"}:
            destination = OUTPUT_DIR / relative.with_suffix(".md")
            text = filing_markdown(converter, source)
        elif source.suffix.lower() == ".json":
            destination = OUTPUT_DIR / relative.with_suffix(".md")
            text = manifest_markdown(json.loads(source.read_text(encoding="utf-8")))
        else:
            destination = OUTPUT_DIR / relative.with_suffix(".md")
            text = source.read_text(encoding="utf-8")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text, encoding="utf-8", newline="\n")
        written += 1
        print(f"Wrote {destination.relative_to(OUTPUT_DIR)}", flush=True)
    return written


if __name__ == "__main__":
    count = convert_downloads()
    print(f"Wrote {count} markdown file(s) to {OUTPUT_DIR}")
