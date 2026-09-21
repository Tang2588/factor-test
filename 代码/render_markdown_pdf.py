# -*- coding: utf-8 -*-
"""Render a local Markdown document to PDF with Chrome."""
from __future__ import annotations

import argparse
import subprocess
import tempfile
from html import escape
from pathlib import Path

from markdown_it import MarkdownIt


CHROME_CANDIDATES = (
    Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
)


CSS = r"""
@page { size: A4; margin: 16mm 14mm 17mm; }
* { box-sizing: border-box; }
html { print-color-adjust: exact; -webkit-print-color-adjust: exact; }
body {
  margin: 0 auto;
  color: #202428;
  background: #fff;
  font-family: "Microsoft YaHei", "SimSun", sans-serif;
  font-size: 10.5pt;
  line-height: 1.68;
}
h1, h2, h3 { color: #183f5b; break-after: avoid-page; page-break-after: avoid; }
h1 {
  margin: 0 0 18pt;
  padding-bottom: 9pt;
  border-bottom: 2px solid #2f6b9a;
  font-size: 23pt;
  line-height: 1.28;
  text-align: center;
}
h2 {
  margin: 20pt 0 8pt;
  padding-left: 7pt;
  border-left: 4px solid #2f6b9a;
  font-size: 16pt;
  line-height: 1.35;
}
h3 { margin: 15pt 0 6pt; font-size: 12.5pt; }
p { margin: 5pt 0; orphans: 3; widows: 3; }
ul, ol { margin: 5pt 0 8pt; padding-left: 22pt; }
li { margin: 2pt 0; }
blockquote {
  margin: 10pt 0;
  padding: 8pt 11pt;
  border-left: 4px solid #6f94ad;
  background: #eef4f7;
  color: #34434d;
}
code {
  padding: 1px 3px;
  border-radius: 2px;
  background: #f1f3f5;
  font-family: Consolas, "Microsoft YaHei", monospace;
  font-size: 0.92em;
}
table {
  width: 100%;
  margin: 9pt 0 13pt;
  border-collapse: collapse;
  font-size: 7.6pt;
  line-height: 1.35;
}
thead { display: table-header-group; }
tr { break-inside: avoid-page; page-break-inside: avoid; }
th, td {
  padding: 4px 5px;
  border: 1px solid #c7cfd5;
  vertical-align: middle;
}
th { background: #e8f0f5; color: #183f5b; font-weight: 700; }
tbody tr:nth-child(even) { background: #f7f9fa; }
img {
  display: block;
  max-width: 100%;
  max-height: 225mm;
  height: auto;
  margin: 10pt auto 13pt;
  break-inside: avoid-page;
  page-break-inside: avoid;
}
strong { color: #172f40; }
hr { border: 0; border-top: 1px solid #cbd3d8; }
"""


def find_browser() -> Path:
    for candidate in CHROME_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Chrome or Edge was not found")


def render(source: Path, output: Path) -> None:
    source = source.resolve()
    output = output.resolve()
    markdown = source.read_text(encoding="utf-8")
    parser = MarkdownIt("commonmark", {"html": True}).enable("table")
    body = parser.render(markdown)
    title = source.stem
    base_url = source.parent.as_uri().rstrip("/") + "/"
    document = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<base href="{escape(base_url, quote=True)}">
<title>{escape(title)}</title>
<style>{CSS}</style>
</head>
<body>{body}</body>
</html>
"""

    output.parent.mkdir(parents=True, exist_ok=True)
    html_path = source.parent / f".{source.stem}_print.html"
    html_path.write_text(document, encoding="utf-8")
    browser = find_browser()
    try:
        with tempfile.TemporaryDirectory(prefix="md-pdf-") as profile:
            subprocess.run(
                [
                    str(browser),
                    "--headless=new",
                    "--disable-gpu",
                    "--allow-file-access-from-files",
                    "--no-pdf-header-footer",
                    f"--user-data-dir={profile}",
                    f"--print-to-pdf={output}",
                    html_path.as_uri(),
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
    finally:
        html_path.unlink(missing_ok=True)

    if not output.exists() or output.stat().st_size == 0:
        raise RuntimeError(f"PDF was not created: {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    render(args.source, args.output)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
