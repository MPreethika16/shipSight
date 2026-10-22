"""Converts SHIP_SIGHT_TECHNICAL_REPORT.md into a high-quality PDF using Microsoft Edge headless.
Verifies page count using pypdf.
"""

import os
import re
import subprocess

try:
    import pypdf
except ImportError:
    pypdf = None


def markdown_to_html(md_text: str) -> str:
    """Simple parser converting standard markdown formatting to clean semantic HTML."""
    lines = md_text.splitlines()
    html_lines = []
    in_table = False
    in_code = False
    table_headers = []

    for line in lines:
        # Code block
        if line.strip().startswith("```"):
            if in_code:
                html_lines.append("</pre>")
                in_code = False
            else:
                html_lines.append("<pre>")
                in_code = True
            continue
        if in_code:
            html_lines.append(line)
            continue

        # Horizontal rule
        if line.strip() == "---":
            if in_table:
                html_lines.append("</tbody></table>")
                in_table = False
            html_lines.append("<hr/>")
            continue

        # Headings
        if line.startswith("# "):
            html_lines.append(f"<h1>{line[2:].strip()}</h1>")
            continue
        if line.startswith("## "):
            html_lines.append(f"<h2>{line[3:].strip()}</h2>")
            continue
        if line.startswith("### "):
            html_lines.append(f"<h3>{line[4:].strip()}</h3>")
            continue

        # Table processing
        if line.strip().startswith("|") and line.strip().endswith("|"):
            cells = [c.strip() for c in line.strip()[1:-1].split("|")]
            # Check separator
            if all(re.match(r"^:?-+:?$", c) for c in cells):
                continue
            if not in_table:
                in_table = True
                html_lines.append("<table><thead><tr>")
                for c in cells:
                    html_lines.append(f"<th>{c}</th>")
                html_lines.append("</tr></thead><tbody>")
            else:
                html_lines.append("<tr>")
                for c in cells:
                    html_lines.append(f"<td>{c}</td>")
                html_lines.append("</tr>")
            continue
        else:
            if in_table:
                html_lines.append("</tbody></table>")
                in_table = False

        # Unordered list
        if line.strip().startswith("- ") or line.strip().startswith("* "):
            content = line.strip()[2:].strip()
            html_lines.append(f"<li>{content}</li>")
            continue

        # Numbered list
        num_match = re.match(r"^(\d+)\.\s+(.*)$", line.strip())
        if num_match:
            html_lines.append(f"<li><strong>{num_match.group(1)}.</strong> {num_match.group(2)}</li>")
            continue

        # Paragraph
        if line.strip():
            html_lines.append(f"<p>{line.strip()}</p>")

    if in_table:
        html_lines.append("</tbody></table>")

    body_html = "\n".join(html_lines)

    # Inline replacements: bold, italic, code
    body_html = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", body_html)
    body_html = re.sub(r"\*(.*?)\*", r"<em>\1</em>", body_html)
    body_html = re.sub(r"`(.*?)`", r"<code>\1</code>", body_html)

    # Clean display math
    body_html = re.sub(r"\$\$(.*?)\$\$", r"<div class='math'>\1</div>", body_html)
    body_html = re.sub(r"\$(.*?)\$", r"<span class='math'>\1</span>", body_html)

    full_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  @page {{
    size: A4;
    margin: 8.5mm 8.5mm 8.5mm 8.5mm;
  }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    font-size: 7.2pt;
    line-height: 1.18;
    color: #1a1a1a;
  }}
  h1 {{
    font-size: 13pt;
    margin: 0 0 3px 0;
    color: #0f172a;
    border-bottom: 2px solid #0284c7;
    padding-bottom: 2px;
  }}
  h2 {{
    font-size: 9.8pt;
    margin: 5px 0 2px 0;
    color: #1e293b;
    border-bottom: 1px solid #cbd5e1;
    padding-bottom: 1px;
    page-break-after: avoid;
  }}
  h3 {{
    font-size: 8.8pt;
    margin: 4px 0 1px 0;
    color: #334155;
    page-break-after: avoid;
  }}
  p {{
    margin: 2px 0;
    text-align: justify;
  }}
  li {{
    margin: 1.5px 0;
  }}
  hr {{
    border: 0;
    border-top: 1px solid #e2e8f0;
    margin: 4px 0;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    margin: 4px 0;
    font-size: 7.5pt;
    page-break-inside: avoid;
  }}
  th, td {{
    border: 1px solid #cbd5e1;
    padding: 2px 4px;
    text-align: left;
  }}
  th {{
    background-color: #f1f5f9;
    font-weight: 600;
    color: #0f172a;
  }}
  pre {{
    background-color: #f8fafc;
    border: 1px solid #e2e8f0;
    padding: 3px 5px;
    font-size: 7.2pt;
    font-family: Consolas, monospace;
    border-radius: 2px;
    margin: 3px 0;
    page-break-inside: avoid;
    white-space: pre-wrap;
  }}
  code {{
    font-family: Consolas, monospace;
    font-size: 7.5pt;
    background-color: #f1f5f9;
    padding: 1px 2px;
    border-radius: 2px;
  }}
  .math {{
    font-family: "Cambria Math", "Times New Roman", serif;
    font-style: italic;
  }}
</style>
</head>
<body>
{body_html}
</body>
</html>
"""
    return full_html


def generate_pdf(
    md_path: str = "SHIP_SIGHT_TECHNICAL_REPORT.md",
    pdf_path: str = "SHIP_SIGHT_TECHNICAL_REPORT.pdf",
) -> int:
    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    html_content = markdown_to_html(md_text)
    temp_html = "temp_report.html"
    with open(temp_html, "w", encoding="utf-8") as f:
        f.write(html_content)

    edge_path = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
    abs_html = os.path.abspath(temp_html)
    abs_pdf = os.path.abspath(pdf_path)
    file_url = "file:///" + abs_html.replace("\\", "/")

    cmd = [
        edge_path,
        "--headless",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={abs_pdf}",
        file_url,
    ]
    subprocess.run(cmd, capture_output=True, check=True)

    if os.path.exists(temp_html):
        os.remove(temp_html)

    if pypdf is not None:
        reader = pypdf.PdfReader(abs_pdf)
        page_count = len(reader.pages)
    else:
        # Standard-library fallback: count /Type /Page objects in PDF stream
        with open(abs_pdf, "rb") as f:
            pdf_bytes = f.read()
        page_count = len(re.findall(rb"/Type\s*/Page\b", pdf_bytes))

    print(f"Generated PDF: {abs_pdf}")
    print(f"Verified Page Count: {page_count} pages")
    return page_count


if __name__ == "__main__":
    generate_pdf()
