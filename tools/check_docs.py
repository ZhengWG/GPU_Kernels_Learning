"""Check local document links and generated SVG freshness without dependencies."""

from pathlib import Path
import re
import xml.etree.ElementTree as ET

from render_diagrams import digest

ROOT = Path(__file__).resolve().parents[1]


def main():
    errors = []
    documents = list(ROOT.rglob("*.md"))
    for path in documents:
        text = path.read_text()
        if text.count("```") % 2:
            errors.append(f"{path}: unbalanced code fences")
        if text.count("<details>") != text.count("</details>"):
            errors.append(f"{path}: unbalanced folding tags")
        if "```mermaid" in text:
            errors.append(f"{path}: use a generated SVG instead of embedded Mermaid")
        for target in re.findall(r"\]\(([^)]+)\)", text):
            if "://" in target or target.startswith("#"):
                continue
            if not (path.parent / target.split("#")[0]).exists():
                errors.append(f"{path}: missing {target}")
    sources = list(ROOT.rglob("*.mmd"))
    for source in sources:
        svg = source.with_suffix(".svg")
        if not svg.exists():
            errors.append(f"{source}: missing SVG")
            continue
        text = svg.read_text()
        if f"source-sha256: {digest(source)}" not in text:
            errors.append(f"{svg}: source or renderer config changed; regenerate")
        try:
            root = ET.fromstring(text)
            if root.tag != "{http://www.w3.org/2000/svg}svg":
                errors.append(f"{svg}: wrong root element")
            for element in root.iter():
                if element.tag.rsplit("}", 1)[-1] in {"script", "foreignObject"}:
                    errors.append(f"{svg}: unsupported element {element.tag}")
        except ET.ParseError as error:
            errors.append(f"{svg}: {error}")
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"Checked {len(documents)} documents and {len(sources)} static diagrams.")


if __name__ == "__main__":
    main()
