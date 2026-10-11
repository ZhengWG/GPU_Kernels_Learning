"""Render Mermaid 10.9.1 sources with a local browser bundle and Chromium.

Readers need neither browser tooling nor Node packages to view checked-in SVGs.
"""

import argparse
import hashlib
from html import escape, unescape
import json
from pathlib import Path
import re
import subprocess
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "tools" / "mermaid-config.json"
VERSION = "10.9.1"
BUNDLE_SHA = "61b335a46df05a7ce1c98378f60e5f3e77a7fb608a1056997e8a649304a936d6"


def digest(source):
    return hashlib.sha256(source.read_bytes() + CONFIG.read_bytes() + VERSION.encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--browser", required=True, help="Chrome/Chromium executable")
    parser.add_argument("--mermaid-js", required=True, type=Path,
                        help="Local mermaid@10.9.1/dist/mermaid.min.js")
    args = parser.parse_args()
    if hashlib.sha256(args.mermaid_js.read_bytes()).hexdigest() != BUNDLE_SHA:
        raise SystemExit("Unexpected Mermaid bundle: use the documented version and checksum")
    sources = [source for source in sorted(ROOT.rglob("*.mmd"))
               if not source.with_suffix(".svg").exists()
               or f"source-sha256: {digest(source)}" not in source.with_suffix(".svg").read_text()]
    if not sources:
        print("All diagrams are current.")
        return
    config = json.loads(CONFIG.read_text())
    config["startOnLoad"] = False
    with tempfile.TemporaryDirectory(prefix="cuda-diagrams-") as tmp:
        folder = Path(tmp)
        page = folder / "render.html"
        source_hashes = [digest(source) for source in sources]
        scripts = json.dumps([source.read_text() for source in sources], ensure_ascii=False)
        # Avoid treating diagram labels as executable HTML.
        scripts = scripts.replace("<", "\\u003c")
        page.write_text('''<!doctype html><meta charset="utf-8"><pre id="result"></pre>
<script src="''' + escape(args.mermaid_js.resolve().as_uri(), quote=True) + '''"></script>
<script>
(async () => {
  try {
    mermaid.initialize(''' + json.dumps(config) + ''');
    const sources = ''' + scripts + ''';
    const result = [];
    for (let i = 0; i < sources.length; i++) {
      const rendered = await mermaid.render("diagram" + i, sources[i]);
      result.push(rendered.svg);
    }
    document.querySelector("#result").textContent = JSON.stringify({svgs:result});
  } catch (error) {
    document.querySelector("#result").textContent = JSON.stringify({error:String(error)});
  }
})();
</script>''')
        # Capture to files: Chrome helper processes can keep pipe handles open
        # after the browser has finished, notably on macOS.
        output = folder / "output.html"
        errors = folder / "browser.log"
        with output.open("w") as stdout, errors.open("w") as stderr:
            process = subprocess.Popen([
                args.browser, "--headless=new", "--disable-gpu", "--no-sandbox",
                "--disable-background-networking", "--disable-extensions", "--no-first-run",
                "--no-default-browser-check", "--allow-file-access-from-files",
                "--user-data-dir=" + str(folder / "profile"),
                "--virtual-time-budget=30000", "--dump-dom", page.as_uri(),
            ], stdout=stdout, stderr=stderr)
            match = None
            try:
                deadline = time.monotonic() + 90
                while time.monotonic() < deadline:
                    match = re.search(r'<pre id="result">(.*?)</pre>', output.read_text(), re.S)
                    if match and match.group(1):
                        break
                    if process.poll() is not None:
                        break
                    time.sleep(0.2)
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
        if not match or not match.group(1):
            raise RuntimeError("Browser returned no diagram output: " + errors.read_text()[-1000:])
        result = json.loads(unescape(match.group(1)))
        if "error" in result:
            raise RuntimeError(result["error"])
        svgs = result["svgs"]
        if len(svgs) != len(sources):
            raise RuntimeError("Some diagrams were not rendered")
        for source, svg, source_hash in zip(sources, svgs, source_hashes):
            if "<foreignObject" in svg or "<script" in svg:
                raise ValueError(f"Unsupported SVG content in {source}")
            target = source.with_suffix(".svg")
            target.write_text(svg + f"\n<!-- source-sha256: {source_hash} -->\n")
            print(target.relative_to(ROOT))


if __name__ == "__main__":
    main()
