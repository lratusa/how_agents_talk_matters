"""Build HTML/docx previews of the papers (pandoc, no LaTeX toolchain needed).

The .tex files reference figures as PDF (vector, for LaTeX). For HTML/docx
previews we swap to the 300-dpi PNG twins, run pandoc from paper/, then
post-process the HTML:

  1. pandoc leaves \\cite as empty <span class="citation"> shells and merges
     thebibliography into one giant paragraph. We rebuild the bibliography
     as a numbered <ol> from the .tex bibitems and replace citation spans
     with linked [n] numbers.
  2. pandoc leaks booktabs \\cmidrule(lr){...} arguments as literal
     "(lr)..." table text; we strip them and the empty cells they leave.

Usage:  python scripts/build_previews.py
Requires: pandoc on PATH.
"""
import html as html_mod
import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER = os.path.join(ROOT, "paper")

TARGETS = [
    ("main_en.tex", ["main_en.html"]),
    ("main_zh.tex", ["main_zh.html", "main_zh.docx"]),
]


def swap_figures(tex: str) -> str:
    """Point \\includegraphics at the PNG twin of each figure PDF."""
    return re.sub(r"(\.\./figures/[A-Za-z0-9_]+)\.pdf", r"\1.png", tex)


# ---------------------------------------------------------------------------
# Minimal LaTeX -> HTML for bibliography entry text
# ---------------------------------------------------------------------------

def tex_text_to_html(s: str) -> str:
    s = s.strip()
    s = html_mod.escape(s, quote=False)          # & < >
    s = re.sub(r"\\emph\{([^{}]*)\}", r"<em>\1</em>", s)
    s = re.sub(r"\\texttt\{([^{}]*)\}", r"<code>\1</code>", s)
    s = re.sub(r"\\url\{([^{}]*)\}", r'<a href="\1">\1</a>', s)
    s = re.sub(r"\$([^\$]+)\$", r"\\(\1\\)", s)  # inline math for MathJax
    s = s.replace("\\S", "§").replace("---", "—").replace("--", "–")
    s = s.replace("``", "“").replace("''", "”").replace("`", "‘").replace("'", "’")
    s = s.replace("~", " ").replace("\\&", "&").replace("\\%", "%")
    s = re.sub(r"\\[a-zA-Z]+\{([^{}]*)\}", r"\1", s)  # unwrap remaining 1-arg cmds
    s = s.replace("{", "").replace("}", "")
    return re.sub(r"\s+", " ", s).strip()


def parse_bibitems(tex: str):
    """Return [(key, entry_html)] in .tex order."""
    m = re.search(r"\\begin\{thebibliography\}\{\d+\}(.*?)\\end\{thebibliography\}",
                  tex, re.S)
    if not m:
        return []
    body = m.group(1)
    items = re.findall(r"\\bibitem\{([^}]+)\}(.*?)(?=\\bibitem|\Z)", body, re.S)
    return [(k, tex_text_to_html(t)) for k, t in items]


def fix_html(path: str, tex: str) -> None:
    with open(path, encoding="utf-8") as f:
        html = f.read()

    items = parse_bibitems(tex)
    if items:
        num = {k: i + 1 for i, (k, _) in enumerate(items)}

        def cite_repl(m):
            keys = m.group(1).split()
            parts = ['<a href="#ref-%s">%d</a>' % (k, num[k])
                     for k in keys if k in num]
            return "[" + ", ".join(parts) + "]" if parts else ""

        html = re.sub(r'<span class="citation" data-cites="([^"]*)">\s*</span>',
                      cite_repl, html)

        lis = "\n".join('<li id="ref-%s">%s</li>' % (k, t) for k, t in items)
        new_bib = ('<div class="thebibliography">\n<h2>References</h2>\n'
                   "<ol>\n" + lis + "\n</ol>\n</div>")
        html = re.sub(r'<div class="thebibliography">.*?</div>', new_bib,
                      html, flags=re.S)

    # booktabs \cmidrule(lr){a-b} leaks as "(lr)<span>a-b</span>" cell text
    html = re.sub(r"\(lr\)<span>[^<]*</span>", "", html)
    html = re.sub(r"\(lr\)", "", html)
    html = re.sub(r"<td[^>]*>\s*</td>", "", html)
    html = re.sub(r"<tr[^>]*>\s*</tr>", "", html)

    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


# ---------------------------------------------------------------------------

def build(tex_name: str, outputs: list) -> None:
    src = os.path.join(PAPER, tex_name)
    with open(src, encoding="utf-8") as f:
        tex = f.read()
    fd, tmp = tempfile.mkstemp(suffix=".tex", dir=PAPER)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(swap_figures(tex))
        for out in outputs:
            cmd = ["pandoc", os.path.basename(tmp), "-s", "--mathjax",
                   "-o", out]
            if out.endswith(".docx"):
                cmd.remove("--mathjax")
            subprocess.run(cmd, cwd=PAPER, check=True)
            if out.endswith(".html"):
                fix_html(os.path.join(PAPER, out), tex)
            print("built", os.path.join("paper", out))
    finally:
        os.remove(tmp)


def check_html(path: str) -> None:
    with open(path, encoding="utf-8") as f:
        html = f.read()
    problems = []
    if re.search(r'data-cites=', html):
        problems.append("unresolved citation spans")
    if re.search(r"\(lr\)", html):
        problems.append(r"(lr) table-command residue")
    if "<ol>" not in html.split('class="thebibliography"')[-1][:200]:
        problems.append("bibliography not rebuilt")
    missing = [m for m in re.findall(r'src="([^"]+)"', html)
               if m.startswith("../") and
               not os.path.exists(os.path.join(PAPER, m))]
    if missing:
        problems.append("missing images: %s" % missing[:3])
    print(("OK   " if not problems else "WARN ") + os.path.basename(path),
          "; ".join(problems) if problems else "")


def main():
    if shutil.which("pandoc") is None:
        sys.exit("pandoc not found on PATH")
    for tex_name, outputs in TARGETS:
        build(tex_name, outputs)
    for _, outputs in TARGETS:
        for out in outputs:
            if out.endswith(".html"):
                check_html(os.path.join(PAPER, out))


if __name__ == "__main__":
    main()
