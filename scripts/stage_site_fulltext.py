"""Stage the full-text paper HTMLs for the company website.

Takes the pandoc-built paper/main_{en,zh}.html, rewrites figure paths to the
site's public path, adds a restrained academic style override, and copies the
10 referenced figures. Output goes to site_dist/research/who-should-review-whom/
— the JA/FR translations (built from the staged en.html) land there too.

Usage:  python scripts/stage_site_fulltext.py
"""
import os
import re
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER = os.path.join(ROOT, "paper")
OUT = os.path.join(ROOT, "site_dist", "research", "who-should-review-whom")
FIG_OUT = os.path.join(OUT, "figures")
SITE_FIG_PREFIX = "/research/who-should-review-whom/figures/"

FIGURES = [
    "F1_architecture", "F2_distance_matchings_mmlu_pro",
    "F2_distance_matchings_supergpqa", "F3_accuracy_by_condition",
    "F4_accuracy_vs_tokens", "F5_decile_correction_mmlu_pro",
    "F5_decile_correction_supergpqa", "F6_error_transitions_mmlu_pro",
    "F6_error_transitions_supergpqa", "F8_repair_funnels",
]

STYLE = """
<style>
/* site override: restrained academic page on top of pandoc defaults */
html { background: #fafafa; }
body { max-width: 46rem; margin: 0 auto; padding: 3rem 1.25rem 5rem;
       font-family: -apple-system, "Segoe UI", "Helvetica Neue", "PingFang SC",
                    "Hiragino Sans GB", "Microsoft YaHei", sans-serif;
       color: #1d1d1f; line-height: 1.75; background: #fafafa; }
h1, h2, h3, h4 { line-height: 1.3; color: #111; margin-top: 2.2em; }
h1 { font-size: 1.7rem; }
h2 { font-size: 1.3rem; border-bottom: 1px solid #e3e3e6; padding-bottom: .3em; }
h3 { font-size: 1.1rem; }
header#title-block-header h1.title { font-size: 1.9rem; }
header#title-block-header p.author, header#title-block-header p.date {
  color: #6e6e73; margin: .2em 0; }
a { color: #0b5cad; text-decoration: none; }
a:hover { text-decoration: underline; }
img { max-width: 100%; height: auto; display: block; margin: 1.2rem auto; }
figure { margin: 1.6rem 0; }
figcaption { font-size: .86rem; color: #6e6e73; margin-top: .5rem; }
table { border-collapse: collapse; width: 100%; font-size: .88rem;
        margin: 1.4rem 0; }
th, td { border-top: 1px solid #d8d8dc; padding: .35em .55em; }
thead th { border-top: 2px solid #1d1d1f; border-bottom: 1px solid #1d1d1f; }
tbody tr:last-child td { border-bottom: 2px solid #1d1d1f; }
code { font-size: .88em; background: #f0f0f2; padding: .1em .3em;
       border-radius: 4px; }
pre { background: #f5f5f7; padding: 1rem; overflow-x: auto; border-radius: 8px; }
pre code { background: none; padding: 0; }
blockquote { border-left: 3px solid #d8d8dc; margin-left: 0;
             padding-left: 1.2rem; color: #3a3a3c; }
.thebibliography ol { font-size: .86rem; color: #3a3a3c; }
.thebibliography li { margin-bottom: .35em; }
.abstract { font-size: .95rem; }
hr { border: none; border-top: 1px solid #e3e3e6; margin: 2.5rem 0; }
</style>
</head>"""


def stage(lang: str) -> None:
    src = os.path.join(PAPER, "main_%s.html" % lang)
    with open(src, encoding="utf-8") as f:
        html = f.read()
    html = html.replace('src="../figures/', 'src="%s' % SITE_FIG_PREFIX)
    html = html.replace("</head>", STYLE, 1)
    dst = os.path.join(OUT, "%s.html" % lang)
    with open(dst, "w", encoding="utf-8") as f:
        f.write(html)
    n_fig = len(re.findall(r'src="%s' % re.escape(SITE_FIG_PREFIX), html))
    print("staged %s (%d figure refs)" % (dst, n_fig))


def main():
    os.makedirs(FIG_OUT, exist_ok=True)
    for fig in FIGURES:
        shutil.copy2(os.path.join(ROOT, "figures", fig + ".png"),
                     os.path.join(FIG_OUT, fig + ".png"))
    print("copied %d figures -> %s" % (len(FIGURES), FIG_OUT))
    for lang in ("en", "zh"):
        stage(lang)


if __name__ == "__main__":
    main()
