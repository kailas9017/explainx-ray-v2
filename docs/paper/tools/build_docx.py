"""Convert docs/paper/manuscript.tex (sn-jnl) to a Word file with yellow-highlighted TODO/RES markers.

usage: python build_docx.py <paper_dir> <figures_src_dir>
"""
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))  # docs/paper/tools
paper_dir, fig_src = sys.argv[1], sys.argv[2]
src = open(os.path.join(paper_dir, "manuscript.tex"), encoding="utf-8").read()


def find_group(s, start):
    """Return (content, end) of the brace group starting at s[start] == '{'."""
    depth = 0
    for i in range(start, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                return s[start + 1:i], i + 1
    raise ValueError("unbalanced braces")


def replace_macro(s, name, fn):
    out, i = [], 0
    pat = re.compile(r"\\" + name + r"(?![A-Za-z])")
    while True:
        m = pat.search(s, i)
        if not m:
            out.append(s[i:])
            return "".join(out)
        j = m.end()
        opt = None
        if j < len(s) and s[j] == "[":
            k = s.index("]", j)
            opt, j = s[j + 1:k], k + 1
        while j < len(s) and s[j] in " \n":
            j += 1
        arg, end = find_group(s, j)
        out.append(s[i:m.start()])
        out.append(fn(arg, opt))
        i = end


# --- body only: drop preamble comments/definitions we handle ourselves ---
body = src[src.index("\\begin{document}") + len("\\begin{document}"):src.index("\\end{document}")]

# Markers -> SmallCaps (the Lua filter highlights SmallCaps).
body = replace_macro(body, "TODO", lambda a, o: r"\textsc{TODO: " + a + "}")
body = replace_macro(body, "RES", lambda a, o: r"\textsc{[" + a + "]}")

# Title / authors / affiliation (sn-jnl macros pandoc does not know).
title = None
def take_title(a, o):
    global title
    title = a
    return ""
body = replace_macro(body, "title", take_title)
authors, emails = [], []
def take_author(a, o):
    fn = re.search(r"\\fnm\{([^}]*)\}", a).group(1)
    sn = re.search(r"\\sur\{([^}]*)\}", a).group(1)
    authors.append(f"{fn} {sn}")
    return ""
body = re.sub(r"\\author\*?", r"\\author", body)
body = replace_macro(body, "author", take_author)
body = replace_macro(body, "email", lambda a, o: emails.append(a) or "")
affil = []
def take_affil(a, o):
    parts = [re.sub(r"\\[a-z]+\{", "", p).replace("}", "") for p in re.findall(r"\\(?:orgdiv|orgname|city|postcode|state|country)\{[^}]*\}", a)]
    affil.append(", ".join(parts))
    return ""
body = re.sub(r"\\affil\*?", r"\\affil", body)
body = replace_macro(body, "affil", take_affil)
author_block = (
    "\\begin{center}\n" + ", ".join(f"{n}{'*' if i == 0 else ''}" for i, n in enumerate(authors))
    + "\\\\\n" + affil[0] + "\\\\\n"
    + "; ".join(f"{n}: {e}" for n, e in zip(authors, emails)) + "\\\\\n"
    + "*Corresponding author\n\\end{center}\n"
)
# sn-jnl puts \abstract before \maketitle; in Word the authors belong right under the title.
body = author_block + body.replace("\\maketitle", "")
body = replace_macro(body, "abstract", lambda a, o: "\\section*{Abstract}\n" + a + "\n")
body = replace_macro(body, "keywords", lambda a, o: "\n\\textbf{Keywords:} " + a + "\n")
body = replace_macro(body, "bmhead", lambda a, o: "\\paragraph{" + a + "}")
body = body.replace("\\backmatter", "")
body = body.replace("\\bibliography{references}", "\\section*{References}\n")

# Resolve \ref ourselves (numbers in order of appearance).
labels, counters = {}, {"sec": 0, "fig": 0, "tab": 0}
for m in re.finditer(r"\\(section(?!\*)|begin\{figure\}|begin\{table\})|\\label\{([^}]*)\}", body):
    if m.group(1):
        kind = {"section": "sec", "begin{figure}": "fig", "begin{table}": "tab"}[m.group(1)]
        counters[kind] += 1
        current = (kind, counters[kind])
    elif m.group(2):
        labels[m.group(2)] = current[1]
body = re.sub(r"\\ref\{([^}]*)\}", lambda m: str(labels.get(m.group(1), "??")), body)
# Number figure/table captions explicitly (pandoc docx captions are unnumbered).
fig_n = tab_n = 0
def number_caption(env):
    def fn(m):
        nonlocal_counter[env] += 1
        prefix = "Fig." if env == "figure" else "Table"
        return m.group(0).replace("\\caption{", f"\\caption{{\\textbf{{{prefix} {nonlocal_counter[env]}}} ", 1)
    return fn
nonlocal_counter = {"figure": 0, "table": 0}
def caption_numbers(s):
    out, i = [], 0
    for m in re.finditer(r"\\begin\{(figure|table)\}.*?\\end\{\1\}", s, flags=re.S):
        env = m.group(1)
        nonlocal_counter[env] += 1
        prefix = "Fig." if env == "figure" else "Table"
        block = m.group(0).replace("\\caption{", f"\\caption{{\\textbf{{{prefix} {nonlocal_counter[env]}}} ", 1)
        out.append(s[i:m.start()] + block)
        i = m.end()
    out.append(s[i:])
    return "".join(out)
body = caption_numbers(body)

# Figures: copy those that exist; mark missing ones as pending (highlighted).
fig_dir = os.path.join(HERE, "figures")
os.makedirs(fig_dir, exist_ok=True)
def figure(m):
    path = m.group(2)
    name = os.path.splitext(os.path.basename(path))[0]
    for ext in (".png", ".pdf"):
        cand = os.path.join(fig_src, name + ".png")  # Word needs raster/vector it can embed; PNG is safest
        if os.path.exists(cand):
            shutil.copy(cand, os.path.join(fig_dir, name + ".png"))
            return f"\\includegraphics{m.group(1) or ''}{{figures/{name}.png}}"
    return r"\textsc{[Figure pending: " + name + " — generated after the experiments finish]}"
body = re.sub(r"\\includegraphics(\[[^\]]*\])?\{([^}]*)\}", figure, body)

tex = ("\\documentclass{article}\n\\usepackage{amsmath,amssymb,graphicx,booktabs,multirow}\n"
       f"\\title{{{title}}}\n\\begin{{document}}\n\\maketitle\n" + body + "\n\\end{document}\n")
tmp = os.path.join(HERE, "manuscript_pandoc.tex")
open(tmp, "w", encoding="utf-8").write(tex)

out = os.path.join(paper_dir, "manuscript.docx")
cmd = ["pandoc", tmp, "-f", "latex", "-o", out, "--resource-path", HERE,
       "--citeproc", "--bibliography", os.path.join(paper_dir, "references.bib"),
       "--csl", os.path.join(HERE, "springer-basic-brackets.csl"),
       "--lua-filter", os.path.join(HERE, "highlight.lua")]
res = subprocess.run(cmd, capture_output=True, text=True, cwd=HERE)
print(res.stdout, res.stderr)
print("labels:", labels)
if res.returncode != 0:
    sys.exit(f"pandoc failed (exit {res.returncode}); is {out} open in Word?")
print("wrote", out, os.path.getsize(out))
