#!/usr/bin/env python3
"""Exact, syntax-highlighted code extraction from src/flux2/model.py.

Two HTML directives are supported inside the article parts:

  <!--CODE 170-265 hl="180,185-187" title="Flux2.forward_kv_extract"-->
      -> a code block with real line numbers, highlighted lines, byte-exact source.

  <!--TRACE-->
  <!--T 401|Explanation of line 401.-->
  <!--T 402-405|Explanation spanning several lines.-->
  <!--/TRACE-->
      -> a line-by-line annotated walkthrough; the code text is pulled from the
         source file so it can never drift from the real thing.
"""
import io, os, re, html, keyword, tokenize, pathlib

HERE = pathlib.Path(__file__).resolve().parent
SRC = pathlib.Path(os.environ.get("FLUX2_SRC", str(HERE / "ref" / "model.py")))
LINES = SRC.read_text().split("\n")

KW = set(keyword.kwlist) | {"None", "True", "False"}
BUILTIN_TYPES = {"Tensor", "nn", "torch", "F", "rearrange", "math", "field", "dataclass",
                 "tuple", "list", "dict", "int", "float", "str", "bool", "set"}

_token_spans = None


def _build_spans():
    """Map every source line to a list of (start_col, end_col, css_class)."""
    global _token_spans
    if _token_spans is not None:
        return _token_spans
    per = [[] for _ in LINES]
    src = SRC.read_text()
    names = [tok for tok in tokenize.generate_tokens(io.StringIO(src).readline)]
    for i, tok in enumerate(names):
        tt, txt, (sr, sc), (er, ec) = tok.type, tok.string, tok.start, tok.end
        cls = None
        if tt == tokenize.COMMENT:
            cls = "c"
        elif tt == tokenize.STRING or tt in (getattr(tokenize, "FSTRING_START", -1),
                                             getattr(tokenize, "FSTRING_MIDDLE", -1),
                                             getattr(tokenize, "FSTRING_END", -1)):
            cls = "s"
        elif tt == tokenize.NUMBER:
            cls = "n"
        elif tt == tokenize.NAME:
            prev = names[i - 1].string if i else ""
            nxt = names[i + 1].string if i + 1 < len(names) else ""
            if txt in KW:
                cls = "k"
            elif txt == "self":
                cls = "self"
            elif prev in ("def",):
                cls = "f"
            elif prev in ("class",):
                cls = "t"
            elif nxt == "(":
                cls = "f"
            elif txt in BUILTIN_TYPES:
                cls = "t"
            elif txt.startswith("_") and len(txt) > 1:
                cls = "f"
        elif tt == tokenize.OP:
            cls = "o" if txt in "=+-*/%@<>!&|^~:" else "p"
        if cls is None:
            continue
        r, c = sr, sc
        while r <= er:
            c_end = ec if r == er else len(LINES[r - 1])
            if c < c_end:
                per[r - 1].append((c, c_end, cls))
            r, c = r + 1, 0
    for p in per:
        p.sort()
    _token_spans = per
    return per


def _hl_line(text: str, spans) -> str:
    out, pos = [], 0
    for a, b, cls in spans:
        if a < pos:
            continue
        out.append(html.escape(text[pos:a]))
        out.append(f'<span class="{cls}">{html.escape(text[a:b])}</span>')
        pos = b
    out.append(html.escape(text[pos:]))
    return "".join(out)


def _ranges(spec: str):
    res = set()
    for part in (spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            res |= set(range(int(a), int(b) + 1))
        else:
            res.add(int(part))
    return res


def code_block(spec: str) -> str:
    m = re.match(r'\s*([\d]+)-([\d]+)\s*(.*)$', spec)
    lo, hi, rest = int(m.group(1)), int(m.group(2)), m.group(3)
    hlm = re.search(r'hl="([^"]*)"', rest)
    ttm = re.search(r'title="([^"]*)"', rest)
    hls = _ranges(hlm.group(1) if hlm else "")
    title = ttm.group(1) if ttm else f"model.py"
    spans = _build_spans()

    rows = []
    for ln in range(lo, hi + 1):
        raw = LINES[ln - 1]
        cls = ' class="hl"' if ln in hls else ""
        body = _hl_line(raw, spans[ln - 1]) or "&nbsp;"
        rows.append(
            f'<tr{cls}><td class="ln-col">{ln}</td><td class="code-col">{body}</td></tr>'
        )
    nlines = hi - lo + 1
    return (
        f'<div class="codeblock"><div class="cb-head"><span class="dot"></span>'
        f'<span class="fname">src/flux2/{title}</span>'
        f'<span class="ln">lines {lo}–{hi} · {nlines} line{"s" if nlines != 1 else ""}</span></div>'
        f'<pre><table class="lineno"><tbody>{"".join(rows)}</tbody></table></pre></div>'
    )


def trace_row(spec: str) -> str:
    m = re.match(r'\s*([\d]+)(?:-([\d]+))?\s*\|(.*)$', spec, re.S)
    lo = int(m.group(1))
    hi = int(m.group(2)) if m.group(2) else lo
    note = m.group(3).strip()
    spans = _build_spans()
    code_html = "<br>".join(_hl_line(LINES[l - 1], spans[l - 1]) or "&nbsp;" for l in range(lo, hi + 1))
    lab = f"L{lo}" if lo == hi else f"L{lo}–{hi}"
    return (f'<div class="tr-row"><div class="tr-ln"><span>{lab}</span></div>'
            f'<div class="tr-body"><div class="tr-code">{code_html}</div>'
            f'<div class="tr-note">{note}</div></div></div>')


def expand(htmltext: str) -> str:
    htmltext = re.sub(r"<!--CODE ([^>]*?)-->", lambda m: code_block(m.group(1)), htmltext)

    def do_trace(m):
        inner = m.group(1)
        rows = re.findall(r"<!--T (.*?)-->", inner, re.S)
        return '<div class="trace">' + "".join(trace_row(r) for r in rows) + "</div>"

    return re.sub(r"<!--TRACE-->(.*?)<!--/TRACE-->", do_trace, htmltext, flags=re.S)


def _hl_blob(text: str) -> str:
    """Highlight an arbitrary snippet (not tied to model.py line numbers)."""
    lines = text.split("\n")
    spans = [[] for _ in lines]
    toks = list(tokenize.generate_tokens(io.StringIO(text + "\n").readline))
    for i, tok in enumerate(toks):
        tt, txt, (sr, sc), (er, ec) = tok.type, tok.string, tok.start, tok.end
        cls = None
        if tt == tokenize.COMMENT:
            cls = "c"
        elif tt == tokenize.STRING:
            cls = "s"
        elif tt == tokenize.NUMBER:
            cls = "n"
        elif tt == tokenize.NAME:
            prev = toks[i - 1].string if i else ""
            nxt = toks[i + 1].string if i + 1 < len(toks) else ""
            if txt in KW:
                cls = "k"
            elif txt == "self":
                cls = "self"
            elif prev == "def":
                cls = "f"
            elif prev == "class":
                cls = "t"
            elif nxt == "(":
                cls = "f"
            elif txt in BUILTIN_TYPES:
                cls = "t"
        elif tt == tokenize.OP:
            cls = "o" if txt in "=+-*/%@<>!&|^~:" else "p"
        if cls is None:
            continue
        r, c = sr, sc
        while r <= er:
            c_end = ec if r == er else len(lines[r - 1])
            if c < c_end:
                spans[r - 1].append((c, c_end, cls))
            r, c = r + 1, 0
    for s in spans:
        s.sort()
    rows = []
    for ln, line in enumerate(lines):
        body = _hl_line(line, spans[ln]) or "&nbsp;"
        rows.append(f'<tr><td class="ln-col">{ln + 1}</td><td class="code-col">{body}</td></tr>')
    return "".join(rows)


def raw_block(spec: str, body: str) -> str:
    m = re.search(r'title="([^"]*)"', spec)
    title = m.group(1) if m else "snippet"
    code = body.strip("\n")
    indent = min(
        (len(l) - len(l.lstrip()) for l in code.split("\n") if l.strip()), default=0
    )
    code = "\n".join(l[indent:] for l in code.split("\n"))
    return (
        f'<div class="codeblock"><div class="cb-head"><span class="dot"></span>'
        f'<span class="fname">{html.escape(title)}</span>'
        f'<span class="ln">reference snippet</span></div>'
        f'<pre><table class="lineno"><tbody>{_hl_blob(code)}</tbody></table></pre></div>'
    )


def _expand_raw(htmltext: str) -> str:
    return re.sub(
        r'<!--RAW ([^>]*?)-->(.*?)<!--/RAW-->',
        lambda m: raw_block(m.group(1), m.group(2)),
        htmltext,
        flags=re.S,
    )


_orig_expand = expand


def expand(htmltext: str) -> str:
    return _orig_expand(_expand_raw(htmltext))
