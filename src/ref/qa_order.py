#!/usr/bin/env python3
"""QA: no chapter may claim the reader has already met something explained later.

The article promises a bottom-up read — basics first, then assembly. A forward
*pointer* is fine and useful ("§7 builds the code this chapter uses"). A claim of
prior knowledge is not ("which you met in §7"): it tells the reader they understand
something they have not been shown, which is exactly the failure the ordering rule
exists to prevent, and it reads as a bug to anyone following the chapters in order.

So for every `§N` reference this finds the section the reference sits in, looks at
the words around it, and if those words assert that the reader has ALREADY seen the
target, the target must not be a later chapter. References inside the same chapter
are always allowed — a subsection may point at its own neighbours.

Usage:  python3 src/ref/qa_order.py [-v] [parts_dir]
"""
import pathlib, re, sys

HERE = pathlib.Path(__file__).resolve().parent
ARGS = [a for a in sys.argv[1:] if not a.startswith("-")]
PARTS = pathlib.Path(ARGS[0]) if ARGS else HERE.parent / "parts"

SEC_DEF = re.compile(r'<span class="secno">§\s*(\d+)</span>')
SEC_REF = re.compile(r"§\s*(\d+)(?:\.(\d+))?")

# Phrases that assert the reader has already been shown the target. Each is tested
# against a window of text around a reference; if one matches, the reference had
# better point backwards.
PAST = [
    (r"met in §", "says the reader met it"),
    (r"(?:we|you) met §", "says the reader met it"),
    (r"(?:saw|seen|read) in §", "says the reader saw it"),
    (r"recall(?:ing)? (?:the |that |from )?[^.]{0,40}§", "asks the reader to recall it"),
    (r"§[\d.]+\s+above\b", "calls it 'above'"),
    (r"above in §", "calls it 'above'"),
    (r"earlier(?: in| §|, §)", "calls it 'earlier'"),
    (r"\b(?:as|which|that) §[\d.]+ (?:explained|showed|built|derived|proved|gave|taught|"
     r"established|did|made)\b", "uses the past tense about it"),
    (r"\b(?:already|just) (?:met|seen|explained|built|derived|showed)[^.]{0,40}§",
     "says it was already covered"),
    (r"from §[\d.]+ (?:above|earlier)", "calls it 'above'"),
    (r"the (?:clock wall|ladder|formula|figure|table|arithmetic|rule|trick) of §[\d.]+ "
     r"(?:above|earlier)", "calls it 'above'"),
]
PAST_RE = [(re.compile(p, re.I), why) for p, why in PAST]
WINDOW_BEFORE, WINDOW_AFTER = 90, 70


def main():
    fatal, checked = [], 0
    for part in sorted(PARTS.glob("[0-9][0-9]_*.html")):
        text = part.read_text()
        # current section at every character position
        marks = [(m.start(), int(m.group(1))) for m in SEC_DEF.finditer(text)]
        lines_upto = [text.count("\n", 0, i) + 1 for i in range(len(text))]

        def section_at(pos):
            cur = 0
            for start, num in marks:
                if start <= pos:
                    cur = num
                else:
                    break
            return cur

        for m in SEC_REF.finditer(text):
            ref = int(m.group(1))
            here = section_at(m.start())
            if here == 0:
                continue                      # before the first section (hero, banners)
            checked += 1
            if ref <= here:
                continue                      # backwards or same chapter: always fine
            lo = max(0, m.start() - WINDOW_BEFORE)
            hi = min(len(text), m.end() + WINDOW_AFTER)
            window = re.sub(r"\s+", " ", text[lo:hi])
            for rx, why in PAST_RE:
                if rx.search(window):
                    fatal.append(
                        f"{part.name}:{lines_upto[m.start()]}: §{here} refers to §{m.group(0)[1:]} "
                        f"as something the reader has already seen ({why}) — but §{ref} comes later\n"
                        f"      …{window.strip()[:150]}…")
                    break

    print(f"qa_order: {checked} cross-references checked for bottom-up order | {len(fatal)} "
          f"claiming knowledge the reader does not have yet")
    for f in fatal:
        print("   !!", f)
    return 1 if fatal else 0


if __name__ == "__main__":
    sys.exit(main())
