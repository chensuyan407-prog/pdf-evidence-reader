#!/usr/bin/env python3
"""Numbered display equations: find them by their number, gather the pieces,
decode unmapped math glyphs, and rebuild a linear LaTeX / Unicode form.

Used by parse_pdf.py. Nothing here knows about particular papers.

1. Glyph names. Characters that PDF text extraction returns as U+FFFD (no
   ToUnicode entry, typical for TeX math-extension fonts) are resolved through
   the embedded font program: the glyph id from page.get_texttrace() indexes
   the font's glyph order (Type 1: CharStrings order, .notdef swapped to slot 0
   as FreeType does; CFF: charset; TrueType: glyph order). Glyph names such as
   "summationdisplay" or "parenleftBig" map to symbols.
2. Anchors. A line that is just "(n)" at the right of a column, or a line
   that ends with "(n)" after a wide gap, anchors equation n.
3. Region. Starting from the anchor, math-like lines in the same column are
   added upward/downward while they touch the region; prose lines stop it.
4. Layout. Big operators take the characters directly above/below them as
   limits; accents (˜ ˆ ¯ ˙ ¨ →) attach to the letter they overlap; smaller
   raised/lowered characters become super/subscripts; a horizontal rule with
   characters above and below becomes a fraction; a radical followed by a rule
   becomes a square root. Anything that cannot be placed is listed in
   `uncertain_symbols` and the status is 'partial' instead of 'auto'.
"""
import binascii
import io
import re
import statistics
from collections import Counter

try:
    import pymupdf as fitz
except ImportError:  # pragma: no cover
    import fitz

MATH_FONT_NAME = re.compile(
    r"^(CM(MI|SY|EX|BX|R|MIB|BSY)\d*|MSBM|MSAM|EUFM|EUSM|rsfs|r?n?tx(mi|r|sy|ex|b?mia?|bsy|bex)\w*|"
    r"px(mi|sy|ex|r)\w*|zx\w*|esint\w*|stmary\w*|wasy\w*|lm(math)?\w*|.*Math.*|Symbol.*|STIX.*|XITS.*)$", re.I)
ITALIC_TEXT_FONT = re.compile(r"(Ital|Italic|Oblique|-It\b|CMMI|rntxmi|rtxmi|txmi)", re.I)

BIGOPS = {"∑": "\\sum", "∏": "\\prod", "∐": "\\coprod", "∫": "\\int", "∮": "\\oint", "⋃": "\\bigcup",
          "⋂": "\\bigcap", "⨁": "\\bigoplus", "⨂": "\\bigotimes", "⋀": "\\bigwedge", "⋁": "\\bigvee"}
ACCENTS = {"˜": ("\\tilde", "̃"), "~": ("\\tilde", "̃"), "ˆ": ("\\hat", "̂"), "^": ("\\hat", "̂"),
           "¯": ("\\bar", "̄"), "˙": ("\\dot", "̇"), "¨": ("\\ddot", "̈"), "→": ("\\vec", "⃗"),
           "´": ("\\acute", "́"), "`": ("\\grave", "̀"), "ˇ": ("\\check", "̌"), "˘": ("\\breve", "̆")}
GLYPH_BASE = {
    "summation": ("∑", "\\sum"), "product": ("∏", "\\prod"), "coproduct": ("∐", "\\coprod"),
    "integral": ("∫", "\\int"), "contourintegral": ("∮", "\\oint"), "union": ("⋃", "\\bigcup"),
    "intersection": ("⋂", "\\bigcap"), "circleplus": ("⨁", "\\bigoplus"), "circlemultiply": ("⨂", "\\bigotimes"),
    "logicaland": ("⋀", "\\bigwedge"), "logicalor": ("⋁", "\\bigvee"),
    "parenleft": ("(", "("), "parenright": (")", ")"), "bracketleft": ("[", "["), "bracketright": ("]", "]"),
    "braceleft": ("{", "\\{"), "braceright": ("}", "\\}"), "angleleft": ("⟨", "\\langle"), "angleright": ("⟩", "\\rangle"),
    "floorleft": ("⌊", "\\lfloor"), "floorright": ("⌋", "\\rfloor"), "ceilingleft": ("⌈", "\\lceil"),
    "ceilingright": ("⌉", "\\rceil"), "radical": ("√", "\\sqrt"), "slash": ("/", "/"), "backslash": ("\\", "\\backslash"),
    "bar": ("|", "|"), "bardbl": ("‖", "\\|"), "minus": ("−", "-"), "multiply": ("×", "\\times"),
    "similar": ("∼", "\\sim"), "element": ("∈", "\\in"), "greaterequal": ("≥", "\\geq"), "lessequal": ("≤", "\\leq"),
    "arrowright": ("→", "\\to"), "arrowboth": ("↔", "\\leftrightarrow"), "infinity": ("∞", "\\infty"),
    "partialdiff": ("∂", "\\partial"), "nabla": ("∇", "\\nabla"), "periodcentered": ("·", "\\cdot"),
    "dotmath": ("·", "\\cdot"), "plusminus": ("±", "\\pm"), "approxequal": ("≈", "\\approx"), "notequal": ("≠", "\\neq"),
    "proportional": ("∝", "\\propto"), "tildewide": ("˜", "\\widetilde"), "hatwide": ("ˆ", "\\widehat"),
}
GREEK = {"α": "\\alpha", "β": "\\beta", "γ": "\\gamma", "δ": "\\delta", "ε": "\\epsilon", "ϵ": "\\epsilon",
         "ζ": "\\zeta", "η": "\\eta", "θ": "\\theta", "κ": "\\kappa", "λ": "\\lambda", "μ": "\\mu", "ν": "\\nu",
         "ξ": "\\xi", "π": "\\pi", "ρ": "\\rho", "σ": "\\sigma", "τ": "\\tau", "φ": "\\phi", "ϕ": "\\phi",
         "χ": "\\chi", "ψ": "\\psi", "ω": "\\omega", "Γ": "\\Gamma", "Δ": "\\Delta", "Θ": "\\Theta", "Λ": "\\Lambda",
         "Σ": "\\Sigma", "Φ": "\\Phi", "Ψ": "\\Psi", "Ω": "\\Omega", "∑": "\\sum", "−": "-", "×": "\\times",
         "·": "\\cdot", "≤": "\\leq", "≥": "\\geq", "≠": "\\neq", "≈": "\\approx", "∈": "\\in", "∞": "\\infty",
         "∂": "\\partial", "∇": "\\nabla", "±": "\\pm", "→": "\\to", "∼": "\\sim", "√": "\\sqrt"}
FUNC_NAMES = ("log", "exp", "max", "min", "sin", "cos", "tan", "tanh", "lim", "arg", "det", "sup", "inf", "ln")
RE_EQNUM = re.compile(r"^\(\s*((?:\d+\.)?\d+[a-z]?|[A-Z]\.?\d+)\s*\)$")
RE_EQNUM_END = re.compile(r"\(\s*((?:\d+\.)?\d+[a-z]?|[A-Z]\.?\d+)\s*\)\s*$")


def _strip_subset(name):
    return name.split("+", 1)[1] if "+" in (name or "") else (name or "")


# ----------------------------------------------------------------- glyph names
def _eexec_decrypt(data, r=55665):
    out = bytearray()
    for b in data:
        out.append(b ^ (r >> 8))
        r = ((b + r) * 52845 + 22719) & 0xFFFF
    return bytes(out[4:])


def _type1_glyph_order(buf):
    i = buf.find(b"eexec")
    if i < 0:
        return None
    rest = buf[i + 5:].lstrip(b"\r\n\t ")
    if re.match(rb"^[0-9A-Fa-f]{8}", rest):
        hexpart = re.sub(rb"[^0-9A-Fa-f]", b"", rest.split(b"00000000000000000000")[0])
        data = binascii.unhexlify(hexpart[:len(hexpart) // 2 * 2])
    else:
        data = rest
    dec = _eexec_decrypt(data)
    j = dec.find(b"/CharStrings")
    if j < 0:
        return None
    names, k = [], j
    pat = re.compile(rb"/([^\s/\[\]{}()<>]+)\s+(\d+)\s+(RD|-\|)\s")
    while True:
        m = pat.search(dec, k)
        if not m:
            break
        names.append(m.group(1).decode("latin1"))
        k = m.end() + int(m.group(2))
    if ".notdef" in names and names[0] != ".notdef":  # FreeType puts .notdef at index 0
        n = names.index(".notdef")
        names[0], names[n] = names[n], names[0]
    return names


def _glyph_order(doc, xref, cache):
    if xref in cache:
        return cache[xref]
    order = None
    try:
        _, ext, _, buf = doc.extract_font(xref)
        if buf:
            if ext in ("pfa", "pfb", "t1") or buf[:2] == b"%!":
                order = _type1_glyph_order(buf)
            elif ext == "cff":
                from fontTools.cffLib import CFFFontSet
                c = CFFFontSet()
                c.decompile(io.BytesIO(buf), None)
                order = list(c[0].charset)
            elif ext in ("ttf", "otf"):
                from fontTools.ttLib import TTFont
                order = TTFont(io.BytesIO(buf)).getGlyphOrder()
    except Exception:
        order = None
    cache[xref] = order
    return order


def glyph_symbol(name):
    """(unicode, latex) for a glyph name, or None."""
    if not name:
        return None
    base = re.sub(r"(big|Big|bigg|Bigg|text|display|tp|bt|mid|ex|wide|wider|widest)+$", "", name)
    if base in GLYPH_BASE:
        return GLYPH_BASE[base]
    if name in GLYPH_BASE:
        return GLYPH_BASE[name]
    try:
        from fontTools import agl
        u = agl.toUnicode(name)
        if u and u != "�":
            return u, GREEK.get(u, u)
    except Exception:
        pass
    return None


class GlyphResolver:
    """Resolve U+FFFD characters on one page by font + origin -> glyph id -> glyph name."""

    def __init__(self, doc, page):
        self.doc, self.page, self.cache = doc, page, {}
        self.font_xref = {}
        try:
            for f in page.get_fonts(full=True):
                self.font_xref.setdefault(_strip_subset(f[3]), f[0])
        except Exception:
            pass
        self.gid = {}
        try:
            for sp in page.get_texttrace():
                fname = _strip_subset(sp.get("font", ""))
                for c in sp.get("chars", []):
                    # no Unicode for this glyph: rawdict then returns the raw char code (e.g. 'X' for a CMEX sum)
                    if c[0] == 0xFFFD:
                        self.gid[(fname, round(c[2][0], 1), round(c[2][1], 1))] = c[1]
        except Exception:
            pass

    def unmapped(self, font, origin):
        return (_strip_subset(font), round(origin[0], 1), round(origin[1], 1)) in self.gid

    def resolve(self, font, origin):
        fname = _strip_subset(font)
        g = self.gid.get((fname, round(origin[0], 1), round(origin[1], 1)))
        xref = self.font_xref.get(fname)
        if g is None or xref is None:
            return None, None
        order = _glyph_order(self.doc, xref, self.cache)
        if not order or g >= len(order):
            return None, None
        name = order[g]
        return name, glyph_symbol(name)


# ----------------------------------------------------------------- characters
def char_style(c, font):
    """(latex, unicode) for one character with its font's math alphabet."""
    f = _strip_subset(font)
    if c.isascii() and c.isalpha():
        if re.match(r"CMSY|EUSM|rsfs|txsy|pxsy", f, re.I) and c.isupper():
            return f"\\mathcal{{{c}}}", _alpha(c, "cal")
        if re.match(r"MSBM", f, re.I) and c.isupper():
            return f"\\mathbb{{{c}}}", _alpha(c, "bb")
        if re.match(r"EUFM", f, re.I):
            return f"\\mathfrak{{{c}}}", _alpha(c, "frak")
        if re.match(r"CMBX|CMMIB|CMBSY|r?n?txb|pxb", f, re.I) or (re.search(r"bold", f, re.I) and MATH_FONT_NAME.match(f)):
            return f"\\mathbf{{{c}}}", _alpha(c, "bold")
    if c in "{}" and re.match(r"CMSY|txsy|pxsy", f, re.I):
        return "\\" + c, c
    return GREEK.get(c, c), c


def _alpha(ch, style):
    up = ch.isupper()
    off = ord(ch) - (65 if up else 97)
    if style == "bold":
        return chr((0x1D400 if up else 0x1D41A) + off)
    if style == "cal" and up:
        return {"B": "ℬ", "E": "ℰ", "F": "ℱ", "H": "ℋ", "I": "ℐ", "L": "ℒ", "M": "ℳ", "R": "ℛ"}.get(ch, chr(0x1D49C + off))
    if style == "bb" and up:
        return {"C": "ℂ", "H": "ℍ", "N": "ℕ", "P": "ℙ", "Q": "ℚ", "R": "ℝ", "Z": "ℤ"}.get(ch, chr(0x1D538 + off))
    if style == "frak":
        return chr((0x1D504 if up else 0x1D51E) + off)
    return ch


def page_chars(page, resolver, clip=None):
    """All non-space characters of a page with font, size, bbox, origin; U+FFFD resolved where possible."""
    out = []
    d = page.get_text("rawdict", clip=clip)
    for b in d.get("blocks", []):
        for l in b.get("lines", []):
            if abs(l.get("dir", (1, 0))[0]) < 0.99:
                continue
            for s in l.get("spans", []):
                for ch in s.get("chars", []):
                    c = ch["c"]
                    if c.isspace():
                        continue
                    rec = {"c": c, "font": s.get("font", ""), "size": s["size"], "bbox": list(ch["bbox"]),
                           "origin": list(ch["origin"]), "line": tuple(round(v, 1) for v in l["bbox"]), "glyph": None}
                    if resolver is not None and (c == "�" or resolver.unmapped(s.get("font", ""), ch["origin"])):
                        name, sym = resolver.resolve(s.get("font", ""), ch["origin"])
                        rec["glyph"] = name or "unknown"
                        if sym:
                            rec["c"], rec["latex_sym"] = sym
                        else:
                            rec["c"] = "�"
                    out.append(rec)
    return out


# ----------------------------------------------------------------- lines and anchors
def group_lines(chars):
    lines = {}
    for c in chars:
        lines.setdefault(c["line"], []).append(c)
    out = []
    for key, cs in lines.items():
        cs.sort(key=lambda c: c["bbox"][0])
        text = ""
        prev = None
        for c in cs:
            if prev is not None and c["bbox"][0] - prev["bbox"][2] > 0.25 * max(c["size"], prev["size"]):
                text += " "
            text += c["c"]
            prev = c
        out.append({"bbox": [min(c["bbox"][0] for c in cs), min(c["bbox"][1] for c in cs),
                             max(c["bbox"][2] for c in cs), max(c["bbox"][3] for c in cs)], "chars": cs, "text": text})
    out.sort(key=lambda l: (l["bbox"][1], l["bbox"][0]))
    return out


def is_mathy(line, col_w):
    cs = line["chars"]
    if not cs:
        return False
    mathf = sum(1 for c in cs if MATH_FONT_NAME.match(_strip_subset(c["font"])) or c.get("glyph") or c["c"] in GREEK)
    ops = sum(1 for c in cs if c["c"] in "=+−-×·∑∏∫√<>≤≥≈∼()[]{}|^_/")
    words = [w for w in re.split(r"\s+", line["text"]) if w]
    long_words = [w for w in words if re.fullmatch(r"[A-Za-z]{4,}[,.;:]?", w) and w.lower().strip(",.;:") not in FUNC_NAMES]
    width = line["bbox"][2] - line["bbox"][0]
    if len(long_words) >= 2 and (mathf + ops) < 0.5 * len(cs):
        return False  # running text
    if width < 0.35 * col_w and not long_words:
        return True
    return (mathf + ops) >= 0.3 * len(cs)


def find_anchors(lines, col_bounds):
    """Equation numbers: whole-line '(n)', or '(n)' ending a line after a wide gap."""
    out = []
    for i, l in enumerate(lines):
        t = l["text"].strip()
        m = RE_EQNUM.match(t)
        col = next((cb for cb in col_bounds if cb[0] - 2 <= (l["bbox"][0] + l["bbox"][2]) / 2 <= cb[1] + 2), None)
        if col is None:
            continue
        if m and l["bbox"][0] > col[0] + 0.55 * (col[1] - col[0]):
            out.append({"number": m.group(1), "line": i, "own_line": True, "col": col, "num_chars": l["chars"]})
            continue
        m = RE_EQNUM_END.search(t)
        if m and len(t) > len(m.group(0)) + 2:
            # the number must be separated from the formula by a wide gap
            num_chars, k = [], len(l["chars"]) - 1
            want = len(re.sub(r"\s", "", m.group(0)))
            while k >= 0 and len(num_chars) < want:
                num_chars.insert(0, l["chars"][k])
                k -= 1
            if k >= 0 and num_chars and num_chars[0]["bbox"][0] - l["chars"][k]["bbox"][2] > 12:
                out.append({"number": m.group(1), "line": i, "own_line": False, "col": col, "num_chars": num_chars})
    return out


def equation_region(lines, anchor, exclude=()):
    """Lines that belong to the equation anchored at `anchor` (same column, math-like, touching)."""
    col = anchor["col"]
    col_w = col[1] - col[0]
    in_col = [i for i, l in enumerate(lines) if col[0] - 2 <= (l["bbox"][0] + l["bbox"][2]) / 2 <= col[1] + 2
              and i not in exclude]
    seed = anchor["line"]
    region = {seed}
    top = lines[seed]["bbox"][1]
    bot = lines[seed]["bbox"][3]
    if anchor["own_line"]:
        # the formula is above (or level with) a number on its own line
        sy0, sy1 = lines[seed]["bbox"][1], lines[seed]["bbox"][3]
        level = [i for i in in_col if i != seed and min(lines[i]["bbox"][3], sy1) - max(lines[i]["bbox"][1], sy0) > 0.3 * (sy1 - sy0)
                 and lines[i]["bbox"][2] < lines[seed]["bbox"][0]]
        prev = [i for i in in_col if lines[i]["bbox"][3] <= sy0 + 2 and i != seed]
        if level:
            cand = max(level, key=lambda i: lines[i]["bbox"][2])  # formula on the same base line as its number
        elif prev:
            cand = max(prev, key=lambda i: lines[i]["bbox"][3])
        else:
            return None
        if not is_mathy(lines[cand], col_w) or lines[seed]["bbox"][1] - lines[cand]["bbox"][3] > 14:
            return None
        region.add(cand)
        top, bot = min(top, lines[cand]["bbox"][1]), max(bot, lines[cand]["bbox"][3])
    changed = True
    while changed:
        changed = False
        for i in in_col:
            if i in region:
                continue
            l = lines[i]
            near = l["bbox"][3] >= top - 9 and l["bbox"][1] <= bot + 9
            if near and is_mathy(l, col_w):
                region.add(i)
                top, bot = min(top, l["bbox"][1]), max(bot, l["bbox"][3])
                changed = True
    return sorted(region)


# ----------------------------------------------------------------- layout
def _main_metrics(chars):
    sizes = [round(c["size"], 1) for c in chars if c["c"] not in BIGOPS and not c.get("glyph")]
    if not sizes:
        sizes = [round(c["size"], 1) for c in chars]
    main = max(sizes)  # the largest text size is the base line (scripts and limits are smaller)
    base = [c["origin"][1] for c in chars if abs(c["size"] - main) < 0.3 and c["c"] not in BIGOPS and not c.get("glyph")]
    return main, (statistics.median(base) if base else statistics.median(c["origin"][1] for c in chars))


def linearize(chars, rules, uncertain, depth=0):
    """Rebuild (latex, unicode) for a set of characters placed in 2D."""
    if not chars:
        return "", ""
    main, baseline = _main_metrics(chars)
    items = []  # (x, kind, latex, unicode, level)
    used = set()
    # radicals with a rule above their argument -> \sqrt{...}
    for ri, r in enumerate(rules):
        rad = next((c for c in chars if id(c) not in used and c["c"] == "√" and abs(c["bbox"][2] - r[0]) < 3), None)
        if rad:
            arg = [c for c in chars if id(c) not in used and c is not rad and r[0] - 1 <= (c["bbox"][0] + c["bbox"][2]) / 2 <= r[1] + 1
                   and c["bbox"][1] >= r[2] - 2 and c["bbox"][1] < r[2] + 2.5 * main]
            for c in arg + [rad]:
                used.add(id(c))
            lx, ux = linearize(arg, [x for j, x in enumerate(rules) if j != ri], uncertain, depth + 1)
            items.append((rad["bbox"][0], "atom", f"\\sqrt{{{lx}}}", f"√({ux})", 0))
            rules = [x for j, x in enumerate(rules) if j != ri]
            break
    # fractions
    for r in sorted(rules, key=lambda r: r[1] - r[0], reverse=True):
        num = [c for c in chars if id(c) not in used and r[0] - 1 <= (c["bbox"][0] + c["bbox"][2]) / 2 <= r[1] + 1
               and c["bbox"][3] <= r[2] + 1.5 and c["bbox"][3] > r[2] - 2.5 * main]
        den = [c for c in chars if id(c) not in used and r[0] - 1 <= (c["bbox"][0] + c["bbox"][2]) / 2 <= r[1] + 1
               and c["bbox"][1] >= r[2] - 1.5 and c["bbox"][1] < r[2] + 2.5 * main]
        if num and den:
            for c in num + den:
                used.add(id(c))
            others = [x for x in rules if x is not r]
            nl, nu = linearize(num, others, uncertain, depth + 1)
            dl, du = linearize(den, others, uncertain, depth + 1)
            items.append((r[0], "atom", f"\\frac{{{nl}}}{{{dl}}}", f"({nu})/({du})", 0))
    rest = [c for c in chars if id(c) not in used]
    if not rest and items:
        items.sort(key=lambda t: t[0])
        return " ".join(t[2] for t in items), " ".join(t[3] for t in items)
    if rest:
        main, baseline = _main_metrics(rest)
    # big operators and their limits
    for op in [c for c in rest if c["c"] in BIGOPS and id(c) not in used]:
        # glyph boxes of big operators are unreliable; place limits relative to the main baseline
        x0, x1 = op["bbox"][0] - 6, op["bbox"][2] + 6
        lim_up = [c for c in rest if c is not op and id(c) not in used and c["size"] < 0.9 * main
                  and c["bbox"][3] < baseline - 0.7 * main and x0 <= (c["bbox"][0] + c["bbox"][2]) / 2 <= x1]
        lim_dn = [c for c in rest if c is not op and id(c) not in used and c["size"] < 0.9 * main
                  and c["bbox"][1] > baseline - 0.05 * main and x0 <= (c["bbox"][0] + c["bbox"][2]) / 2 <= x1]
        lim_dn = _whole_line_run(lim_dn, rest, used)
        lim_up = _whole_line_run(lim_up, rest, used)
        for c in lim_up + lim_dn + [op]:
            used.add(id(c))
        lx = op.get("latex_sym") or BIGOPS[op["c"]]
        ux = op["c"]
        if lim_dn:
            a, b = linearize(lim_dn, [], uncertain, depth + 1)
            lx += f"_{{{a}}}"
            ux += f"_{{{b}}}"
        if lim_up:
            a, b = linearize(lim_up, [], uncertain, depth + 1)
            lx += f"^{{{a}}}"
            ux += f"^{{{b}}}"
        items.append((op["bbox"][0], "atom", lx, ux, 0))
    # accents
    accents = {}
    for a in [c for c in rest if c["c"] in ACCENTS and id(c) not in used]:
        cands = [c for c in rest if c is not a and id(c) not in used and c["c"] not in ACCENTS
                 and min(c["bbox"][2], a["bbox"][2]) - max(c["bbox"][0], a["bbox"][0]) > 0.3 * (a["bbox"][2] - a["bbox"][0])]
        if cands:
            base = min(cands, key=lambda c: abs((c["bbox"][0] + c["bbox"][2]) / 2 - (a["bbox"][0] + a["bbox"][2]) / 2))
            accents[id(base)] = a["c"]
            used.add(id(a))
    # remaining characters: base line or scripts
    for c in rest:
        if id(c) in used:
            continue
        used.add(id(c))
        if c["c"] == "�":
            uncertain.append({"glyph": c.get("glyph") or "unknown", "font": _strip_subset(c["font"]),
                              "bbox": [round(v, 1) for v in c["bbox"]], "note": "glyph without Unicode mapping"})
            lx, ux = "\\square", "□"
        elif c.get("latex_sym"):
            lx, ux = c["latex_sym"], c["c"]
        else:
            lx, ux = char_style(c["c"], c["font"])
        if id(c) in accents:
            cmd, comb = ACCENTS[accents[id(c)]]
            lx, ux = f"{cmd}{{{lx}}}", ux + comb
        small = c["size"] < 0.85 * main
        dy = c["origin"][1] - baseline
        level = 1 if (small and dy < -0.12 * main) else (-1 if (small and dy > 0.05 * main) else 0)
        if not small and c["c"] in "()[]" and c.get("glyph"):
            level = 0
        items.append((c["bbox"][0], "char", lx, ux, level, c))
    items.sort(key=lambda t: (t[0], -t[4]))
    # emit with script grouping and roman function names
    lat, uni = [], []
    i = 0
    while i < len(items):
        it = items[i]
        if it[1] == "char" and it[4] == 0:
            run = [it]
            j = i + 1
            while j < len(items) and items[j][1] == "char" and items[j][4] == 0 and \
                    items[j][5]["bbox"][0] - run[-1][5]["bbox"][2] < 0.15 * main:
                run.append(items[j])
                j += 1
            word = "".join(t[5]["c"] for t in run)
            roman = not any(ITALIC_TEXT_FONT.search(_strip_subset(t[5]["font"])) for t in run)
            if word in FUNC_NAMES and roman:
                lat.append(f"\\{word} ")
                uni.append(word + " ")
                i = j
                continue
            if i > 0 and items[i - 1][1] == "char" and it[5]["bbox"][0] - items[i - 1][5]["bbox"][2] > 0.2 * main:
                uni.append(" ")
            lat.append(it[2])
            uni.append(it[3])
            i += 1
        elif it[1] == "char":
            # all scripts up to the next base character; subscripts and superscripts may share x
            grp = [it]
            j = i + 1
            while j < len(items) and items[j][1] == "char" and items[j][4] != 0:
                grp.append(items[j])
                j += 1
            subs = [t for t in grp if t[4] < 0]
            sups = [t for t in grp if t[4] > 0]
            prev = lat[-1].strip() if lat else ""
            if not prev or prev[-1:] in "-+=×,(" or prev.endswith((r"\times", r"\cdot")):
                uncertain.append({"note": f"script '{''.join(t[3] for t in grp)}' has no plausible base (follows {prev[-8:]!r})"})
            if subs:
                lat.append("_{" + "".join(t[2] for t in subs) + "}")
                uni.append("_{" + "".join(t[3] for t in subs) + "}")
            if sups:
                lat.append("^{" + "".join(t[2] for t in sups) + "}")
                uni.append("^{" + "".join(t[3] for t in sups) + "}")
            i = j
        else:
            if uni and not uni[-1].endswith(" "):
                uni.append(" ")
            lat.append(it[2] + " ")
            uni.append(it[3] + " ")
            i += 1
    latex = re.sub(r"\s+", " ", "".join(lat)).strip()
    unic = re.sub(r"\s+", " ", "".join(uni)).strip()
    return latex, unic


def _whole_line_run(sel, rest, used):
    """Extend a set of limit characters to the full contiguous run on their line (e.g. 'i, j=1')."""
    if not sel:
        return sel
    line = sel[0]["line"]
    same = sorted([c for c in rest if c["line"] == line and id(c) not in used], key=lambda c: c["bbox"][0])
    chosen = {id(c) for c in sel}
    run = [c for c in same if id(c) in chosen]
    lo = min(same.index(c) for c in run)
    hi = max(same.index(c) for c in run)
    while lo > 0 and run[0]["bbox"][0] - same[lo - 1]["bbox"][2] < 0.6 * run[0]["size"]:
        lo -= 1
        run.insert(0, same[lo])
    while hi < len(same) - 1 and same[hi + 1]["bbox"][0] - run[-1]["bbox"][2] < 0.6 * run[-1]["size"]:
        hi += 1
        run.append(same[hi])
    return run


def page_rules(page):
    rules = []
    try:
        for d in page.get_drawings():
            for it in d.get("items", []):
                if it[0] == "l" and abs(it[1].y - it[2].y) < 0.8 and 3 < abs(it[2].x - it[1].x) < 300:
                    rules.append((min(it[1].x, it[2].x), max(it[1].x, it[2].x), (it[1].y + it[2].y) / 2))
                elif it[0] == "re" and it[1].height < 1.2 and 3 < it[1].width < 300:
                    rules.append((it[1].x0, it[1].x1, (it[1].y0 + it[1].y1) / 2))
    except Exception:
        pass
    return rules


def numbered_equations(doc, page, page_no, gutters):
    """All number-anchored equations on a page: list of dicts with bbox, latex, unicode, status."""
    W = page.rect.width
    edges = [0.0] + sorted(gutters or []) + [W]
    col_bounds = list(zip(edges, edges[1:]))
    resolver = GlyphResolver(doc, page)
    chars = page_chars(page, resolver)
    lines = group_lines(chars)
    rules = page_rules(page)
    out = []
    anchors = find_anchors(lines, col_bounds)
    anchor_lines = {a["line"] for a in anchors if a["own_line"]}
    for a in anchors:
        reg = equation_region(lines, a, exclude=anchor_lines - {a["line"]})
        if not reg:
            continue
        num_ids = {id(c) for c in a["num_chars"]}
        eq_chars = [c for i in reg for c in lines[i]["chars"] if id(c) not in num_ids]
        if not eq_chars:
            continue
        x0 = min(c["bbox"][0] for c in eq_chars)
        y0 = min(c["bbox"][1] for c in eq_chars)
        x1 = max(c["bbox"][2] for c in eq_chars)
        y1 = max(c["bbox"][3] for c in eq_chars)
        rr = [r for r in rules if r[0] >= x0 - 3 and r[1] <= x1 + 3 and y0 - 2 <= r[2] <= y1 + 2]
        uncertain = []
        # several base lines in one region = stacked/aligned equations or inline prose: order of pieces unreliable
        main = max(round(c["size"], 1) for c in eq_chars)
        bl = sorted({round(c["origin"][1], 0) for c in eq_chars if abs(c["size"] - main) < 0.3 and c["c"] not in BIGOPS
                     and not c.get("glyph")})
        clusters = []
        for y in bl:
            if clusters and y - clusters[-1][-1] <= 0.8 * main:
                clusters[-1].append(y)
            else:
                clusters.append([y])
        in_fraction = lambda y: any(r[0] <= x1 and r[1] >= x0 and abs(r[2] - y) < 1.6 * main for r in rr)
        real = [c for c in clusters if not all(in_fraction(y) for y in c)]
        if len(real) > 1:
            uncertain.append({"note": f"{len(real)} base lines in the equation region (multi-line, aligned or inline text); "
                                      "the linear form may interleave pieces - use the crop"})
        # running text at the main size inside the region (not limits/scripts, which are smaller)
        words, cur = [], ""
        for c in sorted([c for c in eq_chars if abs(c["size"] - main) < 0.3], key=lambda c: (round(c["origin"][1]), c["bbox"][0])):
            if c["c"].isalpha() and c["c"].isascii() and not ITALIC_TEXT_FONT.search(_strip_subset(c["font"]))                     and not MATH_FONT_NAME.match(_strip_subset(c["font"])):
                cur += c["c"]
            else:
                words.append(cur)
                cur = ""
        words.append(cur)
        prose = [w for w in words if len(w) >= 4 and w.lower() not in FUNC_NAMES]
        if prose:
            uncertain.append({"note": f"running text inside the equation region ({', '.join(prose[:4])}); pieces may interleave"})
        latex, uni = linearize(eq_chars, rr, uncertain)
        nb = [min(c["bbox"][0] for c in a["num_chars"]), min(c["bbox"][1] for c in a["num_chars"]),
              max(c["bbox"][2] for c in a["num_chars"]), max(c["bbox"][3] for c in a["num_chars"])]
        out.append({
            "number": a["number"], "page": page_no,
            "bbox": [round(x0, 2), round(y0, 2), round(max(x1, nb[2]), 2), round(max(y1, nb[3]), 2)],
            "expr_bbox": [round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2)],
            "number_bbox": [round(v, 2) for v in nb],
            "expr_latex": latex, "expr_unicode": uni,
            "uncertain_symbols": uncertain,
            "transcription_status": "auto" if not uncertain else "partial",
            "n_chars": len(eq_chars), "n_lines": len(reg),
            "resolved_glyphs": sorted({c["glyph"] for c in eq_chars if c.get("glyph")}),
            "rule": "number-anchored region (same column, math-like lines touching the anchor)",
        })
    return out
