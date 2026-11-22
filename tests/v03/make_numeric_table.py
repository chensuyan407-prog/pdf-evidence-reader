#!/usr/bin/env python3
"""Synthetic booktabs table with every number form the table parser must keep apart.

Row labels: plain, with a footnote superscript, with a parenthesis, with a citation.
Cells: integers, thousands separator, negatives (hyphen-minus), superscript
footnote markers after numbers (letter and asterisk - NOT exponents), and
scientific notation written with × and ·, plus a bare 10^n.
Ground truth is written next to the PDF.
"""
import json
import os
import sys

try:
    import pymupdf as fitz
except ImportError:
    import fitz

HERE = os.path.dirname(os.path.abspath(__file__))
COLS = [100, 215, 285, 350, 445]  # x of each column's left edge
ROWS = [
    # label, label_sup, params, delta, flops(mantissa, op, exp), score, score_sup
    ("Baseline", "a", "125", "-0.5", ("3.2", "×", "18"), "71.3", "b"),
    ("Ours", "", "340", "1.2", ("1.1", "·", "19"), "74.0", "*"),
    ("Ours (large)", "", "1,024", "-12", (None, None, "20"), "80.2", ""),
    ("Prior work [12]", "", "87", "0.0", ("4.5", "×", "17"), "69.9", ""),
]
TRUTH = {
    "columns": ["Method", "Params (M)", "dAcc", "FLOPs", "Score"],
    "cells": {
        "Baseline": {"Params (M)": 125, "dAcc": -0.5, "FLOPs": 3.2e18, "Score": 71.3, "Score_footnote": "b", "label_footnote": "a"},
        "Ours": {"Params (M)": 340, "dAcc": 1.2, "FLOPs": 1.1e19, "Score": 74.0, "Score_footnote": "*"},
        "Ours (large)": {"Params (M)": 1024, "dAcc": -12, "FLOPs": 1e20, "Score": 80.2},
        "Prior work [12]": {"Params (M)": 87, "dAcc": 0.0, "FLOPs": 4.5e17, "Score": 69.9},
    },
}


def sup(page, x, y, t, size=9):
    page.insert_text((x, y - 3.6), t, fontsize=size * 0.65, fontname="helv")
    return x + fitz.get_text_length(t, fontname="helv", fontsize=size * 0.65)


def txt(page, x, y, t, size=9, font="helv"):
    page.insert_text((x, y), t, fontsize=size, fontname=font)
    return x + fitz.get_text_length(t, fontname=font, fontsize=size)


def main(out=os.path.join(HERE, "numeric_table.pdf")):
    d = fitz.open()
    d.new_page(width=612, height=792)
    p = d[0]
    txt(p, 90, 90, "Table 1: Synthetic table with integers, negatives, footnotes and scientific notation.", 9, "hebo")
    y = 102
    p.draw_line((90, y), (530, y), width=1.0)
    for x, h in zip(COLS, TRUTH["columns"]):
        txt(p, x, y + 12, h, 9, "hebo")
    p.draw_line((90, y + 17), (530, y + 17), width=0.6)
    yy = y + 31
    for label, lsup, params, delta, (m, op, e), score, ssup in ROWS:
        x = txt(p, COLS[0], yy, label)
        if lsup:
            sup(p, x + 0.3, yy, lsup)
        txt(p, COLS[1], yy, params)
        txt(p, COLS[2], yy, delta)
        x = COLS[3]
        if m:
            x = txt(p, x, yy, m)
            x = txt(p, x + 2.5, yy, op)
            x += 2.5
        x = txt(p, x, yy, "10")
        sup(p, x + 0.2, yy, e)
        x = txt(p, COLS[4], yy, score)
        if ssup:
            sup(p, x + 0.3, yy, ssup)
        yy += 14
    p.draw_line((90, yy - 8), (530, yy - 8), width=1.0)
    txt(p, 90, yy + 12, "Footnotes a, b and * mark notes, not exponents.", 8)
    d.save(out)
    json.dump(TRUTH, open(out.replace(".pdf", ".truth.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main(*sys.argv[1:])
