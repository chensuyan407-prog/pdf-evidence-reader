#!/usr/bin/env python3
"""Synthetic PDFs (with exact truth) for the v0.4 capabilities. Not official cases.

    python tests/v04/make_samples_v04.py OUT_DIR

  mixed_raster.pdf    text-layer page whose "Table 1" is an embedded PNG (no text inside the picture)
  three_column.pdf    3-column pages, running header + page number, a paragraph that continues
                      from page 1 column 3 to page 2 column 1, content stream written out of order
  headings.pdf        unnumbered bold headings, an ALL-CAPS bold heading, "Appendix A: ...", "A.1 ..."
  continued_table.pdf ruled Table 2 at the bottom of page 1, "Table 2 (continued)" at the top of page 2
Writes OUT_DIR/truth_v04.json.
"""
import json
import os
import sys

try:
    import pymupdf as fitz
except ImportError:
    import fitz

W, H = 612, 792
LOREM = ("The measured response follows the expected trend across all settings and the residuals remain small "
         "when the calibration is applied consistently to every run in the series.")


def para(n, tag):
    return f"{tag} " + " ".join([LOREM] * n)


def textbox(page, rect, text, size=9.5, font="helv"):
    rc = page.insert_textbox(fitz.Rect(*rect), text, fontsize=size, fontname=font)
    assert rc >= 0, (text[:40], rc)
    return rect[1] + (rect[3] - rect[1]) - rc


def mixed_raster(path):
    # 1) render a small ruled table to an image
    tmp = fitz.open()
    tp = tmp.new_page(width=300, height=110)
    rows = [["Method", "Acc", "F1"], ["Baseline", "71.2", "0.68"], ["Ours", "84.9", "0.81"]]
    for ri, r in enumerate(rows):
        for ci, c in enumerate(r):
            tp.insert_text((12 + ci * 95, 30 + ri * 28), c, fontsize=12, fontname="helv")
    for y in (10, 38, 94):
        tp.draw_line((6, y), (294, y), width=0.8)
    png = tp.get_pixmap(dpi=150).tobytes("png")
    tmp.close()
    d = fitz.open()
    p = d.new_page(width=W, height=H)
    p.insert_text((72, 80), "Raster Table Sample", fontsize=16, fontname="hebo")
    y = textbox(p, (72, 100, 540, 220), para(2, "Body text before the table."))
    p.insert_text((72, 250), "Table 1: Accuracy and F1 on the held-out split.", fontsize=9, fontname="hebo")
    p.insert_image(fitz.Rect(150, 260, 450, 370), stream=png)
    textbox(p, (72, 390, 540, 520), para(2, "Body text after the table."))
    d.save(path)
    return {"page": 1, "caption_prefix": "Table 1", "image_bbox": [150, 260, 450, 370],
            "values_in_image": {"Ours/Acc": "84.9", "Baseline/F1": "0.68"}}


def three_column(path):
    d = fitz.open()
    cols = [(40, 210), (222, 392), (404, 574)]
    order = []
    writes = []
    for pn in range(2):
        d.new_page(width=W, height=H)
        p = pn
        writes.append((p, (40, 30, 574, 44), "Synthetic Journal of Layout Tests  Vol. 1", 8))
        writes.append((p, (290, 760, 330, 774), f"{pn + 1}", 8))
        for ci, (x0, x1) in enumerate(cols):
            y = 70
            for k in range(2):
                tag = f"[P{pn + 1}C{ci + 1}K{k + 1}]"
                text = para(1, tag + " Paragraph start.")
                if pn == 0 and ci == 2 and k == 1:
                    text = tag + " Paragraph start. " + LOREM + " This sentence continues across the page break and"
                if pn == 1 and ci == 0 and k == 0:
                    text = "ends on the next page with the remaining words of the same sentence. " + LOREM
                    tag = "[CONT]"
                writes.append((p, (x0, y, x1, y + 150), text, 9))
                order.append(tag)
                y += 170
    # content stream out of reading order: reverse
    for p, rect, text, size in reversed(writes):
        textbox(d[p], rect, text, size=size)
    d.save(path)
    return {"order": order, "header": "Synthetic Journal of Layout Tests",
            "continued_tail": "continues across the page break and ends on the next page"}


def headings(path):
    d = fitz.open()
    p = d.new_page(width=W, height=H)
    p.insert_text((72, 70), "Heading Structure Sample", fontsize=17, fontname="hebo")
    y = 90
    p.insert_text((72, y + 10), "Abstract", fontsize=11, fontname="hebo")
    y = textbox(p, (72, y + 16, 540, y + 80), para(1, "We study headings.")) + 14
    plan = [("Experimental Setup", "hebo", 11), ("RELATED WORK", "hebo", 11), ("Data Collection", "hebo", 10)]
    for title, font, size in plan:
        p.insert_text((72, y + 10), title, fontsize=size, fontname=font)
        y = textbox(p, (72, y + 16, 540, y + 90), para(1, f"Text under {title}.")) + 14
    d.new_page(width=W, height=H)
    p2 = d[1]
    y = 60
    for title in ("Appendix A: Proofs of the Main Results", "A.1 Proof of Lemma 1"):
        p2.insert_text((72, y + 10), title, fontsize=11, fontname="hebo")
        y = textbox(p2, (72, y + 16, 540, y + 90), para(1, f"Text under {title}.")) + 14
    d.save(path)
    return {"headings": [{"title": "Experimental Setup", "level": 1}, {"title": "RELATED WORK", "level": 1},
                         {"title": "Data Collection", "level": 2},
                         {"title": "Proofs of the Main Results", "number": "A", "level": 1},
                         {"title": "Proof of Lemma 1", "number": "A.1", "level": 2}],
            "not_headings": ["Heading Structure Sample"]}


def continued_table(path):
    d = fitz.open()
    hdr = ["Material", "Density", "Modulus"]
    data1 = [["Steel", "7.85", "200"], ["Aluminium", "2.70", "69"], ["Copper", "8.96", "117"]]
    data2 = [["Titanium", "4.51", "116"], ["Nickel", "8.91", "200"], ["Zinc", "7.14", "108"]]
    cw = [150, 110, 110]

    def table(p, y, rows):
        x0 = 120
        p.draw_line((x0, y), (x0 + sum(cw), y), width=0.9)
        for ri, r in enumerate(rows):
            for ci, c in enumerate(r):
                p.insert_text((x0 + sum(cw[:ci]) + 4, y + 14 + ri * 18), c, fontsize=9, fontname="helv")
            if ri == 0:
                p.draw_line((x0, y + 19), (x0 + sum(cw), y + 19), width=0.5)
        yb = y + 19 + 18 * (len(rows) - 1) + 4
        p.draw_line((x0, yb), (x0 + sum(cw), yb), width=0.9)
        return yb

    d.new_page(width=W, height=H)
    p1 = d[0]
    textbox(p1, (72, 60, 540, 500), para(5, "Body text on page one."))
    p1.insert_text((120, 590), "Table 2: Properties of common metals.", fontsize=9, fontname="hebo")
    table(p1, 600, [hdr] + data1)
    d.new_page(width=W, height=H)
    p1, p2 = d[0], d[1]
    p2.insert_text((120, 60), "Table 2 (continued)", fontsize=9, fontname="hebo")
    yb = table(p2, 70, [hdr] + data2)
    textbox(p2, (72, yb + 30, 540, yb + 300), para(3, "Body text on page two."))
    d.save(path)
    return {"caption_prefix": "Table 2", "row_key": "Material",
            "cells": [{"row": "Copper", "col": "Density", "value": "8.96", "page": 1},
                      {"row": "Nickel", "col": "Modulus", "value": "200", "page": 2},
                      {"row": "Zinc", "col": "Density", "value": "7.14", "page": 2}]}


def main(out):
    os.makedirs(out, exist_ok=True)
    truth = {"mixed_raster": mixed_raster(os.path.join(out, "mixed_raster.pdf")),
             "three_column": three_column(os.path.join(out, "three_column.pdf")),
             "headings": headings(os.path.join(out, "headings.pdf")),
             "continued_table": continued_table(os.path.join(out, "continued_table.pdf"))}
    with open(os.path.join(out, "truth_v04.json"), "w", encoding="utf-8") as fh:
        json.dump(truth, fh, ensure_ascii=False, indent=1)
    print(json.dumps({"out": out, "files": sorted(truth)}))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples"))
