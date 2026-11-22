#!/usr/bin/env python3
"""Generate synthetic layout test PDFs with exact ground truth.

These are NOT the platform's official cases. They are self-built layout
samples whose true reading order, headings, tables, equations and running
headers are known, so the parser can be scored numerically against them.

Content-stream order is deliberately scrambled (right column, footer and
headers written before the left column) to mimic real publisher PDFs, where
the stream order differs from the reading order.
"""
import json
import os
import sys

try:
    import pymupdf as fitz
except ImportError:
    import fitz

W, H = 612, 792
L0, L1, R0, R1 = 50, 296, 316, 562
BODY = 9.5


def measure(text, width, size, font="helv"):
    tmp = fitz.open()
    p = tmp.new_page(width=W, height=H)
    rect = fitz.Rect(0, 0, width, 2000)
    rc = p.insert_textbox(rect, text, fontsize=size, fontname=font)
    tmp.close()
    return 2000 - rc


class Planner:
    """Lay out text boxes per column, remember them, write later in any order."""

    def __init__(self):
        self.items = []

    def box(self, page, x0, x1, y, text, size=BODY, font="helv", gap=6, tag=None):
        h = measure(text, x1 - x0, size, font) + 2
        self.items.append({"page": page, "rect": (x0, y, x1, y + h), "text": text, "size": size, "font": font, "tag": tag})
        return y + h + gap


def sup_equation(page, x, y, base, sup, tail, size=10):
    page.insert_text((x, y), base, fontsize=size, fontname="tiro")
    w = fitz.get_text_length(base, fontname="tiro", fontsize=size)
    page.insert_text((x + w + 0.5, y - size * 0.38), sup, fontsize=size * 0.65, fontname="tiro")
    w2 = fitz.get_text_length(sup, fontname="tiro", fontsize=size * 0.65)
    page.insert_text((x + w + w2 + 1.5, y), tail, fontsize=size, fontname="tiro")


def draw_table(page, x0, y0, col_w, row_h, rows):
    ncol = len(rows[0])
    for ri, row in enumerate(rows):
        for ci, cell in enumerate(row):
            cx = x0 + sum(col_w[:ci])
            page.insert_text((cx + 4, y0 + ri * row_h + row_h - 4.5), cell, fontsize=8.5,
                             fontname="hebo" if ri == 0 else "helv")
    total_w = sum(col_w)
    for ri in range(len(rows) + 1):
        page.draw_line((x0, y0 + ri * row_h), (x0 + total_w, y0 + ri * row_h), width=0.6)
    xs = [x0]
    for w in col_w:
        xs.append(xs[-1] + w)
    for x in xs:
        page.draw_line((x, y0), (x, y0 + len(rows) * row_h), width=0.6)
    return y0 + len(rows) * row_h


# ----------------------------------------------------------------- sample A
P1_INTRO = ("Scientific articles are usually typeset in two columns. When a parser reads the "
            "page strictly from top to bottom, lines from the left and the right column are "
            "interleaved and sentences from different paragraphs are glued together.")
P2_INTRO_A = ("This paper describes a lightweight procedure that restores the reading order "
              "and keeps a pointer from every paragraph back to its page and bounding box, so that "
              "every answer can be checked against the original layout of the")
P2_INTRO_B = ("document without guessing. The procedure is evaluated on synthetic pages with a "
              "known ground truth.")
P3_METHOD = ("We first detect the column gutter from the horizontal occupancy of narrow text "
             "blocks. Blocks that cross the gutter are treated as full-width elements that cut the "
             "page into bands.")
P4_ORDER = ("Inside each band the left column is read before the right column. The energy "
            "relation below is used only as a typographic test of superscripts and equation "
            "numbers.")
P5_AFTER_EQ = ("The equation number must stay attached to the formula, and the superscript must be "
               "recovered as a superscript rather than as a plain digit that follows the letter c in "
               "the running text of the")
P5_CONT = ("same paragraph. This sentence deliberately continues on the next page to test "
           "cross-page joining.")
P6_RESULTS = ("Table 1 lists the scores of the baseline and of the layout-aware parser on the "
              "synthetic benchmark. The layout-aware parser preserves the order of all paragraphs.")
P7_FIG = ("Figure 1 shows accuracy as a function of the number of pages. No trend is claimed "
          "beyond the synthetic setting reported here.")
REFS = ["[1] A. Smith and B. Jones. Reading order detection in scanned documents. Proc. ICDAR, 2019.",
        "[2] C. Wang. Table structure recognition with graph networks. Pattern Recognition, 120:108-121, 2021.",
        "[3] D. Lee, E. Kim. Robust extraction of mathematical formulas. arXiv:2101.00001, 2021."]
TABLE1 = [["Method", "Order acc.", "Header leak"], ["Baseline", "0.61", "4"], ["Layout-aware", "1.00", "0"], ["Human", "1.00", "0"]]
HEADER_A = "Journal of Synthetic Layout Tests, Vol. 3 (2026)"


def sample_a(path):
    doc = fitz.open()
    doc.new_page(width=W, height=H)
    doc.new_page(width=W, height=H)
    p1, p2 = doc[0], doc[1]
    pl = Planner()
    # page 1 full width
    y = 70
    y = pl.box(0, 50, 562, y, "Layout-Aware Parsing of Two-Column Scientific Articles", 18, "hebo", 8, "title")
    y = pl.box(0, 50, 562, y, "Alice Author and Bob Writer", 10.5, "helv", 2, "authors")
    y = pl.box(0, 50, 562, y, "Department of Document Engineering, Example University", 9, "helv", 12, "affil")
    abstract = ("Abstract: We present a small procedure for restoring the reading order of two-column "
                "scientific PDFs while keeping page and block provenance for every extracted element.")
    y = pl.box(0, 50, 562, y, abstract, 9.5, "helv", 14, "abstract")
    top = y
    # left column
    yl = pl.box(0, L0, L1, top, "1 Introduction", 11, "hebo", 4, "h1")
    yl = pl.box(0, L0, L1, yl, P1_INTRO, tag="p1")
    yl = pl.box(0, L0, L1, yl, P2_INTRO_A, tag="p2a")
    # right column
    yr = pl.box(0, R0, R1, top, P2_INTRO_B, tag="p2b")
    yr = pl.box(0, R0, R1, yr, "2 Method", 11, "hebo", 4, "h2")
    yr = pl.box(0, R0, R1, yr, P3_METHOD, tag="p3")
    yr = pl.box(0, R0, R1, yr, "2.1 Reading order", 10, "hebo", 4, "h21")
    yr = pl.box(0, R0, R1, yr, P4_ORDER, tag="p4")
    eq_y = yr + 12
    yr = eq_y + 14
    yr = pl.box(0, R0, R1, yr, P5_AFTER_EQ, tag="p5")
    # scrambled write order for page 1: right column, footer, header, left column, top
    order = [it for it in pl.items if it["rect"][0] == R0] + [it for it in pl.items if it["rect"][0] == L0 and it["rect"][2] == L1] + \
            [it for it in pl.items if it["rect"][0] == 50 and it["rect"][2] == 562]
    p1.insert_text((300, 772), "1", fontsize=9, fontname="helv")
    p1.insert_text((50, 40), HEADER_A, fontsize=8, fontname="helv")
    for it in order:
        p1.insert_textbox(fitz.Rect(it["rect"]), it["text"], fontsize=it["size"], fontname=it["font"])
    sup_equation(p1, R0 + 70, eq_y, "E = mc", "2", "            (1)")
    p1.insert_textbox(fitz.Rect(L0, 715, L1, 745), "1 Code and samples are available from the authors on request.", fontsize=7.5, fontname="helv")
    p1.insert_text((22, 600), "arXiv:2611.00030v1 [cs.DL] 21 Nov 2026", fontsize=8, fontname="helv", rotate=90)
    # page 2
    pl2 = Planner()
    p2.insert_text((50, 40), HEADER_A, fontsize=8, fontname="helv")
    p2.insert_text((300, 772), "2", fontsize=9, fontname="helv")
    y = pl2.box(1, 50, 562, 62, "Table 1: Results on the synthetic benchmark.", 9, "hebo", 4, "cap_t1")
    ty = draw_table(p2, 120, y, [150, 110, 110], 16, TABLE1)
    top2 = ty + 18
    yl = pl2.box(1, L0, L1, top2, P5_CONT, tag="p5b")
    yl = pl2.box(1, L0, L1, yl, "3 Results", 11, "hebo", 4, "h3")
    yl = pl2.box(1, L0, L1, yl, P6_RESULTS, tag="p6")
    yr = top2
    p2.draw_rect(fitz.Rect(R0 + 10, yr, R1 - 10, yr + 110), color=(0, 0, 0), width=0.8)
    p2.draw_line((R0 + 20, yr + 100), (R1 - 20, yr + 20), width=1)
    yr = pl2.box(1, R0, R1, yr + 116, "Figure 1: Accuracy versus page count.", 8.5, "hebo", 8, "cap_f1")
    yr = pl2.box(1, R0, R1, yr, P7_FIG, tag="p7")
    yr = pl2.box(1, R0, R1, yr, "References", 11, "hebo", 4, "href")
    for i, r in enumerate(REFS):
        yr = pl2.box(1, R0, R1, yr, r, 8, "helv", 3, f"ref{i + 1}")
    for it in reversed(pl2.items):
        p2.insert_textbox(fitz.Rect(it["rect"]), it["text"], fontsize=it["size"], fontname=it["font"])
    doc.set_metadata({"title": "Layout-Aware Parsing of Two-Column Scientific Articles", "author": "synthetic"})
    doc.save(path)
    gt = {
        "name": "two_column_en",
        "pdf": os.path.basename(path),
        "reading_order": [abstract, P1_INTRO, P2_INTRO_A + " " + P2_INTRO_B, P3_METHOD, P4_ORDER,
                          P5_AFTER_EQ + " " + P5_CONT, P6_RESULTS, P7_FIG],
        "split_sentences": [P2_INTRO_A[-60:] + " " + P2_INTRO_B[:40], P5_AFTER_EQ[-60:] + " " + P5_CONT[:40]],
        "title": "Layout-Aware Parsing of Two-Column Scientific Articles",
        "headings": [["1", "Introduction", 1], ["2", "Method", 1], ["2.1", "Reading order", 2], ["3", "Results", 1], ["", "References", 1]],
        "captions": ["Table 1: Results on the synthetic benchmark.", "Figure 1: Accuracy versus page count."],
        "equations": [{"number": "1", "marked_contains": "^{2}"}],
        "footnotes": ["Code and samples are available from the authors on request."],
        "headers_footers": [HEADER_A],
        "page_numbers": ["1", "2"],
        "margin_notes": ["arXiv:2611.00030v1"],
        "tables": [{"caption_prefix": "Table 1", "rows": TABLE1}],
        "n_references": 3,
    }
    return gt


# ----------------------------------------------------------------- sample B (Chinese, single column)
CN_P1 = "科研文献的自动解析需要同时保留文字内容、章节层级和原文位置，以便后续的问答与核验能够回到原文证据。"
CN_P2 = "本文在单栏中文版式上测试章节编号识别、图表题注提取以及页眉页脚的剔除效果。所有样例均为合成数据。"
CN_P3 = "实验数据共包含三组样本，每组样本的页数与版式参数列于表 1，表格的行列结构需要被完整还原。"
CN_P4 = "图 1 给出了不同版式下的解析耗时，本文不对合成设定之外的情形作出推断。"
CN_TABLE = [["样本", "页数", "栏数"], ["A", "2", "2"], ["B", "1", "1"], ["C", "3", "1"]]
CN_HEADER = "合成版式测试报告 第 3 卷"


def sample_b(path):
    doc = fitz.open()
    p = doc.new_page(width=W, height=H)
    f = "china-s"
    p.insert_text((50, 40), CN_HEADER, fontsize=8, fontname=f)
    p.insert_text((300, 772), "1", fontsize=9, fontname="helv")
    y = 70
    def box(text, size, gap, x0=60, x1=552):
        nonlocal y
        h = 2000 - _cjk_measure(text, x1 - x0, size)
        p.insert_textbox(fitz.Rect(x0, y, x1, y + h + 2), text, fontsize=size, fontname=f)
        y += h + 2 + gap
    box("中文科研文献版式解析测试", 18, 10)
    box("摘要：本文构造一个单栏中文合成样例，用于检验章节层级、表格与题注的解析。", 9.5, 12)
    box("1 引言", 12, 4)
    box(CN_P1, 10, 6)
    box("2 方法", 12, 4)
    box(CN_P2, 10, 6)
    box("2.1 数据", 11, 4)
    box(CN_P3, 10, 8)
    box("表 1 样本参数", 9.5, 4)
    ty = draw_table_cn(p, 150, y, [100, 100, 100], 16, CN_TABLE)
    y = ty + 14
    p.draw_rect(fitz.Rect(150, y, 450, y + 90), width=0.8)
    y += 96
    box("图 1 解析耗时示意", 9.5, 8)
    box(CN_P4, 10, 6)
    doc.save(path)
    return {
        "name": "single_column_zh",
        "pdf": os.path.basename(path),
        "reading_order": ["摘要：本文构造一个单栏中文合成样例", CN_P1, CN_P2, CN_P3, CN_P4],
        "split_sentences": [],
        "title": "中文科研文献版式解析测试",
        "headings": [["1", "引言", 1], ["2", "方法", 1], ["2.1", "数据", 2]],
        "captions": ["表 1 样本参数", "图 1 解析耗时示意"],
        "equations": [],
        "footnotes": [],
        "headers_footers": [CN_HEADER],
        "page_numbers": ["1"],
        "margin_notes": [],
        "tables": [{"caption_prefix": "表 1", "rows": CN_TABLE}],
        "n_references": 0,
    }


def _cjk_measure(text, width, size):
    tmp = fitz.open()
    pg = tmp.new_page(width=W, height=H)
    rc = pg.insert_textbox(fitz.Rect(0, 0, width, 2000), text, fontsize=size, fontname="china-s")
    tmp.close()
    return rc


def draw_table_cn(page, x0, y0, col_w, row_h, rows):
    for ri, row in enumerate(rows):
        for ci, cell in enumerate(row):
            cx = x0 + sum(col_w[:ci])
            page.insert_text((cx + 4, y0 + ri * row_h + row_h - 4.5), cell, fontsize=8.5, fontname="china-s")
    total_w = sum(col_w)
    for ri in range(len(rows) + 1):
        page.draw_line((x0, y0 + ri * row_h), (x0 + total_w, y0 + ri * row_h), width=0.6)
    xs = [x0]
    for w in col_w:
        xs.append(xs[-1] + w)
    for x in xs:
        page.draw_line((x, y0), (x, y0 + len(rows) * row_h), width=0.6)
    return y0 + len(rows) * row_h


# ----------------------------------------------------------------- sample E (booktabs table)
BT_ROWS = [["Model", "Accuracy (%)", None, "Time (s)"],
           ["", "Dev", "Test", ""],
           ["BERT-base", "91.2", "90.1", "12.4"],
           ["RoBERTa", "92.8", "91.7", "13.0"],
           ["Ours", "93.5", "92.9", "9.8"]]


def sample_booktabs(path):
    doc = fitz.open()
    doc.new_page(width=W, height=H)
    p = doc[0]
    p.insert_textbox(fitz.Rect(80, 80, 532, 100), "Table 2: Accuracy and runtime of three models.", fontsize=9, fontname="hebo")
    x = [100, 230, 300, 370, 450]
    y = 110
    p.draw_line((x[0], y), (x[-1] + 40, y), width=1.0)          # top rule
    p.insert_text((x[0], y + 12), "Model", fontsize=9, fontname="hebo")
    p.insert_text((x[1] + 25, y + 12), "Accuracy (%)", fontsize=9, fontname="hebo")
    p.insert_text((x[3], y + 12), "Time (s)", fontsize=9, fontname="hebo")
    p.draw_line((x[1], y + 16), (x[3] - 10, y + 16), width=0.5)  # cmidrule under the spanning header
    p.insert_text((x[1], y + 27), "Dev", fontsize=9, fontname="hebo")
    p.insert_text((x[2], y + 27), "Test", fontsize=9, fontname="hebo")
    p.draw_line((x[0], y + 31), (x[-1] + 40, y + 31), width=0.6)  # mid rule
    yy = y + 44
    for r in BT_ROWS[2:]:
        p.insert_text((x[0], yy), r[0], fontsize=9, fontname="helv")
        p.insert_text((x[1], yy), r[1], fontsize=9, fontname="helv")
        p.insert_text((x[2], yy), r[2], fontsize=9, fontname="helv")
        p.insert_text((x[3], yy), r[3], fontsize=9, fontname="helv")
        yy += 13
    p.draw_line((x[0], yy - 6), (x[-1] + 40, yy - 6), width=1.0)  # bottom rule
    p.insert_textbox(fitz.Rect(80, yy + 10, 532, yy + 60),
                     "The proposed model is the fastest of the three and has the highest test accuracy in this synthetic table.",
                     fontsize=9.5, fontname="helv")
    doc.save(path)
    return {"name": "booktabs_table", "pdf": os.path.basename(path), "reading_order": [], "split_sentences": [],
            "title": None, "headings": [], "captions": ["Table 2: Accuracy and runtime of three models."],
            "equations": [], "footnotes": [], "headers_footers": [], "page_numbers": [], "margin_notes": [],
            "tables": [{"caption_prefix": "Table 2", "rows": BT_ROWS,
                        "spans": [{"row": 0, "col": 1, "colspan": 2}]}],
            "n_references": 0}


# ----------------------------------------------------------------- C/D: scanned + encrypted
def sample_scanned(src, path, dpi=150):
    s = fitz.open(src)
    d = fitz.open()
    pix = s[0].get_pixmap(dpi=dpi, colorspace=fitz.csGRAY)
    pg = d.new_page(width=W, height=H)
    pg.insert_image(pg.rect, pixmap=pix)
    d.save(path)
    return {"name": "scanned_en", "pdf": os.path.basename(path), "expect_status": ["partial", "success"],
            "note": "image-only page 1 of two_column_en; needs OCR"}


def sample_encrypted(src, path):
    s = fitz.open(src)
    s.save(path, encryption=fitz.PDF_ENCRYPT_AES_256, user_pw="secret-user", owner_pw="secret-owner")
    return {"name": "encrypted_en", "pdf": os.path.basename(path), "expect_status": ["failed"], "expect_error": "encrypted"}


def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    gts = []
    a = os.path.join(out_dir, "two_column_en.pdf")
    gts.append(sample_a(a))
    gts.append(sample_b(os.path.join(out_dir, "single_column_zh.pdf")))
    gts.append(sample_booktabs(os.path.join(out_dir, "booktabs_table.pdf")))
    gts.append(sample_scanned(a, os.path.join(out_dir, "scanned_en.pdf")))
    gts.append(sample_encrypted(a, os.path.join(out_dir, "encrypted_en.pdf")))
    with open(os.path.join(out_dir, "ground_truth.json"), "w", encoding="utf-8") as fh:
        json.dump(gts, fh, ensure_ascii=False, indent=1)
    print(f"wrote {len(gts)} samples to {out_dir}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "samples")
