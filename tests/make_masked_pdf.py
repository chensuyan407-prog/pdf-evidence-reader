#!/usr/bin/env python3
"""Make the occluded BERT control: page 1 of the scanned BERT PDF with the
abstract body (where both answers are) painted white. Everything else unchanged."""
import os
import sys

try:
    import pymupdf as fitz
except ImportError:
    import fitz

HERE = os.path.dirname(os.path.abspath(__file__))
MASK = (85, 240, 280, 552)  # PDF points, page 1: abstract body in the left column


def main(src=os.path.join(HERE, "pdfs", "bert.pdf"), dst=os.path.join(HERE, "pdfs", "bert_masked.pdf")):
    d = fitz.open(src)
    d[0].draw_rect(fitz.Rect(MASK), color=(1, 1, 1), fill=(1, 1, 1), overlay=True)
    d.save(dst)
    print("wrote", dst)


if __name__ == "__main__":
    main(*sys.argv[1:])
