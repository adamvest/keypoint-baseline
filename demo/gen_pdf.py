import os
from fpdf import FPDF
from PIL import Image

for root, _, fnames in os.walk("./demo/rec_segments_visualizations"):
    fnames.sort(key=lambda x: int(x.split("_")[0]))
    limit = 100
    max_w, max_h = 0, 0

    for i in range(limit):
        img = Image.open(os.path.join(root, fnames[i]))
        w, h = img.size

        if w > max_w:
            max_w = w
        if h > max_h:
            max_h = h

    pdf = FPDF(unit="pt", format=[max_w, max_h])

    for fname in fnames:
        if limit == 88 or limit == 70:
            limit -=1
            continue

        pdf.add_page()
        pdf.image(os.path.join(root, fname), 0, 0)
        limit -= 1

pdf.output("./demo/rec_keypoints_results.pdf", "F")
