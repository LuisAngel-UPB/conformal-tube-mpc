"""Convert a '# %%'-cell Python script into a Jupyter notebook: python3 py2nb.py src.py out.ipynb"""
import sys, re, nbformat

src, dst = sys.argv[1], sys.argv[2]
text = open(src, encoding="utf-8").read()
cells = []
for block in re.split(r"^# %%", text, flags=re.M)[1:]:
    head, _, body = block.partition("\n")
    if head.strip() == "[markdown]":
        md = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in body.strip("\n").splitlines())
        cells.append(nbformat.v4.new_markdown_cell(md))
    else:
        cells.append(nbformat.v4.new_code_cell(body.strip("\n")))
nb = nbformat.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbformat.write(nb, dst)
print(f"{dst}: {len(cells)} cells")
