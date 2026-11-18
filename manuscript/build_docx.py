"""Build the manuscript .docx in the Bilisim Teknolojileri Dergisi (Gazi BTD) layout.

Input : manuscript/source.md (light markup, {{macro}} placeholders, [@key] citations)
        manuscript/macros.json (from code/make_tables.py), tables/*.json, figures/*.png
        literature/refs_crossref.json + manuscript/refs_extra.json (non-DOI items)
Output: manuscript/main.docx (blind) and manuscript/main_authors.docx

Layout (from the journal's guide, journal-file/10109): A4, margins 2/2/2/2.5 cm,
Times New Roman; front matter single column, body in two 8.25 cm columns with
0.5 cm gap; body 10 pt, references 8 pt; titles 20 pt; all headings, table and
figure captions carry an English translation in parentheses (8 pt).
"""
import datetime
import json
import re
import sys
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FONT = "Times New Roman"

AUTHOR = {
    "name": "Osman Can ÇETLENBİK",
    "addr": "Büyük Veri Analistliği Programı, İstatistik Bölümü, Manisa Teknik Bilimler "
            "Meslek Yüksekokulu, Manisa Celal Bayar Üniversitesi, 45140, Manisa, Türkiye",
    "mail": "osman.cetlenbik@cbu.edu.tr",
    "orcid": "ORCID: 0009-0000-6561-1943",
}


# ------------------------------------------------------------- helpers
def set_cols(section, num, space_twips=284):
    sp = section._sectPr
    for c in sp.findall(qn("w:cols")):
        sp.remove(c)
    cols = OxmlElement("w:cols")
    cols.set(qn("w:num"), str(num))
    cols.set(qn("w:space"), str(space_twips))
    sp.append(cols)


def page_setup(section):
    section.page_width, section.page_height = Cm(21.0), Cm(29.7)
    section.top_margin = section.left_margin = section.right_margin = Cm(2.0)
    section.bottom_margin = Cm(2.5)


def fmt(p, size=10, align=None, before=0, after=0, line=1.0, indent=None, hanging=None):
    pf = p.paragraph_format
    pf.space_before, pf.space_after, pf.line_spacing = Pt(before), Pt(after), line
    if align is not None:
        p.alignment = align
    if indent is not None:
        pf.first_line_indent = Cm(indent)
    if hanging is not None:
        pf.left_indent, pf.first_line_indent = Cm(hanging), Cm(-hanging)
    for r in p.runs:
        style_run(r, size)
    return p


def style_run(r, size, bold=None, italic=None):
    r.font.name = FONT
    r._element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
    r.font.size = Pt(size)
    r.font.color.rgb = RGBColor(0, 0, 0)
    if bold is not None:
        r.bold = bold
    if italic is not None:
        r.italic = italic


INLINE = re.compile(r"(\*\*[^*]+\*\*|\*[^*]+\*|\^[^^]+\^|~[^~]+~|`[^`]+`)")


def add_rich(p, text, size=10, italic=False, bold=False):
    """**bold**, *italic*, ^superscript^, ~subscript~."""
    for part in INLINE.split(text):
        if not part:
            continue
        b, i, sup, sub = bold, italic, False, False
        if part.startswith("**"):
            part, b = part[2:-2], True
        elif part.startswith("*"):
            part, i = part[1:-1], not italic
        elif part.startswith("^"):
            part, sup = part[1:-1], True
        elif part.startswith("~"):
            part, sub = part[1:-1], True
        mono = part.startswith("`") and part.endswith("`") and len(part) > 1
        if mono:
            part = part[1:-1]
        r = p.add_run(part)
        style_run(r, size, bold=b, italic=i)
        if mono:
            r.font.name = "Courier New"
            r._element.rPr.rFonts.set(qn("w:ascii"), "Courier New")
            r._element.rPr.rFonts.set(qn("w:hAnsi"), "Courier New")
        r.font.superscript, r.font.subscript = sup, sub
    return p


# ------------------------------------------------------------- citations
class Cites:
    def __init__(self, refs):
        self.refs, self.order = refs, []

    def num(self, key):
        if key not in self.refs:
            raise KeyError(f"unknown citation key {key}")
        if key not in self.order:
            self.order.append(key)
        return self.order.index(key) + 1

    def sub(self, text):
        def repl(m):
            nums = sorted(self.num(k.strip().lstrip("@")) for k in m.group(1).split(";"))
            # compress runs: 3,4,5 -> 3-5
            out, i = [], 0
            while i < len(nums):
                j = i
                while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
                    j += 1
                out.append(f"{nums[i]}-{nums[j]}" if j - i >= 2 else
                           ", ".join(map(str, nums[i:j + 1])))
                i = j + 1
            return "[" + ", ".join(out) + "]"
        return re.sub(r"\[(@[^\]]+)\]", repl, text)


def initials(given):
    parts = re.split(r"[\s-]+", given.strip())
    return " ".join(p[0] + "." for p in parts if p)


def format_ref(item):
    """Journal style: A. Surname, B. Surname, "Title", Venue, vol(issue), pages, year. doi"""
    if "text" in item:
        return item["text"]
    m = item["meta"]
    auth = []
    for a in (m.get("author") or []):
        if "family" in a:
            auth.append((initials(a.get("given", "")) + " " + a["family"]).strip())
        elif "name" in a:
            auth.append(a["name"])
    if len(auth) > 6:
        auth = auth[:6] + ["et al."]
    title = re.sub(r"\s+", " ", m["title"][0]).strip()
    venue = (m.get("container-title") or [""])[-1]
    if not venue and m.get("event"):
        venue = m["event"].get("name", "")
    if not venue:
        venue = m.get("publisher", "")
    year = (m.get("published") or m.get("issued"))["date-parts"][0][0]
    vol = m.get("volume")
    vi = (vol or "") + (f"({m['issue']})" if m.get("issue") else "")
    pages = m.get("page") or m.get("article-number")
    bits = [", ".join(auth), f"“{title}”", f"*{venue}*"]
    if vi:
        bits.append(vi)
    if pages:
        bits.append(str(pages).replace("-", "–"))
    bits.append(str(year))
    return ", ".join(b for b in bits if b) + f". doi:{m['DOI'].lower()}"


# ------------------------------------------------------------- tables & figures
def caption(doc, kind, num, tr, en, after_fig=False):
    p = doc.add_paragraph()
    add_rich(p, f"**{kind} {num}.** {tr}", 9)
    fmt(p, 9, WD_ALIGN_PARAGRAPH.CENTER if after_fig else WD_ALIGN_PARAGRAPH.LEFT,
        before=6 if after_fig else 9, after=0)
    p2 = doc.add_paragraph()
    add_rich(p2, f"({en})", 8)
    fmt(p2, 8, WD_ALIGN_PARAGRAPH.CENTER if after_fig else WD_ALIGN_PARAGRAPH.LEFT,
        after=9 if after_fig else 3)
    if not after_fig:  # a table caption stays with its table
        p.paragraph_format.keep_with_next = True
        p2.paragraph_format.keep_with_next = True
    else:  # the two caption lines of a figure stay together
        p.paragraph_format.keep_with_next = True


def add_table(doc, spec, num, macros, cites):
    caption(doc, "Tablo", num, spec["caption_tr"], spec["caption_en"])
    rows = [spec["header"]] + spec["rows"]
    t = doc.add_table(rows=len(rows), cols=len(spec["header"]))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.style = doc.styles["Table Grid"] if "Table Grid" in [s.name for s in doc.styles] else None
    fs = spec.get("font", 8)
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            cell = t.cell(i, j)
            cell.text = ""
            p = cell.paragraphs[0]
            add_rich(p, cites.sub(str(val)), fs, bold=(i == 0))
            fmt(p, fs, WD_ALIGN_PARAGRAPH.LEFT if j == 0 else WD_ALIGN_PARAGRAPH.CENTER)
            if i < len(rows) - 1:  # keep the table on one page where it fits
                p.paragraph_format.keep_with_next = True
    # header row repeats on each page; rows are not split across pages
    for i, r in enumerate(t.rows):
        trPr = r._tr.get_or_add_trPr()
        cs = OxmlElement("w:cantSplit"); trPr.append(cs)
        if i == 0:
            h = OxmlElement("w:tblHeader"); trPr.append(h)
    # thin borders on all cells
    tblPr = t._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        e = OxmlElement(f"w:{edge}")
        e.set(qn("w:val"), "single"); e.set(qn("w:sz"), "4"); e.set(qn("w:color"), "000000")
        borders.append(e)
    tblPr.append(borders)
    w = OxmlElement("w:tblW"); w.set(qn("w:w"), "5000"); w.set(qn("w:type"), "pct")
    for old in tblPr.findall(qn("w:tblW")):
        tblPr.remove(old)
    tblPr.append(w)
    if spec.get("note"):
        p = doc.add_paragraph()
        add_rich(p, cites.sub(spec["note"]), 8)
        fmt(p, 8, WD_ALIGN_PARAGRAPH.JUSTIFY, before=2, after=6)
    else:  # a small gap between the table and the following text
        p = doc.add_paragraph()
        fmt(p, 4, WD_ALIGN_PARAGRAPH.LEFT, before=0, after=4)


def clean_png(path):
    """Image bytes with only the chunks needed to draw the picture (no metadata)."""
    import io, struct
    d = open(path, "rb").read()
    if not d.startswith(b"\x89PNG"):
        return io.BytesIO(d)
    keep = {b"IHDR", b"PLTE", b"tRNS", b"pHYs", b"IDAT", b"IEND", b"gAMA", b"sRGB"}
    out, i = [d[:8]], 8
    while i < len(d):
        n = struct.unpack(">I", d[i:i + 4])[0]
        if d[i + 4:i + 8] in keep:
            out.append(d[i:i + 12 + n])
        i += 12 + n
    return io.BytesIO(b"".join(out))


def add_figure(doc, path, num, tr, en, width_cm):
    p = doc.add_paragraph()
    p.add_run().add_picture(clean_png(path), width=Cm(width_cm))
    fmt(p, 10, WD_ALIGN_PARAGRAPH.CENTER, before=6)
    p.paragraph_format.keep_with_next = True  # figure stays with its caption
    caption(doc, "Şekil", num, tr, en, after_fig=True)


# ------------------------------------------------------------- main
DRAFT = "--draft" in sys.argv


def build(with_authors, out):
    macros = json.load(open(HERE / "macros.json"))
    refs = {k: v for k, v in json.load(open(ROOT / "literature" / "refs_crossref.json")).items()}
    extra = HERE / "refs_extra.json"
    if extra.exists():
        refs.update(json.load(open(extra)))
    src = open(HERE / "source.md", encoding="utf-8").read()
    # body is kept in parts; "BODY" in source.md expands to them in order
    parts = ("body_intro_method.md", "body_results.md")
    src = src.replace("\nBODY", "\n" + "\n\n".join(open(HERE / p, encoding="utf-8").read() for p in parts))

    def mac(m):
        k = m.group(1)
        if k not in macros:
            if DRAFT:
                return "??"
            raise KeyError(f"missing macro {k}")
        return str(macros[k])
    src = re.sub(r"\{\{([A-Za-z0-9_]+)\}\}", mac, src)
    src = src.replace("= <", "< ")  # "p = <0,001" -> "p < 0,001"
    front, body = src.split("\n=====\n", 1)
    fm = dict(re.findall(r"^(\w+):\s*(.*)$", front, re.M))

    cites = Cites(refs)
    doc = Document(str(HERE / "btd_template.docx"))
    body_el = doc.element.body
    for el in list(body_el):
        if el.tag != qn("w:sectPr"):
            body_el.remove(el)
    sec = doc.sections[0]
    page_setup(sec)
    set_cols(sec, 1, 708)

    # --- front matter (single column)
    p = doc.add_paragraph(); add_rich(p, fm["title_tr"], 20)
    fmt(p, 20, WD_ALIGN_PARAGRAPH.CENTER, before=24, after=12)
    if with_authors:
        p = doc.add_paragraph(); add_rich(p, AUTHOR["name"] + "^*^", 11)
        fmt(p, 11, WD_ALIGN_PARAGRAPH.CENTER, after=6)
        for line in (AUTHOR["addr"], AUTHOR["mail"], AUTHOR["orcid"]):
            p = doc.add_paragraph(); add_rich(p, line, 8)
            fmt(p, 8, WD_ALIGN_PARAGRAPH.CENTER)
        doc.add_paragraph()
    else:
        p = doc.add_paragraph()  # blind copy: no author block and no note
        fmt(p, 8, WD_ALIGN_PARAGRAPH.CENTER, after=6)
    for lab, key, kwl, kwkey in (("Özet", "abstract_tr", "Anahtar Kelimeler", "keywords_tr"),):
        p = doc.add_paragraph(); add_rich(p, f"**{lab}—** ", 10)
        add_rich(p, cites.sub(fm[key]), 10)
        fmt(p, 10, WD_ALIGN_PARAGRAPH.JUSTIFY, after=6)
        p = doc.add_paragraph(); add_rich(p, f"**{kwl}—** ", 10); add_rich(p, fm[kwkey], 10)
        fmt(p, 10, WD_ALIGN_PARAGRAPH.JUSTIFY, after=18)
    p = doc.add_paragraph(); add_rich(p, fm["title_en"], 20)
    fmt(p, 20, WD_ALIGN_PARAGRAPH.CENTER, before=12, after=12)
    p = doc.add_paragraph(); add_rich(p, "**Abstract—** ", 10); add_rich(p, cites.sub(fm["abstract_en"]), 10)
    fmt(p, 10, WD_ALIGN_PARAGRAPH.JUSTIFY, after=6)
    p = doc.add_paragraph(); add_rich(p, "**Keywords—** ", 10); add_rich(p, fm["keywords_en"], 10)
    fmt(p, 10, WD_ALIGN_PARAGRAPH.JUSTIFY, after=12)

    def new_section(cols):
        s = doc.add_section(WD_SECTION.CONTINUOUS)
        page_setup(s)
        set_cols(s, cols, 284 if cols == 2 else 708)
        return s

    new_section(2)
    tnum = fnum = 0
    state = {"wide": False}

    def to_wide():
        if not state["wide"]:
            new_section(1)
            state["wide"] = True

    for block in re.split(r"\n\s*\n", body.strip()):
        block = block.strip()
        if not block:
            continue
        is_wide = (block.startswith("TABLE:") and ("wide" in [o.strip() for o in block[6:].split("|")[1:]]
                   or json.load(open(HERE / "tables" / f"{block[6:].split('|')[0].strip()}.json")).get("wide"))) \
            or (block.startswith("FIGURE:") and "wide" in [x.strip() for x in block[7:].split("|")][3:])
        if state["wide"] and not is_wide:
            new_section(2)  # back to two columns once a run of wide blocks ends
            state["wide"] = False
        if block.startswith("# "):
            tr, en = re.match(r"# (.*?)\s*\((.*)\)\s*$", block).groups()
            p = doc.add_paragraph(); add_rich(p, f"**{tr}** ", 10); add_rich(p, f"**({en})**", 8)
            fmt(p, 10, WD_ALIGN_PARAGRAPH.LEFT, before=10, after=4)
            p.paragraph_format.keep_with_next = True
        elif block.startswith("## "):
            tr, en = re.match(r"## (.*?)\s*\((.*)\)\s*$", block).groups()
            p = doc.add_paragraph(); add_rich(p, f"*{tr}* ", 10); add_rich(p, f"*({en})*", 8)
            fmt(p, 10, WD_ALIGN_PARAGRAPH.LEFT, before=8, after=3)
            p.paragraph_format.keep_with_next = True
        elif block.startswith("TABLE:"):
            name, *opt = block[6:].split("|")
            spec = json.load(open(HERE / "tables" / f"{name.strip()}.json"))
            wide = "wide" in [o.strip() for o in opt] or spec.get("wide")
            if wide:
                to_wide()
            tnum += 1
            add_table(doc, spec, tnum, macros, cites)
        elif block.startswith("FIGURE:"):
            path, tr, en, *opt = [x.strip() for x in block[7:].split("|")]
            wide = "wide" in opt
            if wide:
                to_wide()
            fnum += 1
            add_figure(doc, HERE / path, fnum, cites.sub(tr), en, 16.5 if wide else 8.0)
        elif block.startswith("EQ:"):
            eq, n = block[3:].rsplit("|", 1)
            p = doc.add_paragraph(); add_rich(p, eq.strip(), 10)
            p.add_run("\t(" + n.strip() + ")")
            fmt(p, 10, WD_ALIGN_PARAGRAPH.LEFT, before=4, after=4)
            p.paragraph_format.tab_stops.add_tab_stop(Cm(8.0), 2)  # right tab
        elif block.startswith("- "):
            for item in block.split("\n- "):
                p = doc.add_paragraph(); add_rich(p, "• " + cites.sub(item.lstrip("- ").strip()), 10)
                fmt(p, 10, WD_ALIGN_PARAGRAPH.JUSTIFY, after=2, hanging=0.4)
        elif block == "REFS":
            p = doc.add_paragraph(); add_rich(p, "**KAYNAKLAR** ", 10); add_rich(p, "**(REFERENCES)**", 8)
            fmt(p, 10, WD_ALIGN_PARAGRAPH.LEFT, before=10, after=4)
            p.paragraph_format.keep_with_next = True
            for i, k in enumerate(cites.order, 1):
                p = doc.add_paragraph(); add_rich(p, f"[{i}] " + format_ref(refs[k]), 8)
                fmt(p, 8, WD_ALIGN_PARAGRAPH.JUSTIFY, after=1, hanging=0.6)
            new_section(2)  # continuous break: balances the reference columns
        else:
            p = doc.add_paragraph(); add_rich(p, cites.sub(" ".join(block.split("\n"))), 10)
            fmt(p, 10, WD_ALIGN_PARAGRAPH.JUSTIFY, after=4, indent=0.5)
    # core properties
    cp = doc.core_properties
    cp.author = "Osman Can Çetlenbik" if with_authors else ""
    cp.last_modified_by = cp.author
    cp.title = fm["title_tr"]
    cp.comments = ""
    now = datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0, tzinfo=None)
    cp.created = cp.modified = now
    cp.last_printed = now
    cp.revision = 1
    # app.xml from the template carries stale page/word counts and editing time
    for part in doc.part.package.iter_parts():
        if str(part.partname) == "/docProps/app.xml":
            part._blob = (b'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
                          b'<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties">'
                          b'<Application>Microsoft Office Word</Application></Properties>')
    doc.save(out)
    json.dump(cites.order, open(HERE / "cited_keys.json", "w"), indent=0)
    unused = sorted(set(refs) - set(cites.order))
    print(f"{out.name}: tables {tnum}, figures {fnum}, refs cited {len(cites.order)}, unused {unused}")


if __name__ == "__main__":
    build(False, HERE / "main.docx")
    build(True, HERE / "main_authors.docx")
