"""Hard checks shared by the v3 and later figure scripts: white background, 3:1 contrast of every
colour used against white, 7 pt minimum on every text artist, exact 6.5 in page width, pdf.fonttype 42."""
import matplotlib as mpl
from matplotlib.colors import to_rgb
from matplotlib.text import Text


def lum(c):
    r, g, b = [x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4 for x in to_rgb(c)]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast_on_white(c):
    return 1.05 / (lum(c) + 0.05)


def final_asserts(fig, colors, pdf_path=None):
    """colors: dict name -> hex of every mark/line/text colour drawn. Returns a short report string."""
    lines = []
    for name, c in colors.items():
        cr = contrast_on_white(c)
        lines.append(f"  contrast {name} {c}: {cr:.2f}:1")
        assert cr >= 3.0, (name, c, cr)
    small = [(t.get_text(), t.get_fontsize()) for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()
             and t.get_fontsize() < 7]
    assert not small, small
    sizes = sorted({t.get_fontsize() for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()})
    lines.append(f"  font sizes in use (pt): {sizes}; minimum {sizes[0]}")
    fb = fig.bbox
    for leg in fig.legends:
        e = leg.get_window_extent(fig.canvas.get_renderer())
        assert e.x0 >= fb.x0 and e.y0 >= fb.y0 and e.x1 <= fb.x1 and e.y1 <= fb.y1, ("legend outside figure", e, fb)
    for t in fig.findobj(Text):
        if t.get_visible() and t.get_text().strip():
            e = t.get_window_extent(fig.canvas.get_renderer())
            assert e.x0 >= fb.x0 - 1 and e.y0 >= fb.y0 - 1 and e.x1 <= fb.x1 + 1 and e.y1 <= fb.y1 + 1, ("text outside figure", t.get_text())
    assert tuple(fig.get_size_inches())[0] == 6.5
    assert mpl.rcParams["pdf.fonttype"] == 42
    assert to_rgb(fig.get_facecolor()) == (1.0, 1.0, 1.0)
    lines.append(f"  page width 6.5 in, pdf.fonttype 42, white figure background, height {fig.get_size_inches()[1]} in")
    return "\n".join(lines)
