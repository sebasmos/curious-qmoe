#!/bin/bash
# Builds the anonymous supplementary code archive from a git ref of this repository.
#   bash scripts/make_anonymous_archive.sh [REF] [OUT_ZIP]      (defaults: HEAD, ./anonymous-code.zip)
# Steps: git archive of REF; remove files that only make sense with names attached; rename the package directory
# QWave to qmoe_lib (archive only; the repository keeps QWave); rewrite every text file (author names, affiliations,
# emails, GitHub URLs of the authors, the repository and handle names, arXiv links, absolute home paths including
# Windows ones, the private source path in the plans, the names of the private project documents, the results root
# as RESULTS_DIR, GPU lock tags, the package name in imports and setup metadata); LICENSE copyright line to
# "Anonymous authors"; check that every .py still compiles and every .sh still parses; remove any .git; zip; unzip the
# zip into a fresh folder and grep it for every identifying string. Exits non-zero, and deletes the zip, if anything identifying is left.
# Needs git, python3, zip, unzip.
set -euo pipefail
REPO=$(git rev-parse --show-toplevel)
REF=${1:-HEAD}
OUT=${2:-anonymous-code.zip}
case $OUT in /*) ;; *) OUT=$PWD/$OUT ;; esac
PY=${PY:-python3}
WORK=$(mktemp -d); trap 'rm -rf "$WORK"' EXIT
TOP=code

COMMIT=$(git -C "$REPO" rev-parse --short "$REF")
echo "exporting $REF ($COMMIT)"
mkdir -p "$WORK/src"
git -C "$REPO" archive --format=tar --prefix="$TOP/" "$REF" | tar -x -C "$WORK/src"
D=$WORK/src/$TOP

# Files that exist to name the authors or the public repository.
rm -f "$D/CITATION.cff" "$D/CITATION.CFF" "$D/CONTRIBUTING.md" "$D/.gitignore" "$D/.gitattributes"
rm -rf "$D/.github"
rm -f "$D/scripts/make_anonymous_archive.sh"   # this script lists the names it removes
find "$D" -name .git -prune -exec rm -rf {} +
find "$D" -name '*.ipynb' -delete

# Package name: QWave matches a public repository, so the archive ships it as qmoe_lib. Directories first (deepest
# first), then the text rewrite below changes the imports, paths and setup metadata.
PKG=qmoe_lib
find "$D" -depth -type d -name QWave | while read -r d; do mv "$d" "$(dirname "$d")/$PKG"; done

# Rewrite every text file.
"$PY" -I - "$D" <<'PYEOF'
import re, sys
from pathlib import Path

root = Path(sys.argv[1])
ALLOWED_GITHUB = {"fschmid56", "karolpiczak", "karoldvl", "qiuqiangkong", "openai", "google-research",
                  "peterliht", "jmlrorg", "huggingface", "pytorch", "facebookresearch", "microsoft"}
ANON_URL = "https://anonymous.invalid/removed"
names = [  # longest first; every author of the paper, with and without accents, and their handles
    r"Sebasti[aá]n Andr[eé]s Cajas[ -]Ord[oó][nñ]ez", r"Sebasti[aá]n Cajas[ -]Ord[oó][nñ]ez", r"Sebasti[aá]n Cajas",
    r"Cajas[ -]Ord[oó][nñ]ez", r"Luis Fernando Torres Torres", r"Luis Torres", r"Mackenzie J\. Meni", r"Mackenzie Meni",
    r"Carlos Andr[eé]s Dur[aá]n Paredes", r"Carlos Dur[aá]n", r"Eric Arazo", r"Cristian Bosch", r"Ricardo Simon Carbajo",
    r"Yuan Lai", r"Leo Anthony Celi", r"Leo Celi",
    r"cajas\.sebastian", r"sebasmos", r"ltorres", r"lceli", r"laiy",
    r"Sebasti[aá]n", r"Cajas", r"Ord[oó][nñ]ez", r"Mackenzie", r"Dur[aá]n Paredes", r"Arazo", r"Carbajo",
]
names_cs = [r"\bMeni\b", r"\bCeli\b", r"\bTorres\b", r"\bParedes\b"]  # case-sensitive surnames
affils = [r"MIT Critical Data", r"Critical ?Data", r"Massachusetts Institute of Technology", r"Universit[eé] de Rennes",
          r"univ-rennes\.fr", r"Technetium Engineering", r"Technetium", r"Instituci[oó]n Universitaria Colegio Mayor del Cauca",
          r"Colegio Mayor del Cauca", r"unimayor\.edu\.co", r"unicauca", r"CeADAR", r"University College Dublin", r"ucd\.ie",
          r"Tsinghua University", r"Tsinghua", r"Beth Israel Deaconess Medical Center", r"Beth Israel",
          r"Harvard T\.H\. Chan School of Public Health", r"Harvard", r"mit\.edu"]
private = [r"AnpassenNN/mit-projects/36-qmoe", r"mit-projects/36-qmoe", r"AnpassenNN", r"mit-projects", r"QuantumVE"]  # repo names

def home(ext):
    return {".sh": "$HOME", ".bash": "$HOME", ".template": "$HOME", ".yaml": "${oc.env:HOME}",
            ".yml": "${oc.env:HOME}"}.get(ext, "~")

def github(m):
    owner = m.group(1).lower()
    return m.group(0) if owner in ALLOWED_GITHUB else ANON_URL

RD_PY = 'Path(os.environ.get("RESULTS_DIR", str(Path.home() / "Documents/results")))'
RD_SH = "${RESULTS_DIR:-$HOME/Documents/results}"

def results_dir(text, ext):
    """The results root (one folder per run) becomes the RESULTS_DIR environment variable."""
    if ext == ".py":
        sub = lambda m: f' / "{m.group("s").strip("/")}"' if m.group("s") and m.group("s").strip("/") else ""
        # os.path.expanduser("~/Documents/qmoe-results[/sub]") -> a string path, trailing slash kept
        new = re.sub(r"os\.path\.expanduser\((?P<q>['\"])~/Documents/qmoe-results(?P<s>/[^'\"]*)?(?P=q)\)",
                     lambda m: 'os.path.join(os.environ.get("RESULTS_DIR", os.path.expanduser("~/Documents/results")), '
                               f'"{(m.group("s") or "").lstrip("/")}")', text)
        # Path("~/Documents/qmoe-results[/sub]") (an absolute home path after the home rewrite) and
        # Path.home() / "Documents/qmoe-results[/sub]", HOME / "..." -> a Path
        new = re.sub(r"Path\((?P<q>['\"])~/Documents/qmoe-results(?P<s>/[^'\"]*)?(?P=q)\)", lambda m: RD_PY + sub(m), new)
        new = re.sub(r'(?:Path\.home\(\)|\bHOME) / "Documents/qmoe-results(?P<s>/[^"]*)?"', lambda m: RD_PY + sub(m), new)
        if new != text and not re.search(r"^import os\b|^import [^\n]*\bos\b", new, re.M):
            m = re.search(r"^(?:import|from)\s(?!__future__)", new, re.M)
            new = new[:m.start()] + "import os\n" + new[m.start():] if m else "import os\n" + new
        text = re.sub(r"~/Documents/qmoe-results", "$RESULTS_DIR", new)          # comments and docstrings
    elif ext in (".sh", ".bash", ".template"):
        text = re.sub(r"(?:\$HOME|\$\{HOME\}|~)/Documents/qmoe-results", lambda m: RD_SH, text)
    else:
        text = re.sub(r"(?:\$HOME|\$\{HOME\}|~)/Documents/qmoe-results", "$RESULTS_DIR", text)
    return text

def scrub(text, ext):
    text = re.sub(r"(?im)^(\s*Copyright \(c\) \d{4}).*$", r"\1 Anonymous authors", text)
    text = re.sub(r"/private/tmp/claude-[^\s\"')]*", "/tmp/scratch", text)
    # the plans name the private file the round's design came from; that file is not distributed
    text = re.sub(r"`?(?:AnpassenNN/)?mit-projects/36-qmoe/blindspot-reviews/[^\s`)]+`?(?: in AnpassenNN(?= \())?",
                  "the round's plan (not distributed)", text)
    text = re.sub(r" of the AnpassenNN repo\b", "", text)
    text = re.sub(r"[A-Za-z]:\\Users\\[^\x22\x27\n]*?QWave-dev\\", "", text)          # legacy Windows defaults
    text = re.sub(r"[A-Za-z]:\\Users\\[^\\\"']+", "~", text)
    text = re.sub(r"(?<![\w.])(?:/Users|/home)/[A-Za-z0-9._-]+", lambda m: home(ext), text)
    text = results_dir(text, ext)
    if ext == ".py":                                                            # pathlib does not expand "~"
        new = re.sub(r'''Path\((?P<q>['"])~/(?P<s>[^'"]*)(?P=q)\)''', lambda m: f'(Path.home() / "{m.group("s")}")', text)
        new = re.sub(r'''(?m)^(?P<i>\s*\w+ = )(?P<q>['"])(?P<s>~/[^'"]*)(?P=q)''', lambda m: f'{m.group("i")}os.path.expanduser("{m.group("s")}")', new)
        new = re.sub(r'''(os\.environ\.get\(["'][A-Z_0-9]+["'], )(?P<q>['"])(?P<s>~/[^'"]*)(?P=q)\)''', lambda m: f'{m.group(1)}os.path.expanduser("{m.group("s")}"))', new)
        if new != text and not re.search(r"^import os\b|^import [^\n]*\bos\b", new, re.M):
            m = re.search(r"^(?:import|from)\s(?!__future__)", new, re.M)
            new = new[:m.start()] + "import os\n" + new[m.start():] if m else "import os\n" + new
        text = new
    text = re.sub(r"\b(?:claude|agent)-t(\d+)", r"run-t\1", text)                # GPU lock tags
    text = re.sub(r"QWave", "qmoe_lib", text)
    text = re.sub(r"\bqwave\b", "qmoe_lib", text, flags=re.I)
    text = re.sub(r"curious[-_]moe[-_A-Za-z]*", "the draft", text, flags=re.I)  # the draft's project name
    text = re.sub(r"https?://(?:www\.)?arxiv\.org/[^\s)\"'>\]]*", ANON_URL, text, flags=re.I)
    text = re.sub(r"\b(?:arXiv:?\s*)?2511\.11743(?:v\d+)?", "[preprint removed]", text, flags=re.I)
    text = re.sub(r"(?:https?://)?(?:www\.)?github\.com/([A-Za-z0-9_.-]+)(?:/[^\s)\"'>\]`]*)?", github, text, flags=re.I)
    text = re.sub(r"(?:https?://)?([A-Za-z0-9-]+)\.github\.io[^\s)\"'>\]]*", github, text, flags=re.I)
    text = re.sub(r"(?:https?://)?raw\.githubusercontent\.com/([A-Za-z0-9_.-]+)[^\s)\"'>\]]*", github, text, flags=re.I)
    text = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}",
                  lambda m: m.group(0) if m.group(0) == "anonymous@example.com" else "anonymous@example.com", text)
    text = re.sub(r"curious[-_]qmoe", "qmoe", text, flags=re.I)
    for a in private:
        text = re.sub(a, "private-notes", text, flags=re.I)
    for a in affils:
        text = re.sub(a, "Anonymous institution", text, flags=re.I)
    for n in names:
        text = re.sub(n, "Anonymous", text, flags=re.I)
    for n in names_cs:
        text = re.sub(n, "Anonymous", text)
    return internal_docs(text)

def internal_docs(text):
    """The plans and two T0 scripts name the authors' private project documents; neutral wording instead."""
    text = re.sub(r"(?m)^\| Documents \|.*\n", "", text)    # the T21 to T26 plans: which documents an agent may not edit
    text = re.sub(r"\bBACKLOG -(\d+)((?:, -\d+)*)",
                  lambda m: ("project log entries -" if m.group(2) else "project log entry -") + m.group(1) + m.group(2), text)
    text = re.sub(r"\bthe(\s+)BACKLOG\b", r"the\1project log", text)
    text = re.sub(r"\bBACKLOG\b", "the project log", text)
    text = text.replace("Nothing under Overleaf, the private-notes docs or any paper folder",
                        "Nothing in the draft, the project notes or any paper folder")
    text = text.replace("The Table 4 text of `index.md`", "The Table 4 text of the project notes")
    text = text.replace("The index.md text calls", "The project notes call")
    text = re.sub(r"\bOverleaf\b", "the draft", text, flags=re.I)
    return text

changed = 0
for p in sorted(root.rglob("*")):
    if not p.is_file() or p.is_symlink():
        continue
    raw = p.read_bytes()
    if b"\0" in raw:
        continue
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        continue
    new = scrub(text, p.suffix.lower())
    if new != text:
        p.write_text(new, encoding="utf-8")
        changed += 1
print(f"rewrote {changed} files")
PYEOF

# LICENSE: the copyright line names the authors.
if [ -f "$D/LICENSE" ]; then
  sed -i.bak -E 's/^(Copyright \(c\) [0-9]{4}).*$/\1 Anonymous authors/' "$D/LICENSE" && rm -f "$D/LICENSE.bak"
  grep -n '^Copyright' "$D/LICENSE"
fi

# The rewrite must not break code: every .py compiles (in memory, nothing written) and every .sh parses.
"$PY" -I - "$D" <<'PYEOF2'
import sys
from pathlib import Path
bad = 0
for p in sorted(Path(sys.argv[1]).rglob("*.py")):
    try:
        compile(p.read_text(encoding="utf-8"), str(p), "exec")
    except Exception as e:
        print(f"does not compile: {p}: {e}"); bad += 1
sys.exit(1 if bad else 0)
PYEOF2
while IFS= read -r f; do bash -n "$f" || { echo "does not parse: $f"; exit 1; }; done < <(find "$D" -name '*.sh')
echo "syntax ok (.py compile, .sh parse)"

rm -f "$OUT"
(cd "$WORK/src" && zip -q -r -X "$OUT" "$TOP")
echo "wrote $OUT ($(du -h "$OUT" | cut -f1)), $(unzip -Z1 "$OUT" | grep -vc '/$') files"

# Leak check on the unpacked zip.
CHK=$WORK/check; mkdir -p "$CHK"; unzip -q "$OUT" -d "$CHK"
fail=0
check() {  # label, grep flags, pattern
  local hits; hits=$(grep -r -n -I $2 -E "$3" "$CHK" | sed "s|$CHK/||" || true)
  if [ -n "$hits" ]; then echo "LEAK [$1]"; echo "$hits" | head -20; fail=1; else echo "clean [$1]"; fi
}
check "author names and handles" -i 'sebasmos|cadur|cajas|sebasti[aá]n|ord[oó][nñ]ez|ltorres|luis fernando|torres torres|mackenzie|dur[aá]n paredes|carlos dur[aá]n|arazo|cristian bosch|carbajo|yuan lai|laiy|leo anthony|lceli'
check "surnames, case-sensitive" '' '\b(Meni|Celi|Torres|Paredes)\b'
check "affiliations" -i 'critical ?data|massachusetts|mit\.edu|rennes|technetium|unimayor|colegio mayor|unicauca|ceadar|university college dublin|ucd\.ie|tsinghua|beth israel|harvard'
check "repository and private-notes names" -i 'curious[-_]?q?moe|anpassen|mit-projects|36-qmoe|quantumve|blindspot|fable'
check "internal document names" -i 'backlog|index\.md|master[- _]table|private-draft|overleaf|context\.md|status table|paper-versions'
check "package name" -i 'qwave'
check "results root" -i 'qmoe-results'
check "assistant traces" -i 'claude|anthropic'
check "Windows home paths" -i '[a-z]:\\users|onedrive'
check "arXiv links and the preprint id" -i 'arxiv\.org|2511\.11743'
check "absolute home paths" '' '(/Users/|/home/[A-Za-z]|-Users-)'
# emails: only the placeholder is allowed
hits=$(grep -r -n -I -o -E '[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}' "$CHK" | grep -v 'anonymous@example\.com' | sed "s|$CHK/||" || true)
if [ -n "$hits" ]; then echo "LEAK [emails other than the placeholder]"; echo "$hits" | head; fail=1; else echo "clean [emails other than the placeholder]"; fi
hits=$(grep -r -n -I -o -i -E '(github\.com/|github\.io|githubusercontent\.com/)[A-Za-z0-9_.-]*' "$CHK" \
  | grep -v -i -E 'github\.com/(fschmid56|karolpiczak|karoldvl|qiuqiangkong|openai|google-research|peterliht|jmlrorg|huggingface|pytorch|facebookresearch|microsoft)$' | sed "s|$CHK/||" || true)
if [ -n "$hits" ]; then echo "LEAK [GitHub URLs outside the third-party allowlist]"; echo "$hits" | head; fail=1; else echo "clean [GitHub URLs outside the third-party allowlist]"; fi
[ -z "$(find "$CHK" -name .git)" ] && echo "clean [.git]" || { echo "LEAK [.git]"; fail=1; }
grep -q '^Copyright (c) [0-9]* Anonymous authors$' "$CHK/$TOP/LICENSE" && echo "clean [LICENSE copyright line]" || { echo "LEAK [LICENSE]"; fail=1; }

if [ $fail -ne 0 ]; then rm -f "$OUT"; echo "FAILED: identifying strings remain, zip removed"; exit 1; fi
echo "PASSED: no identifying string found in $OUT (from $REF at $COMMIT)"
