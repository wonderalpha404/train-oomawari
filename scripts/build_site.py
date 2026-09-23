from pathlib import Path
import csv,json,shutil
from pykakasi import kakasi
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/"site"; DATA=ROOT/"data"/"tokyo_near_zone.csv"
OUT.mkdir(exist_ok=True)
for src,dst in [(ROOT/"docs"/"index.html",OUT/"index.html"),(ROOT/"oomawari.py",OUT/"oomawari.py"),(DATA,OUT/"data.csv")]: shutil.copy2(src,dst)
prefixes=("(武蔵)","(横)","(岸)","(中)","(川)","(篠)","(北)","(成)","(両)","(烏)","(信)","(房)","(総)","(臨)","（臨）")
def clean(s):
    for p in prefixes:s=s.replace(p,"")
    return s
names=set()
with DATA.open(encoding="utf-8",newline="") as f:
    for row in csv.reader(f):
        if not row or row[0].startswith("#"):continue
        names.add(clean(row[0]));names.add(clean(row[1]))
conv=kakasi(); out=[]
for name in names:
    out.append({"name":name,"reading":"".join(x["hira"] for x in conv.convert(name))})
out.sort(key=lambda x:x["reading"])
(OUT/"stations.json").write_text(json.dumps(out,ensure_ascii=False,separators=(",",":")),encoding="utf-8")
print("built",len(out),"stations")
