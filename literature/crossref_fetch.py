"""Fetch Crossref metadata for every DOI in refs_seed.txt -> refs_crossref.json."""
import json, time, urllib.request, urllib.parse, urllib.error
out = {}
for line in open("refs_seed.txt"):
    key, doi = line.strip().split("|")
    url = "https://api.crossref.org/works/" + urllib.parse.quote(doi)
    for k in range(6):
        try:
            m = json.load(urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "refcheck/1.0"}), timeout=60))["message"]
            break
        except urllib.error.HTTPError as e:
            if e.code == 404: m = None; break
            time.sleep(4 * (k + 1))
    out[key] = {"doi": doi, "found": m is not None, "meta": m and {k2: m.get(k2) for k2 in ("title", "author", "container-title", "volume", "issue", "page", "article-number", "issued", "published", "type", "publisher", "event", "DOI")}}
    print(key, "OK" if m else "NOT FOUND", (m or {}).get("title", [""])[0][:80] if m else "")
    time.sleep(0.7)
json.dump(out, open("refs_crossref.json", "w"), indent=1, ensure_ascii=False)
