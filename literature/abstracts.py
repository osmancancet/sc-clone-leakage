"""Fetch OpenAlex abstracts for DOIs (reconstructed from the inverted index)."""
import json, sys, time, urllib.request
def get(doi):
    url = "https://api.openalex.org/works/https://doi.org/" + doi
    for k in range(5):
        try:
            return json.load(urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "lit-review-script/1.0"}), timeout=60))
        except Exception as e:
            time.sleep(3 * (k + 1))
    return {}
out = {}
for doi in sys.argv[1:]:
    w = get(doi); inv = w.get("abstract_inverted_index") or {}
    pos = sorted((p, t) for t, ps in inv.items() for p in ps)
    out[doi] = {"title": w.get("title"), "year": w.get("publication_year"), "abstract": " ".join(t for _, t in pos),
                "oa_url": (w.get("best_oa_location") or {}).get("pdf_url") or (w.get("open_access") or {}).get("oa_url")}
    print("==", doi, out[doi]["year"], out[doi]["title"]); print(out[doi]["abstract"][:900]); print("OA:", out[doi]["oa_url"])
json.dump(out, open("abstracts_%d.json" % int(time.time()), "w"), indent=1)
