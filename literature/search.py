"""Query Crossref for topic searches; print candidate works (2023-2026 first)."""
import json, sys, time, urllib.parse, urllib.request, urllib.error
def q(term, rows=15, frm="2023-01-01"):
    url = ("https://api.crossref.org/works?rows=%d&query.bibliographic=%s&filter=from-pub-date:%s,type:journal-article,type:proceedings-article"
           "&select=DOI,title,author,container-title,issued,is-referenced-by-count" % (rows, urllib.parse.quote(term), frm))
    req = urllib.request.Request(url, headers={"User-Agent": "lit-review-script/1.0"})
    for k in range(6):
        try:
            return json.load(urllib.request.urlopen(req, timeout=60))["message"]["items"]
        except urllib.error.HTTPError as e:
            if e.code != 429: raise
            time.sleep(5 * (k + 1))
    return []
for term in sys.argv[1:]:
    time.sleep(2)
    print("##", term)
    for it in q(term):
        a = it.get("author", [{}])[0].get("family", "?")
        y = it["issued"]["date-parts"][0][0]
        print(f"  {y} | {a} | {it.get('title',[''])[0][:110]} | {(it.get('container-title') or [''])[0][:40]} | {it['DOI']} | cit={it.get('is-referenced-by-count')}")
