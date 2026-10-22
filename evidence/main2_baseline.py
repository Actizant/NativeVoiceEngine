"""Read-only "before" baseline for the main-2 experiment.

Anonymous only (no credential is sent anywhere), redirects disabled, and every
status is accepted only when the response body has the shape that status implies
- a 200 carrying an error payload is a 404, and that mistake already produced one
wrong report in this task. Records the request id so each observation is
attributable. Nothing here changes the repository.
"""
import json
import ssl
import urllib.error
import urllib.request

API = "https://api.github.com"
REPO = "Actizant/NativeVoiceEngine"
ORPHAN_ROOT = "3c2857597f323227b4946ef10b073d760775a32a"
NEW_DOC = "e60538a2e29a995f1a5ea37a847b5aff690a2790"
CTX = ssl.create_default_context()
lines = []


def get(path, accept="application/vnd.github+json"):
    req = urllib.request.Request(API + path, headers={
        "Accept": accept, "User-Agent": "attribution-audit-readonly",
        "X-GitHub-Api-Version": "2022-11-28"})
    try:
        with urllib.request.urlopen(req, context=CTX, timeout=45) as r:
            body, status, hdrs = r.read(), r.status, r.headers
    except urllib.error.HTTPError as e:
        body, status, hdrs = e.read(), e.code, e.headers
    # header names are case-preserved by urllib; a case-sensitive .get() silently returns None
    folded = {k.lower(): v for k, v in hdrs.items()}
    return status, body, folded


def probe(label, path, accept="application/vnd.github+json"):
    status, body, hdrs = get(path, accept)
    rid = hdrs.get("x-github-request-id", "?")
    extra = " ".join(f"{k}={hdrs[k]}" for k in ("retry-after", "content-type") if k in hdrs)
    try:
        data = json.loads(body.decode("utf-8", "replace"))
    except Exception:
        data = None
    # body-shape guard: a JSON object with these keys is an API error, whatever status said
    is_error = isinstance(data, dict) and {"message", "documentation_url"} <= set(data)
    shape = "ERROR-BODY" if is_error else ("list" if isinstance(data, list)
                                          else "dict" if isinstance(data, dict) else "raw")
    out = f"{label:<34} HTTP {status}  body-shape={shape:<10} req-id={rid}  bytes={len(body)} {extra}"
    if is_error:
        out += f"  message={data.get('message')!r}"
    lines.append(out)
    print(out)
    return status, data, is_error


print("=== B-1  branch surface that exists today ===")
status, meta, err = probe("B-1a repo meta", f"/repos/{REPO}")
if isinstance(meta, dict) and not err:
    print(f"      default_branch={meta.get('default_branch')!r} "
          f"pushed_at={meta.get('pushed_at')} updated_at={meta.get('updated_at')} "
          f"size={meta.get('size')} fork={meta.get('fork')}")
status, data, err = probe("B-1b branch list", f"/repos/{REPO}/branches?per_page=100")
branches = [b["name"] for b in data] if isinstance(data, list) and not err else None
print(f"      branches -> {branches}   main-2 present: {bool(branches and 'main-2' in branches)}")
status, data, err = probe("B-1c git refs", f"/repos/{REPO}/git/refs/heads")
if isinstance(data, list) and not err:
    for ref in data:
        print(f"      {ref['ref']} = {ref['object']['sha'][:12]}")

print("\n=== B-2  commit surface keyed on the branch that exists ===")
status, data, err = probe("B-2a main commit list", f"/repos/{REPO}/commits?sha=main&per_page=100")
if isinstance(data, list) and not err:
    txt = json.dumps(data).lower()
    authors = sorted({(c["author"] or {}).get("login") or c["commit"]["author"]["email"] for c in data})
    print(f"      count={len(data)}  'claude' occurrences={txt.count('claude')}  "
          f"'co-authored-by'={txt.count('co-authored-by')}")
    print(f"      author logins on main: {authors}")
    print(f"      tip = {data[0]['sha'][:12]} {data[0]['commit']['message'].splitlines()[0][:60]!r}")
status, data, err = probe("B-2b main-2 commit list (must 404 now)",
                          f"/repos/{REPO}/commits?sha=main-2&per_page=100")
status, data, err = probe("B-2c new doc commit by SHA", f"/repos/{REPO}/commits/{NEW_DOC}")
if isinstance(data, dict) and not err:
    m = data["commit"]["message"]
    print(f"      served anonymously BEFORE any push: author={data['author'] and data['author']['login']}"
          f" 'claude' in message={m.lower().count('claude')} parents={[p['sha'][:12] for p in data['parents']]}")

print("\n=== B-3  contributor surfaces (the two that disagree today) ===")
status, data, err = probe("B-3a REST /contributors", f"/repos/{REPO}/contributors?per_page=100&anon=0")
if isinstance(data, list) and not err:
    print(f"      logins: {[c['login'] for c in data]}")
status, data, err = probe("B-3b graphs/contributors-data",
                          f"/repos/{REPO}/graphs/contributors-data",
                          accept="application/json")
if isinstance(data, list) and not err:
    print(f"      {len(data)} entries: "
          f"{[(d['name'], d['email'], d['total']) for d in data]}")
    print(f"      'claude' anywhere in payload: {json.dumps(data).lower().count('claude')}")

print("\n=== B-4  the orphaned original root, by direct object route ===")
status, data, err = probe("B-4a orphan root /git/commits", f"/repos/{REPO}/git/commits/{ORPHAN_ROOT}")
if isinstance(data, dict) and not err:
    m = data.get("message", "")
    print(f"      FETCHABLE by anonymous object route: sha={data.get('sha','')[:12]} "
          f"parents={[p['sha'][:12] for p in data.get('parents',[])]}")
    print(f"      message tail={m[-80:]!r}")
    print(f"      author={data.get('author')}")
    print(f"      'claude' occurrences in this object: {m.lower().count('claude')}")
status, data, err = probe("B-4b positive control: a SHA that cannot exist",
                          f"/repos/{REPO}/git/commits/0000000000000000000000000000000000000000")
status, data, err = probe("B-4c new doc commit /git/commits (must be absent)",
                          f"/repos/{REPO}/git/commits/{NEW_DOC}")

with open("main2_baseline_rest.txt", "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines) + "\n")
print("\nrestored -> main2_baseline_rest.txt")
