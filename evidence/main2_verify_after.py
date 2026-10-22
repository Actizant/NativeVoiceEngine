"""Post-push verification for the main-2 experiment. Anonymous only.

Same discipline as main2_baseline.py: redirects are not followed, and a status is
believed only when the response body has the shape that status implies. Diff against
the baseline lines in main2_baseline_rest.txt by request label, not by memory.
Run AFTER the approved push; before it, B-2b/B-2c/B-4c are expected to be 404/422.
"""
import json
import ssl
import subprocess
import urllib.error
import urllib.request

REPO = "Actizant/NativeVoiceEngine"
API = "https://api.github.com"
CTX = ssl.create_default_context()
MAIN = "8ba505b6c22e83e473187bb4c63b4ab71c8e3c49"
DOC = "e60538a2e29a995f1a5ea37a847b5aff690a2790"
ORPHAN_ROOT = "3c2857597f323227b4946ef10b073d760775a32a"
CORRECTED_ROOT = "25661e2af9b0200f8ac0fca5cd57be7af64ea1fc"
results = []


def get(path, accept="application/vnd.github+json"):
    req = urllib.request.Request(API + path, headers={
        "Accept": accept, "User-Agent": "attribution-audit-readonly",
        "X-GitHub-Api-Version": "2022-11-28"})
    try:
        with urllib.request.urlopen(req, context=CTX, timeout=45) as r:
            return r.status, r.read(), {k.lower(): v for k, v in r.headers.items()}
    except urllib.error.HTTPError as e:
        return e.code, e.read(), {k.lower(): v for k, v in e.headers.items()}


def probe(label, path, accept="application/vnd.github+json"):
    status, body, hdrs = get(path, accept)
    try:
        data = json.loads(body.decode("utf-8", "replace"))
    except Exception:
        data = None
    err = isinstance(data, dict) and {"message", "documentation_url"} <= set(data)
    print(f"{label:<32} HTTP {status}  req-id={hdrs.get('x-github-request-id','?')}  "
          f"bytes={len(body)}" + (f"  message={data.get('message')!r}" if err else ""))
    return status, data, err


print("=== A-1  ref reachability (ls-remote is the authority) ===")
out = subprocess.run(["git", "-c", "credential.helper=", "ls-remote",
                      f"https://github.com/{REPO}.git"], capture_output=True, text=True).stdout
refs = {}
for line in out.splitlines():
    if "\t" in line:
        oid, name = line.split("\t", 1)
        refs[name] = oid
for name in sorted(refs):
    print(f"      {name} = {refs[name]}")
results.append(("A-1a main still exactly the published tip", refs.get("refs/heads/main") == MAIN))
results.append(("A-1b main-2 now names the doc commit", refs.get("refs/heads/main-2") == DOC))
results.append(("A-1c exactly HEAD + 2 branches, nothing else", len(refs) == 3
                and set(refs) == {"HEAD", "refs/heads/main", "refs/heads/main-2"}))
results.append(("A-1d no ref names the pre-rewrite tip 6ab1e58",
                "6ab1e58631f584df4b06d8046a0738a3655bb52d" not in set(refs.values())))
results.append(("A-1e no new tag appeared", not any(n.startswith("refs/tags/") for n in refs)))

print("\n=== A-2  branch and repo meta ===")
s, d, e = probe("A-2a branch list", f"/repos/{REPO}/branches?per_page=100")
names = sorted(b["name"] for b in d) if isinstance(d, list) and not e else None
print(f"      branches -> {names}")
results.append(("A-2a branches are exactly main + main-2", names == ["main", "main-2"]))
s, d, e = probe("A-2b repo meta", f"/repos/{REPO}")
if isinstance(d, dict) and not e:
    print(f"      default_branch={d.get('default_branch')!r} pushed_at={d.get('pushed_at')}")
    results.append(("A-2b default branch is STILL main (not switched)", d.get("default_branch") == "main"))

print("\n=== A-3  branch-derived commit surface ===")
for label, sha, want in (("A-3a main", "main", 35), ("A-3b main-2", "main-2", 36)):
    s, d, e = probe(f"{label} commit list", f"/repos/{REPO}/commits?sha={sha}&per_page=100")
    if isinstance(d, list) and not e:
        txt = json.dumps(d).lower()
        authors = sorted({(c["author"] or {}).get("login") or c["commit"]["author"]["email"] for c in d})
        print(f"      count={len(d)} tip={d[0]['sha'][:12]} 'claude'={txt.count('claude')} "
              f"'co-authored-by'={txt.count('co-authored-by')} authors={authors}")
        results.append((f"{label}: {len(d)} == expected {want}", len(d) == want))
        results.append((f"{label}: zero 'claude' in branch history", txt.count("claude") == 0))
        results.append((f"{label}: no trailer introduced", "claude opus" not in txt))

print("\n=== A-4  the doc commit is now served (was 404/422 at baseline) ===")
s, d, e = probe("A-4a /git/commits/e60538a", f"/repos/{REPO}/git/commits/{DOC}")
if isinstance(d, dict) and not e:
    m = d.get("message", "")
    print(f"      parents={[p['sha'][:12] for p in d.get('parents',[])]} 'claude'={m.lower().count('claude')}")
    results.append(("A-4a served with parent = published tip",
                    [p["sha"] for p in d.get("parents", [])] == [MAIN]))
    results.append(("A-4b message carries no claude token", "claude" not in m.lower()))

print("\n=== A-5  contributor surfaces vs the 17:47Z baseline ===")
s, d, e = probe("A-5a REST /contributors", f"/repos/{REPO}/contributors?per_page=100&anon=0")
logins = [c["login"] for c in d] if isinstance(d, list) and not e else None
print(f"      logins -> {logins}   'claude' in payload={json.dumps(d).lower().count('claude') if logins else 'n/a'}")
print("      baseline was ['ShreshthaDas-1999','swarnajitdas108']")
s, d, e = probe("A-5b graphs/contributors-data", f"/repos/{REPO}/graphs/contributors-data",
                accept="application/json")
print("      NOTE: unstable surface (200 earlier today, 404 at baseline) - recorded, not used as evidence")
s, d, e = probe("A-5c stats/contributors", f"/repos/{REPO}/stats/contributors")
if isinstance(d, list) and not e:
    print(f"      {len(d)} entries: {[(x['author']['login'], x['total']) for x in d]}")

print("\n=== A-6  the orphaned objects: still served, still unreferenced ===")
s, d, e = probe("A-6a orphan root by URL", f"/repos/{REPO}/git/commits/{ORPHAN_ROOT}")
if isinstance(d, dict) and not e:
    m = d.get("message", "")
    print(f"      STILL FETCHABLE: parents={len(d.get('parents',[]))} trailer present="
          f"{'Co-Authored-By: Claude Opus 5' in m}")
s, d, e = probe("A-6b corrected root", f"/repos/{REPO}/commits/{CORRECTED_ROOT}")
if isinstance(d, dict) and not e:
    print(f"      clean control: message 'claude'={d['commit']['message'].lower().count('claude')}")
s, d, e = probe("A-6c control SHA that cannot exist",
                 f"/repos/{REPO}/git/commits/0000000000000000000000000000000000000000")
results.append(("A-6c control still 404 (the probe discriminates)", s == 404 and e))
results.append(("A-6d the main branch surface is unchanged vs baseline",
                refs.get("refs/heads/main") == MAIN))

print("\n=== A-7  local store untouched by the read ===")
loc = subprocess.run(["git", "-C", r"[USER_HOME]\Documents\Qoder\2026-10-14\e14afe8e\scratch\attrib\rewrite",
                      "for-each-ref", "--format=%(refname) %(objectname)"], capture_output=True, text=True).stdout
print("".join(f"      {l}\n" for l in loc.splitlines()))
print("      origin/* must still be 8ba505b until an explicit fetch is authorised")

print("\n" + ("-" * 60))
bad = [n for n, ok in results if not ok]
for n, ok in results:
    print(f"{'PASS' if ok else 'FAIL'}  {n}")
print("\n" + ("MAIN-2 VERIFICATION PASSED" if not bad else f"FAILURES: {bad}"))
