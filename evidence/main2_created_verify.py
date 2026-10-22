"""Verification of the authorized LOCAL-only creation of refs/heads/main-2.

Checks exactly the four conditions the owner named, plus the two that follow from
"do not rewind or move local main": main must still be where it was, and naming an
existing commit must have created zero new objects. Remote is read anonymously.
"""
import hashlib
import os
import subprocess
import sys

CLONE = r"[USER_HOME]\Documents\Qoder\2026-10-14\e14afe8e\scratch\attrib\rewrite"
PERM = r"[USER_HOME]\Documents\Qoder\nativevoice"
BUNDLE = os.path.join(os.path.dirname(CLONE), "pre-rewrite-main.bundle")
DOC = "e60538a2e29a995f1a5ea37a847b5aff690a2790"
MAIN_PUB = "8ba505b6c22e83e473187bb4c63b4ab71c8e3c49"
PREW = "6ab1e58631f584df4b06d8046a0738a3655bb52d"
BLOBS = {"DECISIONS.md": "f9cc6114ec6525ed03cd671d962a53a43fa8a7e9",
         "PROGRESS.md": "c9b650b6c5e31ed938a6df474dfffda7988221c0",
         "docs/audit-register.md": "9221d8cfa4979546f9e039772d8197a3ebf29db9"}
ok = []


def git(*a, cwd=CLONE):
    return subprocess.run(["git", "-C", cwd, *a], capture_output=True, text=True, check=True).stdout


def check(name, cond, detail=""):
    ok.append(cond)
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  | {detail}" if detail else ""))


print("=== V-1  main-2 points to the accepted documentation commit ===")
refs = dict(l.split(" ", 1) for l in git("for-each-ref", "--format=%(refname) %(objectname)").splitlines())
check("V-1a refs/heads/main-2 == e60538a", refs.get("refs/heads/main-2") == DOC,
      refs.get("refs/heads/main-2"))
check("V-1b main-2 resolves to the same commit object as e60538a",
      git("rev-parse", "main-2^{commit}").strip() == DOC)
check("V-1c main-2 is not a tag or a peeled ref", "refs/tags/main-2" not in refs)

print("\n=== V-2  parentage unchanged ===")
check("V-2a main-2's parent is the published tip 8ba505b", git("rev-parse", "main-2^").strip() == MAIN_PUB,
      git("rev-parse", "main-2^").strip())
check("V-2b single parent (no merge introduced)", len(git("rev-list", "--parents", "-n1", "main-2").split()) == 2)
check("V-2c main-2 history = 36 commits, i.e. published 35 + the doc commit",
      len(git("rev-list", "main-2").split()) == len(git("rev-list", MAIN_PUB).split()) + 1,
      f"{len(git('rev-list', MAIN_PUB).split())} -> {len(git('rev-list', 'main-2').split())}")
check("V-2d same root as the published branch (corrected root, no ancestor rewrite)",
      git("rev-list", "--max-parents=0", "main-2").strip()
      == "25661e2af9b0200f8ac0fca5cd57be7af64ea1fc",
      git("rev-list", "--max-parents=0", "main-2").strip()[:12])
check("V-2e no ref anywhere names the pre-rewrite chain",
      PREW not in set(refs.values()) - {refs.get("refs/heads/pre-rewrite-main")},
      "only refs/heads/pre-rewrite-main carries it, local-only as before")

print("\n=== V-3  the commit content main-2 names is the reviewed content ===")
tree = {l.split(None, 3)[3]: l.split(None, 3)[2] for l in git("ls-tree", "-r", "main-2").splitlines()}
check("V-3a main-2's tree is the accepted tree 976d56384c1e",
      git("rev-parse", "main-2^{tree}").strip() == "976d56384c1e3d23c468cbc3dbef7bcd3aa9092a")
check("V-3b the three authorized blobs are the reviewed ones", all(tree[p] == o for p, o in BLOBS.items()))
check("V-3c 207 tracked paths, none added or deleted", len(tree) == 207, str(len(tree)))
msg = git("log", "-1", "--format=%B", "main-2")
check("V-3d message unchanged and claude-free", "claude" not in msg.lower() and "Co-Authored-By" not in msg)

print("\n=== V-4  local main left exactly where it was ===")
check("V-4a refs/heads/main still == e60538a (not rewound, per instruction)",
      refs.get("refs/heads/main") == DOC, refs.get("refs/heads/main"))
check("V-4b HEAD still attached to refs/heads/main (no checkout happened)",
      git("symbolic-ref", "HEAD").strip() == "refs/heads/main", git("symbolic-ref", "HEAD").strip())
check("V-4c main and main-2 now name the same commit", git("rev-parse", "main", "main-2").split() == [DOC, DOC])

print("\n=== V-5  working tree clean, and no new objects were created ===")
st = git("status", "--porcelain", "--untracked-files=all")
check("V-5a status --porcelain --untracked-files=all is empty", st.strip() == "", repr(st))
def blob_sha(p):
    return hashlib.sha1(b"blob %d\x00" % len(open(p, "rb").read()) + open(p, "rb").read()).hexdigest()
check("V-5b every worktree file still hashes to main-2's blob (207 paths)",
      all(blob_sha(os.path.join(CLONE, *p.split("/"))) == o for p, o in tree.items()))
loose_before = subprocess.run(["git", "-C", CLONE, "count-objects", "-v"], capture_output=True, text=True).stdout
print("      count-objects -v:", " ".join(loose_before.split()))
check("V-5c naming an existing commit added no objects: e60538a already existed before this step",
      git("cat-file", "-t", DOC).strip() == "commit" and git("log", "-1", "--format=%H", "main-2^").strip() == MAIN_PUB)

print("\n=== V-6  remote refs unchanged (read anonymously) ===")
remote = subprocess.run(["git", "-c", "credential.helper=", "ls-remote",
                         "https://github.com/Actizant/NativeVoiceEngine.git"],
                        capture_output=True, text=True, check=True).stdout
rr = {l.split("\t", 1)[1]: l.split("\t", 1)[0] for l in remote.splitlines() if "\t" in l}
for k in sorted(rr):
    print(f"      {k} = {rr[k]}")
check("V-6a remote still has exactly HEAD + refs/heads/main", set(rr) == {"HEAD", "refs/heads/main"}, str(sorted(rr)))
check("V-6b remote main still 8ba505b", rr.get("refs/heads/main") == MAIN_PUB)
check("V-6c NO main-2 on the remote (nothing was pushed)", "refs/heads/main-2" not in rr)
check("V-6d local remote-tracking refs untouched",
      refs.get("refs/remotes/origin/main") == MAIN_PUB and refs.get("refs/remotes/origin/HEAD") == MAIN_PUB)

print("\n=== V-7  other stores untouched ===")
b = open(BUNDLE, "rb").read()
check("V-7a pre-rewrite bundle unchanged (905,415 B, sha256 0212a8dd…)",
      len(b) == 905415 and hashlib.sha256(b).hexdigest()
      == "0212a8ddefa109884554f79dd5e442aee88c5021c5bbec97a27c34f29e629e5d", f"{len(b)} B")
bv = subprocess.run(["git", "-C", CLONE, "bundle", "verify", BUNDLE], capture_output=True, text=True)
bvo = bv.stdout + bv.stderr  # git prints the tip list and the summary to stderr on this build
check("V-7b git bundle verify still reports complete history tipped at 6ab1e58 (not at 8ba505b)",
      PREW in bvo and MAIN_PUB not in bvo and "complete history" in bvo,
      " | ".join(bvo.strip().splitlines()[-2:]))
pperm = dict(l.split(" ", 1) for l in git("for-each-ref", "--format=%(refname) %(objectname)", cwd=PERM).splitlines())
check("V-7c permanent copy refs unchanged", pperm.get("refs/heads/master") == PREW, str(pperm))
pstat = git("status", "--porcelain", cwd=PERM).splitlines()
check("V-7d permanent copy still its own 2 dirty protected files, nothing more",
      sorted(pstat) == [" M artifacts/audit/mechanical-findings.json", " M artifacts/audit/offline-proof.json"],
      str(pstat))
art = {p: o for p, o in tree.items() if p.startswith("artifacts/")}
check(f"V-7e all {len(art)} artifacts/ paths in main-2 identical to the published tip's",
      all(o == {l.split(None, 3)[3]: l.split(None, 3)[2] for l in
                git("ls-tree", "-r", MAIN_PUB).splitlines()}[p] for p, o in art.items()))

print("\n" + ("ALL POST-CREATION CHECKS PASSED" if all(ok) else "SOME CHECKS FAILED: "
             + str([i for i, v in enumerate(ok) if not v])))
sys.exit(0 if all(ok) else 1)
