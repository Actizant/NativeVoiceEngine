# Repository relocation — plan, verification, and what is still his to decide

Status: **verified in the relocated copy; the original workspace is untouched and neither copy is
authoritative yet.** Nothing was deleted.

## Target

`[USER_HOME]\Documents\Qoder\2026-10-12\nativevoice` — chosen because it sits beside the session
folder, is human-named, and needs no privileges. The original remains at
`[USER_HOME]\Documents\Qoder\2026-10-12\76969c5c`.

## Method

`git clone` of the local repository rather than a file copy. That is the safe direction on purpose: a
clone can only contain tracked history, so an accidental copy of ignored or generated material is
impossible, and any file the clone lacks is a file that was never versioned — which is exactly the
question a relocation should answer.

## Verification performed from the new location

| Check | Result |
|---|---|
| Commit history preserved | **6 commits**, identical `HEAD` `55d1cde…` in both copies |
| Tree identity | `HEAD^{tree}` `ad936f8…` identical (pre-fix) in both — content is byte-identical, not merely similar |
| Tracked files | **157 = 157** |
| Ignored material stayed out | `Python/`, `artifacts/**/*.wav`, corpus clip directory, `.pytest_cache` all **absent** by design; `.gitignore` + `.gitattributes` tracked |
| Working tree | clean after `git pull` |
| Test suite | **669 passed, 7 skipped** |
| Audit instrument | **11/11 PASS** |
| Asset resolution | data root resolves to the same absolute `%LOCALAPPDATA%\ultron-voice`; all three models report `installed/verified/usable = True` |
| Voice identity | `voice.platform.default` present: `en` measured WER 0.049383, `zh` measured-failed CER 0.462963, `hi`/`bn` unsupported; **no unreadable identity files** |
| Real inference offline | `tools/offline_proof.py` **PASS**, transcribed its own synthesised input, **zero network attempts** |

## Defect the relocation exposed — and it was a real one

`diagnostics.offline.static_scan` derived reported paths by splitting the full path on the literal
package name at its **first** occurrence. In the original workspace that happened to work. The new
repository directory is named `nativevoice`, so the first match was the repository itself and every
finding came back as `src/nativevoice/diagnostics/offline.py`, which stopped matching the allowlist —
so the offline-compliance scan in a clone would have reported the guard module as an unexplained
network importer.

Fixed to compute paths relative to the directory actually being scanned, with a regression test that
rebuilds the nested-name case in a temporary directory. This is **finding A26** in the audit register.
It is also the strongest argument for having done the relocation now rather than after an external
developer depended on the layout.

## What remains, and who does it

1. **You pick the permanent home.** The verified copy is fine as a destination, but a directory under a
   dated Qoder session path is not a good long-term parent. Say the path and I move the clone there and
   re-run this same verification table.
2. **Delete nothing yet.** The original workspace stays until you confirm the relocated copy is the one
   you want to keep working in. Deleting a workspace is the one step here that is not reversible.
3. **`origin` of the clone currently points at the original path.** Before the original goes away, it
   must be repointed (`git remote set-url origin <intended-remote>`) or pushed to a real remote. A clone
   whose origin vanishes still works — history is local — but the ambiguity is worth avoiding.

## N10 — the unexplained interpreter directory, recorded rather than removed

- **Location:** `Python/` (inside the original
  workspace root, therefore absent from the clone).
- **Contents:** `pythoncore-3.15-64` (42 MB, the interpreter itself: `python.exe`, `python315.dll`,
  `DLLs/`, `Lib/`, CPython `LICENSE.txt`), `_pythoncore-3.15-64` (9 KB metadata), `_cache` with
  `last_welcome.txt`. 744 files total.
- **Ownership evidence:** it is **not ours**. (a) We never invoke it — every command in this project runs
  `[USER_HOME]\AppData\Local\Python\pythoncore-3.14-64\python.exe`, Python **3.14.7**, and the stray
  tree is **3.15**; (b) the `pythoncore-<version>-<arch>` layout plus `_cache/last_welcome.txt` is the
  Windows Python install manager's own on-disk shape, and its install root on this machine is
  `%LOCALAPPDATA%\Python` — the same naming we see here, resolved relative to the current working
  directory; (c) the tree appeared during this session's timestamp window, not at workspace inspection,
  when the directory was empty.
- **Conclusion:** the install manager materialised a runtime into whatever directory was current. It is
  a toolchain side effect belonging to the user's environment, not project content.
- **Action taken:** listed in `.gitignore` with a comment explaining why, and **not deleted and not
  modified** — a script that did not create a 42 MB interpreter is not the right thing to remove it.
- **Action left to you:** delete it, move it, or leave it. It will not be committed either way, and the
  relocated copy does not contain it.
