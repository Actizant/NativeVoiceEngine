# evidence/ — attribution-correction measurement record (test branch `main-2`)

This directory is a **measurement record**, not application material. It exists so the owner can fetch the
whole attribution-correction evidence trail from another machine instead of carrying it on a USB copy, and so
anyone reading the `main-2` experiment can re-check the numbers rather than take them on trust.

* Provenance: added on `refs/heads/main-2`, on top of `e60538a2e29a995f1a5ea37a847b5aff690a2790` (parent
  `8ba505b6c22e83e473187bb4c63b4ab71c8e3c49`, the default-branch tip). `main` is untouched by this work.
* Nothing here is a legal conclusion about authorship. The documents separate *what a surface displayed at a
  named UTC minute* from *what might cause it*, and every zero is paired with the positive control that makes
  it a measurement.
* The GitHub Support request in `gh_support_pack.md` §6 is **prepared and unsubmitted**. Publishing this
  directory does not submit it.

## Contents (21 files, 109,677 bytes; plus this README)

Inclusion rule, so the set is not arbitrary: a file is here if the support pack or the handover checkpoint
cites it as evidence, **or** it is the measurement predicate (verifier script / DOM probe) that produced an
output cited there. Everything else under `scratch/attrib` stays local.

| File | Bytes | sha256/16 | Transform |
|---|---|---|---|
| `gh_support_pack.md` | 18,696 | `bdafd99068f3a049` | redacted |
| `CHECKPOINT_2026-10-22T19-01Z.md` | 12,817 | `a554a818950dd1a3` | redacted |
| `main2_measure_T0.md` | 8,022 | `7f6665eadeceea57` | verbatim |
| `main2_created_verify.py` | 7,488 | `e0583d83c497bbd6` | redacted |
| `main2_verify_after.py` | 7,346 | `701db1948a4d0877` | redacted |
| `main2_measure_manual_T63m.md` | 6,321 | `a550464bf0e1e031` | verbatim |
| `main2_baseline.py` | 6,092 | `c9ebada7df0c1642` | verbatim |
| `main2_dom_probe.js` | 4,391 | `c9553f6ee367c7ae` | verbatim |
| `main2_measure_T15.md` | 4,481 | `a0fd4820253ec4eb` | verbatim |
| `gh_support_evidence_raw.txt` | 4,080 | `a7e01836d1f5c302` | verbatim |
| `main2_verify_T15.txt` | 3,882 | `606b0612ecea9978` | LF only |
| `main2_verify_preevidencepush.txt` | 3,834 | `cd7b1f6a7563b287` | LF only |
| `main2_verify_T0.txt` | 3,834 | `de9533bb42838d42` | LF only |
| `main2_verify_manual_T63m.txt` | 3,830 | `2f20015c64fe1c4a` | LF only |
| `main2_verify_prepush_control.txt` | 3,397 | `e38115fbad00d1b3` | LF only |
| `main2_baseline_out.txt` | 3,100 | `a3c7166cfaab9753` | redacted + LF |
| `main2_created_verify.txt` | 3,085 | `fba0e6f15872a21c` | redacted + LF |
| `main2_baseline_rest.txt` | 2,032 | `5cac4cbc1aa3bd87` | LF only |
| `gh_support_evidence.txt` | 1,630 | `20e63bd51ec265c3` | LF only |
| `gh_support_evidence3.txt` | 736 | `9dc547b5b738fac0` | LF only |
| `main2_push_pre.txt` | 583 | `f99a707c5e633a2c` | verbatim |

`Transform` is what was done to the local original, nothing more. Read order: `gh_support_pack.md` (the case
and the capture passes) → `CHECKPOINT_2026-10-22T19-01Z.md` (frozen state, fingerprints, the manual
procedure, the PC-transfer inventory) → the `main2_measure_*` tables (rendered surfaces per pass) → the
`main2_verify_*` outputs (the REST/git passes behind them) → `main2_verify_after.py`,
`main2_created_verify.py`, `main2_baseline.py` and `main2_dom_probe.js` (the predicates, so a re-run is
comparable rather than merely similar).

`main2_verify_preevidencepush.txt` is the newest pass: the same 17-predicate verifier, run read-only at
2026-10-22 19:39Z, **immediately before the commit that publishes this directory** — 17 PASS, 0 FAIL, `main`
and `refs/heads/main` both still `8ba505b`, `main-2` still `e60538a`. It is included as the pre-commit
control, and it plainly cannot describe the commit that contains it; the post-push state is whatever
`git ls-remote` reports when you fetch.

## What was changed, and what was not

Two content substitutions, applied to 6 of the 21 files, plus one line-ending normalisation. **No measurement
figure, count, timestamp, request-id or SHA was altered, added or removed, and no line count changed.**

1. **Home directory → `[USER_HOME]`.** 11 occurrences across `CHECKPOINT_…md` (6),
   `main2_created_verify.py` (2), `gh_support_pack.md` (1), `main2_verify_after.py` (1) and
   `main2_created_verify.txt` (1). The substitution is the repository's own
   `nativevoice.diagnostics.evidence_paths.redact()` — the same helper and the same placeholder that
   `tests/test_no_personal_paths.py` pins — so the published copies satisfy the existing guard instead of
   inventing a second convention. That guard was run against this directory before the commit: every file
   here, this README included, is inside its whole-tree scanned set, and a planted personal path was
   detected and removed again to prove the sweep is not a no-op.
2. **One personal e-mail address → `[AUTHOR-EMAIL-REDACTED]`** in `main2_baseline_out.txt`. It appears inside
   a quoted API author object; no counted figure depends on it. `noreply@anthropic.com` is **kept**, because
   it is part of the trailer string being evidenced, not a personal address.
3. **CRLF → LF in 10 files** (byte deltas 11–67, each exactly the number of lines it applies to). The repo
   declares `* text=auto eol=lf` in `.gitattributes`, so git stores and checks out LF regardless;
   normalising here means the SHA-256 table above is what you get on disk after a clone, not what a working
   copy looked like on one Windows machine. A scan for carriage returns *not* followed by a line feed found
   0 of them, so nothing that could be content was touched.

Against the fingerprints in `CHECKPOINT_…md` §3: 7 files are byte-identical to their originals, 8 differ only
by line endings, and 6 differ by the redactions above (2 of those also by line endings). The originals stay
on the machine where the measurements were taken. `main2_verify_preevidencepush.txt` post-dates that
checkpoint, so it has no §3 fingerprint — its digest is the table row.

Consequence worth knowing before re-running anything: two predicates hard-code the local clone path —
`main2_verify_after.py` line 124 (`git -C`) and `main2_created_verify.py` lines 12–13 (`CLONE`, `PERM`).
Both now carry the `[USER_HOME]` placeholder; set them to your own clone before running either script. Every
other check is a network read and needs no local path.

GitHub usernames (`ShreshthaDas-1999`, `swarnajitdas108`) are **retained deliberately**: they are the positive
controls that make each reported zero a measurement rather than an empty read. Public logins are not the
sensitive category here. Absolute paths that survive the `[USER_HOME]` substitution keep the
repository-relative tail (`Documents\Qoder\2026-10-14\e14afe8e\scratch\attrib\…`): the chat-workspace folder
name is neither an account name nor a secret, and it is what tells the owner which private copy a report came
from.

## Deliberately excluded

* **`pre-rewrite-main.bundle` (905,415 B, sha256 `0212a8dde…`) — left out, and this is the one exclusion
  that needed a judgement.** It verifies cleanly (`git bundle verify`: *"is okay"*, *"records a complete
  history"*, 5 refs all `6ab1e58631f584df4b06d8046a0738a3655bb52d`, sha1 — re-run read-only 2026-10-22
  19:36Z from inside the clone) and contains no credential material (scanned for
  `ghp_`/`github_pat_`/`Bearer`/private-key/password patterns — no match). Its author/committer addresses are
  the same two already published in `main-2`'s own commit metadata, so it adds no new identity exposure. It
  was still excluded because committing it would:
  1. place the removed `Co-Authored-By: Claude Opus 5` trailer **inside the repository as file content**, on
     a branch of the very repo whose Support request asks GitHub to stop displaying that attribution — the
     request and the artifact would contradict each other;
  2. make the orphaned 34-commit chain reconstructible by anyone who clones this branch (`git unbundle`
     recreates the old tip locally), which changes the exposure statement the pack is careful to keep narrow
     (§4: object-level fetchability from the remote was *not* claimed and is not tested);
  3. be un-reviewable in the way the review was asked for: a packfile is opaque binary, so it cannot be read
     for sensitive content line by line and cannot be redacted without rewriting the commits inside it —
     which is exactly the history rewrite that stays unauthorized.
  The bundle's role — recovering the pre-rewrite chain — is a **private-copy** role, best served off-remote.
  Its integrity is separately evidenced here by `main2_created_verify.txt` (V-7b: `bundle verify` still
  reports complete history tipped at `6ab1e58`) and by the `main`→`main-2` content delta quoted in the
  checkpoint.
* `rewrite_strip_trailer.py` (8,264 B) and `rewrite_strip_trailer.orig.py` (5,288 B) — the rewrite tooling.
  Not a measurement; the old→new commit map it produced is already the auditable record in
  `docs/audit-register.md` Addendum 28.
* `main2_push_attempt1.stdout` (0 B) / `.stderr` (266 B) — transport chatter from the authorized push
  (GitHub's create-a-PR notice). `main2_push_pre.txt`, the part that actually records ref state, **is**
  included.
* `main2_plan.md` (12,657 B) — the pre-execution route for the branch. Excluded because its own heading says
  *"NOT executed"* and *"Nothing below has been run"*, which the push at 18:09:15Z superseded; publishing a
  document whose status line is false is the kind of stale published statement this repository actively
  corrects. Its execution is recorded here instead by `main2_push_pre.txt` and the T+0 pass. Its size and
  SHA-256 are quoted in `CHECKPOINT_…md` §3, so the private copy stays verifiable, and the one place a report
  names it (`main2_measure_T0.md`) is a locator to that inventory, not a measurement that depends on it.
* The remaining `scratch/attrib` material — `gate3/4/5/6_restore_tmp/`, `mech/`, `p0/`, `prefixprobe/`,
  `review_probe/` (measured `du -sb`: 16,629,137 B for those five together), the gate/amend/verify driver
  scripts, and the Route-A/committed-tree gate outputs (`committed_census.txt`, `postcommit_verify*.txt`).
  Process scaffolding and checks whose *conclusions* are already published in `docs/audit-register.md`; none
  is cited by the support pack as evidence for the attribution claim.
* **One honest gap:** the V-7 tree census has no saved raw output file. It was run read-only in the shell at
  19:12Z on 2026-10-22 and its result, including both controls, is recorded in `gh_support_pack.md` §1 row
  V-7 and quoted again in `CHECKPOINT_…md` §3/§6. The row names the method (`git rev-list` +
  `git log --format=%T`); the exact per-commit form that makes the positional comparison non-vacuous is
  reproduced here so it can be re-run rather than trusted:

  ```bash
  git rev-list <ref> | while read -r c; do git log -1 --format='%T' "$c"; done
  ```

* **`artifacts/audit/mechanical-findings.json` and `artifacts/audit/offline-proof.json` — never copied,
  never staged, never committed.** They remain exactly as found on the source machine, uncommitted and
  untouched, per the standing protection of `artifacts/`.

## How to read the `Claude` strings in these files

Several files quote the literal text `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`. That is
**evidence text about what GitHub displays**, held in a file body. The commit that adds this directory carries
no co-author trailer, and no commit on `main-2` does — the pack's own §2 R-2 warns against confusing the word
in prose with the rendered co-author chip, and the same caution applies here.

## Fetching this from another machine

```bash
git clone https://github.com/Actizant/NativeVoiceEngine.git
cd NativeVoiceEngine
git switch -c evidence-review origin/main-2      # origin/main-2 after the evidence commit
ls evidence
sha256sum evidence/gh_support_pack.md            # compare against the table above
```

Anonymous read needs no credential. `main-2` is an experiment branch: nothing here proposes merging it into
`main`, changing the default branch, or deleting any ref.
