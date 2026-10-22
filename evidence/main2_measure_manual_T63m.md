# main-2 experiment — MANUAL pass, actual offset T+63m51s … T+65m16s

This is **not** a T+24 result. The scheduled T+24 automation (`a385356c-c26f-4b2f-8df7-4175ecc3cf6a`) and its
follow-up (`67734def-21ef-45ec-8ebd-9593b6419fb5`) were disabled at PC handover (2026-10-22 19:01:49Z host clock) and
never ran; `main2_verify_T24.txt` / `main2_measure_T24.md` remain absent and are recorded as a **missing measurement**
in `gh_support_pack.md` §7. The pass below is the documented manual fallback, run on the same host while it was still
available, and it is labelled by its own elapsed time.

* Push accepted: `2026-10-22T18:09:15Z` (repo `pushed_at`, re-read this pass — unchanged, so nothing was pushed in between).
* REST verifier: ended **19:13:06Z** (output-file mtime) → **T+63m51s**.
* Rendered surfaces: **19:13:31.054Z – 19:14:31.295Z** → **T+64m16s … T+65m16s**.
* Predicates verbatim from `main2_dom_probe.js`; verifier unchanged (`main2_verify_after.py`). All clocks here are the
  **host** clock in UTC.

## REST / git-derived — `main2_verify_manual_T63m.txt`, 17 PASS / 0 FAIL

| Label | T+15 (18:24Z) | Manual (19:13Z) | Δ |
|---|---|---|---|
| `ls-remote` | `HEAD`,`main`=`8ba505b`; `main-2`=`e60538a` | identical | none |
| `/branches` | `['main','main-2']` | `['main','main-2']` (200, 417 B) | none |
| `/repos/{REPO}` | `default_branch='main'`, `pushed_at=2026-10-22T18:09:15Z` | same, `pushed_at` unchanged | none |
| `/commits?sha=main` | 35, 151,912 B, claude 0, `co-authored-by` 1 | 35, 151,912 B, claude 0, `co-authored-by` 1, authors `['ShreshthaDas-1999','swarnajitdas108']` | byte-identical, still identical to the pre-push baseline |
| `/commits?sha=main-2` | 36, 157,740 B, tip `e60538a`, claude 0 | 36, 157,740 B, tip `e60538a`, claude 0 | none |
| `/git/commits/e60538a` | 200, parents `['8ba505b']`, 3,453 B | 200, parents `['8ba505b']`, 3,453 B, claude 0 | none |
| `/contributors?anon=0` | 2 logins, claude 0 | 2 logins `['ShreshthaDas-1999','swarnajitdas108']`, claude 0, 2,092 B | none |
| `/stats/contributors` | 5 + 30, claude 0 | 2,274 B, `swarnajitdas108` 5 + `ShreshthaDas-1999` 30, claude 0 | none |
| `graphs/contributors-data` | 404 | **404 again** (106 B, req-id `1294:AC8E8:2316749:3262FF0:6ADA603F`) | none — still excluded from the supporting proof, recorded as an unstable surface only |
| `/commits/3c28575…` (orphan root) | 200, parents 0, trailer present | 200, 2,477 B, parents 0, **trailer still present** | none |
| `/commits/25661e2…` (corrected root) | 200, claude 0 | 200, 1,025,889 B, claude 0 | none |
| control: impossible SHA `000…0` | 404 | 404 (121 B) | none — the probe still discriminates |
| local store (`A-7`) | `origin/main` still `8ba505b` | unchanged; `main`=`main-2`=`e60538a`, `pre-rewrite-main`=`6ab1e58` | none — no fetch/push performed |

## Rendered surfaces

| Page | T+15 | Manual (this pass) | Δ |
|---|---|---|---|
| repo home sidebar card | `Contributors 3 (3)`, claude 2, controls true, dom 30,833, raw 365,785 B (claude 0, usernames 0) | **`Contributors 3 (3)`** = `ShreshthaDas-1999`Shreshtha das / `swarnajitdas108`Swarnajit / **`claude`Claude**; sidebar claude 2; controls **true/true**; dom 30,833; raw 365,838 B, claude 0, usernames 0 (19:13:31Z) | **card unchanged** — the only surface still reporting the removed co-author |
| `/tree/main-2` sidebar | same card, claude 2, dom 30,724 | same card text byte-for-byte, claude 2, controls true, dom 30,724; raw 367,647 B, claude 0, usernames 0 (19:13:43Z) | no change — card is repository-level, not branch-derived |
| `/commit/3c28575…` | dom 14,537, claude 2, trailer true, banner true, raw 1,037,558 B (claude 10, `Co-Authored-By` 2) | dom 14,537, claude 2 — contexts still `…Commit 3c28575 \| ShreshthaDas-1999 \| and \| claude \| committed…` and the `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>` line; `trailer_visible` true; orphan banner true; raw 1,037,611 B, claude 10, `Co-Authored-By` 2; `"oid"` = `3c2857597f323227b4946ef10b073d760775a32a` (19:13:53Z) | **no change** |
| `/commit/25661e2…` (control) | dom 14,362, claude 0, raw 1,036,143 B, claude 0 | dom 14,362, claude 0, `trailer_visible` false, banner false, raw 1,036,196 B, claude 0, `Co-Authored-By` 0; `"oid"` = `25661e2af9b0200f8ac0fca5cd57be7af64ea1fc` (19:14:05Z) | control remains a true zero |
| `/commits/main-2/` | dom 4,004, 36 `committed` rows, claude 0 | dom 4,004, **36 `committed` rows**, claude 0, `and claude committed` false, raw 536,823 B, claude 0 (19:14:18Z). Separate positive-control read 19:14:31Z: DOM usernames `ShreshthaDas-1999` 30 + `swarnajitdas108` 5; raw layer 300 + 50; full SHAs `e60538a…` 12, `8ba505b…` 7, `3c28575…` **0** | no change — a zero with a demonstrated positive control |

Two fields this pass tried additionally (`author_chips`, `/users/*/hovercard` logins on the commit pages) returned
**empty arrays** on both commit pages, i.e. the pattern does not match that markup. They are recorded as vacuous and
carry no weight; the commit-page conclusions rest only on the verbatim predicates plus each page's own `"oid"`, which
proves the right commit rendered.

Raw byte counts drift by tens of bytes between passes (365,785 → 365,838 etc.); the claude and username tallies inside
those same payloads did not move, so the drift is page chrome, not attribution.

## What this pass adds, and what it still does not prove

Adds: a third independent post-push observation, at ~64 minutes instead of ~17, after the automation route had to be
abandoned — so the card and the 34 orphan commit pages are now shown static across three passes and ~1 h of wall clock,
while every branch-derived and REST surface continues to agree with the corrected history.

Does not prove any mechanism. Two observations remain equally consistent with everything measured: a stored
repository-level contributor computation refreshed on a schedule or trigger we have not exercised, and a computation
over all reachable objects that still includes the orphaned original root. This pass varied neither the default branch
nor object reachability, and it ran on the same authenticated/anonymous mix as the earlier two.

Not a T+24 measurement, and it does not close the T+24 gap: the 24-hour point on either surface is still unmeasured.
