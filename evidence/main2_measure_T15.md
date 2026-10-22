# main-2 experiment — T+15 pass (actual measurements)

Capture window: REST verifier started **18:24:47 UTC**; rendered surfaces **18:26:07–18:26:51 UTC**.
Push was accepted at 18:09:15Z, so this pass is **T+15m32s … T+17m36s**, not a nominal 15:00.
Predicates: identical to `main2_dom_probe.js` / the T+0 pass. Verifier output: `main2_verify_T15.txt`.

## REST / git-derived — 17/17 PASS

| Label | T+0 (18:10Z) | T+15 (18:24Z) | Δ |
|---|---|---|---|
| `ls-remote` | `HEAD`,`main`=`8ba505b`; `main-2`=`e60538a` | identical | none |
| `/branches` | `['main','main-2']` | `['main','main-2']` | none |
| `/repos/{REPO}` | `default_branch='main'`, `pushed_at=2026-10-22T18:09:15Z` | same, `pushed_at` unchanged | none |
| `/commits?sha=main` | 35, 151,912 B, claude 0, `co-authored-by` 1 | 35, 151,912 B, claude 0, `co-authored-by` 1 | **byte-identical, and identical to the pre-push baseline** |
| `/commits?sha=main-2` | 36, 157,740 B, tip `e60538a`, claude 0 | 36, 157,740 B, tip `e60538a`, claude 0 | none |
| `/git/commits/e60538a` | 200, parents `['8ba505b']`, 3,453 B | 200, parents `['8ba505b']`, 3,453 B | none |
| `/contributors?anon=0` | 2 logins, claude 0 | 2 logins, claude 0 | none |
| `/stats/contributors` | `swarnajitdas108` 5 + `ShreshthaDas-1999` 30, claude 0 | identical, 2,274 B | none |
| `graphs/contributors-data` | 404 (106 B, `Not Found`) | **404 again** (req-id `6AF1:1922A7:22603C3:313763D:6ADA54F4`) | none — 3rd consecutive 404 today |
| `/git/commits/3c28575` (orphan) | 200, parents 0, trailer present | 200, parents 0, trailer present | none |
| `/commits/25661e2` (clean root) | 200, claude 0 | 200, claude 0 | none |
| control SHA `000…0` | 404 | 404 | none — probe still discriminates |

## Rendered surfaces

| Page | T+0 | T+15 | Δ |
|---|---|---|---|
| repo home sidebar card | `Contributors 3 (3)` = ShreshthaDas-1999, swarnajitdas108, **claudeClaude**; sidebar claude 2; controls true; dom 30,833; raw 365,843 B, claude 0, usernames 0 | **identical card and text**; sidebar claude 2; controls true; dom 30,833; raw 365,785 B, claude 0, usernames 0 | **no change** (raw −58 B, no claude/usernames either time) |
| `/tree/main-2` sidebar | same card, claude 2, controls true, dom 30,724 | same card, claude 2, controls true, dom 30,724 | no change |
| `/commit/3c28575` | dom 14,537; claude 2; `trailer_visible` true; orphan banner true; raw 1,037,616 B, claude 10, `Co-Authored-By` 2 | dom 14,537; claude 2 — contexts still `…Commit 3c28575 | ShreshthaDas-1999 | and | claude | committed` and the trailer line; `trailer_visible` true; orphan banner true; raw 1,037,558 B, claude 10, `Co-Authored-By` 2 | **no change** |
| `/commit/25661e2` | dom 14,362; claude 0; trailer false; banner false; raw 1,036,201 B, claude 0, `Co-Authored-By` 0 | dom 14,362; claude 0; trailer false; banner false; raw 1,036,143 B, claude 0, `Co-Authored-By` 0; author chips = `['ShreshthaDas-1999']` only | no change — control stays a true zero |
| `/commits/main-2/` | dom 4,004; 36 `committed` rows; branch label `main-2`; doc commit present; claude 0 | dom 4,004; 36 rows; branch label `main-2`; doc commit present; claude 0; `and claude committed` false; hovercard logins = `ShreshthaDas-1999`, `swarnajitdas108` | no change |

## Local state during the pass (read-only)

`for-each-ref`: `main` = `main-2` = `e60538a`, `pre-rewrite-main` = `6ab1e58`, `origin/HEAD` = `origin/main` =
`8ba505b`, `origin/main-2` = `e60538a`; `status --porcelain --untracked-files=all` = 0 lines; `artifacts/` differs
from `main-2` by 0 paths. No push, fetch, checkout or write was performed by this pass.

## What this pass adds

- A second independent observation of every surface ~17 min after the push: **nothing moved**, and the branch-derived
  payloads did not even drift in byte count.
- Two fresh 404s on `graphs/contributors-data` (18:10Z and 18:24Z) against 200 responses earlier the same day. It is
  therefore excluded from the supporting proof and recorded only as an unstable surface. `/stats/contributors` is the
  stable equivalent and returned 200 with 30 + 5 both times.
- Still **not** evidence about mechanism. Two observations are compatible with everything measured: a stored
  repository-level computation refreshed on some schedule or trigger we have not exercised, and a computation over all
  reachable objects that still includes the orphaned original root. This pass varied neither the default branch nor
  object reachability.
