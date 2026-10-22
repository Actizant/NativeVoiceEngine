# main-2 experiment — measurement record

Push executed: `git push origin refs/heads/main-2:refs/heads/main-2`
- exit 0, single attempt, no retry, no `--force`, no `--all`, credential helper left at the configured
  `manager` (deliberately NOT overridden with `-c credential.helper=` — that form is for anonymous reads)
- remote output verbatim: `* [new branch]      main-2 -> main-2` + the PR-creation hint lines
  (`main2_push_attempt1.stderr`, 266 B; stdout empty; `main2_push_pre.txt` holds the pre-push state)
- T0 clock: push accepted ~18:09:15Z (`pushed_at` from the API), DOM probes 18:10:44Z–18:12:32Z
  → T+0 is ≈ 2 minutes after the push, not 0.

## 1. REST / git-derived surfaces — verifier `main2_verify_after.py`, output `main2_verify_T0.txt`

17/17 assertions PASS. Baseline = `main2_baseline_out.txt` at 17:47–17:49Z.

| Surface | Baseline (pre-push) | T+0 | Change |
|---|---|---|---|
| anonymous `ls-remote` | `HEAD`, `refs/heads/main` = `8ba505b` | `HEAD` = `8ba505b`, `refs/heads/main` = `8ba505b`, `refs/heads/main-2` = `e60538a` | **main-2 added; main byte-identical** |
| `/repos/.../branches` | `['main']` | `['main','main-2']` | main-2 listed |
| `/repos/{REPO}` meta | `default_branch='main'`, `pushed_at=16:07:49Z` | `default_branch='main'`, `pushed_at=18:09:15Z` | **default branch unchanged** |
| `/commits?sha=main` | count 35, bytes 151,912, `claude` 0, `co-authored-by` 1, logins 2 | count 35, bytes 151,912, `claude` 0, `co-authored-by` 1, logins 2 | **byte-identical payload** |
| `/commits?sha=main-2` | HTTP 404 | count 36, tip `e60538a`, `claude` 0, `co-authored-by` 1, logins 2 | new; claude-free |
| `/git/commits/e60538a` | HTTP 404 | HTTP 200, parents `['8ba505b']`, `claude` 0 | doc commit now served |
| `/contributors?anon=0` | `['ShreshthaDas-1999','swarnajitdas108']` | same, `claude` in payload 0 | unchanged |
| `/stats/contributors` | (baseline: 30 + 5) | 2 entries: `swarnajitdas108` 5, `ShreshthaDas-1999` 30, `claude` 0 | unchanged; total still 35 = main only, **not 36** |
| `/graphs/contributors-data` | 404 (200 earlier the same day) | 404 | unstable — recorded, excluded from evidence |
| orphan root `/git/commits/3c28575` | 200, `parents=[]`, message tail carries `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`, `claude` 1 | 200, parents 0, trailer present | **still served, still orphaned** |
| corrected root `/commits/25661e2` | 200, `claude` 0 | 200, `claude` 0 | unchanged |
| control SHA `000…000` | 404 | 404 | the probe still discriminates |

The `co-authored-by 1` on both branch commit lists is **not** a trailer on any commit: it is the token
appearing inside a commit *message body* in the history. It is the same count as baseline, and `claude` is 0
in the same payloads.

## 2. Rendered surfaces — predicates from `main2_dom_probe.js`, unchanged from baseline

| Page | Baseline | T+0 | Change |
|---|---|---|---|
| repo home sidebar | `Contributors 3 (3)` = ShreshthaDas-1999, swarnajitdas108, **claudeClaude**; sidebar claude hits 2; controls true; raw HTML claude 0 (365,806 B) | **`Contributors 3 (3)` with `claudeClaude`**; sidebar claude hits 2; both controls true; raw HTML claude 0, usernames 0 (365,843 B) | **UNCHANGED — attribution still displayed** |
| `/tree/main-2` sidebar | not measured at baseline (branch did not exist) | same card: `3 (3)`, `claudeClaude`, hits 2, controls true | card is **repo-level, not branch-scoped** |
| `/commit/3c28575…` (orphan root) | dom 14,537; claude 2 (`…ShreshthaDas-1999 | and | claude | committed` + trailer line); `trailer_visible` true; `orphan_banner` true; raw 1,037,579 B, claude 10, `Co-Authored-By` 2 | dom 14,537; claude 2 with the same two contexts; `trailer_visible` true; `orphan_banner` true; raw 1,037,616 B, claude 10, `Co-Authored-By` 2, oid `3c28575…` | **UNCHANGED** |
| `/commit/25661e2…` (clean root) | dom ~14.4 k; claude 0; trailer false; orphan false; raw claude 0 | dom 14,362; claude 0; `trailer_visible` false; `orphan_banner` false; raw 1,036,201 B claude 0, `Co-Authored-By` 0 | **UNCHANGED (control)** — a zero here is a real zero |
| `/commit/e60538a…` (doc commit) | 404 — page did not exist | dom 26,679; claude 2 — **both inside our own documentation text** ("…that Claude did or did not write any line of…"); `trailer_visible` **false**; `orphan_banner` false; raw 336,159 B claude 0, `Co-Authored-By` 3 | new page; the token is prose, not a byline |
| `/commits/main-2/` | — | 36 "committed" rows read, branch selector labelled `main-2`, doc commit present; **claude hits 0** | branch list page claude-free |
| `/commits/main/` | — | 36 "committed" rows read, branch selector `main`; **claude hits 0**, `and claude committed` false | unchanged |

Local note the verifier did not anticipate: the push created `refs/remotes/origin/main-2` in the clone
(push writes back its own tracking ref; no fetch was run). `refs/remotes/origin/main` is still `8ba505b`,
so the script's "origin/* must still be 8ba505b" line is accurate for `main` only.

## 4. Post-push integrity of the local store and the exact remote delta

- `git status --porcelain --untracked-files=all` in the clone: 0 lines. HEAD still `refs/heads/main`.
- `git diff --name-only 8ba505b refs/heads/main-2` → exactly `DECISIONS.md`, `PROGRESS.md`,
  `docs/audit-register.md`.
- Objects the push made newly reachable on the remote (`rev-list --objects main-2 ^8ba505b`), counted: **6** —
  1 commit (`e60538a`), 2 trees (root `976d56384c1e`, `docs` `7eeeadecc451`), 3 blobs
  (`f9cc6114ec65`, `c9b650b6c5e3`, `9221d8cfa497`). Matches the blast-radius figure predicted in
  `main2_plan.md`. No source, test, tool or artifact object was transferred: those tree entries are inherited
  unchanged from `8ba505b`.
- `refs/remotes/origin/main-2` was created locally **by the push itself** (git writes back the ref it pushed);
  `refs/remotes/origin/main` and `origin/HEAD` still `8ba505b`. No fetch was run.
- Preserved artifacts re-checked after the push: `pre-rewrite-main.bundle` 905,415 B, sha256 prefix
  `0212a8ddefa10988…` — unchanged.

## 3. What is confirmed vs. hypothesis

CONFIRMED (measured twice, same predicates):
- The push reached the remote and created exactly one new ref; `main`, the default branch, `refs/heads/main`'s
  value, and main's commit-list payload are byte-unchanged.
- Every branch-derived and API surface reports zero `claude` for both branches, and did before the push.
- The rendered Contributors card still lists `claude` as a third contributor, and the direct page of the
  orphaned original root still renders the "…and claude committed" byline and the trailer.
- The card does not vary with the branch being viewed (`/tree/main-2` shows the identical card), so it is
  not a per-branch computation.

HYPOTHESIS, NOT ESTABLISHED BY THIS EXPERIMENT:
- *Why* the card still shows `claude`. Two candidate explanations remain live and this push cannot separate
  them: (a) a stored computation that refreshes only on the default branch or on a schedule, and (b) a
  repository-wide computation over every reachable object, in which case the orphaned original root — still
  served by the remote at its direct URL, with no remote ref naming it — keeps contributing the co-author.
  The push is not a default-branch change, so it cannot test (a); it does not delete the orphan object, so it
  cannot test (b).
- Note (b) as stated already survived the 2026-10-22 `main` force-push, which left the same 34 orphans
  reachable by URL — so (b) is not newly created by this push and is not what this experiment varies.
- Any claim about GitHub's cache TTL or refresh cadence: no observation of a refresh has occurred yet.
- Whether the T+24 pass differs from T+0: pending, not assumed. The T+24 pass is scheduled as a local
  automation (`a385356c-c26f-4b2f-8df7-4175ecc3cf6a`, runs 2026-10-23T18:12:00Z ≈ push+24h02m, read-only
  constraints baked into its prompt); the T+15 pass ran from this conversation.
