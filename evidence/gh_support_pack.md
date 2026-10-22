# GitHub Support — evidence pack and request (FINALIZED DRAFT, awaiting owner approval)

**Status: PREPARED, NOT SUBMITTED.** No contact has been made with GitHub. Nothing in this file has been sent.
Every figure below carries the UTC minute in which it was captured. Four capture passes are cited:
**baseline 17:47–17:50Z**, **T+0 18:10–18:12Z**, **T+15 18:24–18:28Z** and a **manual pass 19:13:06–19:14:31Z**, all on
2026-10-22. **The planned T+24 pass did not run and is recorded as missed** (§7); no T+24 figure appears in this pack.
The fourth pass is the documented manual fallback, run at its own actual elapsed offset (**T+63m51s … T+65m16s** from
the 18:09:15Z push) and labelled as such — it is not a T+24 result and does not close the 24-hour gap.

Repository: `https://github.com/Actizant/NativeVoiceEngine`
Default branch: `refs/heads/main` = `8ba505b6c22e83e473187bb4c63b4ab71c8e3c49` (unchanged by every pass)
Test branch: `refs/heads/main-2` = `e60538a2e29a995f1a5ea37a847b5aff690a2790`, parent `8ba505b…`, pushed 18:09:15Z
Rewritten root (published): `25661e2af9b0200f8ac0fca5cd57be7af64ea1fc`
Original root (orphaned by the force-push): `3c2857597f323227b4946ef10b073d760775a32a`
Pre-rewrite tip (orphaned): `6ab1e58631f584df4b06d8046a0738a3655bb52d` — no remote ref names it
Commits orphaned by the force-push: **34**

---

## 1. The change, and what it achieved on the branch

A message-only rewrite removed one line — `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>` — from the
**root** commit's message. Trees, parents, author identities, author dates and committer dates were preserved;
no file content changed. The corrected chain was force-pushed to `main` on 2026-10-22 with the owner's
authorization. The owner states the named assistant did not author the code.

| # | Fact | How measured | Result (capture time) |
|---|---|---|---|
| V-1 | The remote advertises only `main` (plus the new test branch) and never the old chain | `git -c credential.helper= ls-remote` (anonymous) | 17:47Z: `HEAD` + `refs/heads/main` = `8ba505b`. 18:10Z and 18:24Z: `HEAD`, `main` = `8ba505b`, `main-2` = `e60538a`; **no ref names `6ab1e58`**, no tags. Manual pass 19:13Z: identical three refs, still no ref for the old chain. |
| V-2 | Published history is 35 commits | `git rev-list --count`; REST `GET /repos/…/commits?sha=main&per_page=100` | 35 both ways, at 17:47Z, 18:10Z, 18:24Z and 19:13Z. Payload **byte-identical across all four passes: 151,912 B.** |
| V-3 | No published commit carries the trailer | trailer string searched in all 35 messages locally; `claude` searched across the whole REST payload | **0** of 35 in the payloads at every pass. Positive control: the same search over the 34 pre-rewrite messages returns **1**. |
| V-4 | GitHub's attached identities are clean | REST commit list + `GET /repos/…/contributors?per_page=100&anon=0` | authors/committers = `ShreshthaDas-1999`, `swarnajitdas108`; `claude` count **0**; identical at 17:47Z, 18:10Z, 18:24Z and 19:13Z (payload 2,092 B each time). |
| V-5 | The committed-count surface is clean | `GET /repos/…/stats/contributors` | 200 at 18:10Z, 18:24Z and 19:13Z: `ShreshthaDas-1999` 30 + `swarnajitdas108` 5 = 35, `claude` absent, payload 2,274 B every time. |
| V-6 | The published root page is clean | browser `innerText` of `/commit/25661e2…` | 18:11:09Z and 18:26:31Z: 14,362 chars, `claude` 0, `Co-Authored-By` 0, orphan banner false, author chips = `[ShreshthaDas-1999]`. The raw HTML behind it (1,036,143–1,036,201 B) also returns `claude` 0. Manual pass 19:14:05Z: same 14,362 chars, `claude` 0, `Co-Authored-By` 0, banner false, `"oid"` = `25661e2…` (two extra chip/hovercard patterns tried that pass returned empty and are discarded — they prove nothing either way). |
| V-7 | "No file content changed" is measurable, not just asserted | `git rev-list` + `git log --format=%T` in the local clone, both chains compared position by position | 19:12Z: the orphan chain has **34 commits and 34 distinct trees**; the published chain has 35/35. Excluding the documentation tip, the 34 orphan trees and the 34 published trees are **identical in commit order** (`diff` empty), and the two roots share tree `bf289bcae1796633c37b2b7d57064bc757e242d2`. Controls: the distinct-tree count proves the 34 lines are not one repeated value, and re-writing one line of the comparison makes `diff` report exactly that line — so the empty diff is a discriminating result, not a blind match. |

## 2. What still displays the attribution (the reason for this request)

| # | Surface | Measurement | Result |
|---|---|---|---|
| R-1 | **Code-tab sidebar ▸ Contributors card** | browser `innerText` of the repository root page | 17:49:52Z, 18:10:44Z and 18:26:07Z, all identical: heading `Contributors 3 (3)`; entries `ShreshthaDas-1999`, `swarnajitdas108`, **`claude`** (link `/claude`, display name `Claude`). Region proven to be read: both real usernames are present in the same string. The raw HTML (365,785–365,843 B) contains **0** usernames and **0** `claude`, so the card is assembled client-side. Re-read **19:13:31Z** and **19:13:43Z** (`/tree/main-2`): same heading, same three entries, same controls, `sidebar claude` 2 — a fourth pass, ~64 min after the push, still unchanged. |
| R-2 | **Direct pages of the 34 orphaned commits** | HTTP GET of each `/commit/<sha>`, redirects not followed, request-id captured (17:12Z pass); rendered DOM for the original root at 18:10:57Z and 18:26:18Z | **34/34 HTTP 200.** Exactly one body mentions `claude`: `3c28575…` (request-id `2B08:DDE3:196D229:227FBC8:6ADA43F3`). Rendered, that page shows the orphan banner *"This commit does not belong to any branch on this repository, and may belong to a fork outside of the repository."*, the byline **"Commit 3c28575 · ShreshthaDas-1999 and claude committed"**, a `claude` co-author chip, and the message line `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>` (`dom_len` 14,537 and `raw_len` ≈1,037,600 B identical across both passes; raw carries `claude` 10 and `Co-Authored-By` 2). Re-rendered **19:13:53Z**: `dom_len` 14,537, `claude` 2, banner true, trailer visible, raw 1,037,611 B with `claude` 10 / `Co-Authored-By` 2, and the page's own `"oid"` still `3c28575…`; the published-root control `/commit/25661e2…` at **19:14:05Z** remains `claude` 0 / `Co-Authored-By` 0 with `"oid"` `25661e2…`. |

Two layers, both true, and this pack separates them: the **raw HTML** of `/commit/3c28575…` does not contain the
"and claude committed" pair line, while the **rendered page** does — that line is injected client-side. An earlier
draft of this pack stated the pair line "was not found"; that was a raw-fetch observation written as if it were a
rendered one, and it is corrected above.

## 3. What the owner has already tried, and what that experiment actually shows

`main-2` was created from the published tip and pushed at 18:09:15Z carrying one documentation-only commit
(`e60538a`, 3 doc paths, 6 objects, no source/test/artifact change).

* **Observed:** the Contributors card is **byte-for-byte the same on `/` and on `/tree/main-2`** — same
  `Contributors 3 (3)` heading, same three entries including `claude`, same 30,724/30,833-char pages (18:11:22Z
  and 18:26:42Z; re-read 19:13:31Z/19:13:43Z, same two page lengths). The card therefore does **not** depend on
  which branch is being viewed.
* **Observed:** the Insights ▸ Contributors page identifies its own scope in its heading —
  *"Contributions per week to main, excluding merge commits"* (18:27:50Z). **No contributor count is taken from
  that page**: its contributor list never populated (see below), so the only count stated anywhere in this pack is
  the REST `/stats/contributors` figure in V-5, and it is labelled as a REST figure, not as what the page shows.
* **Not claimed:** that this identifies the mechanism. The card is repository-level; whether it is a stored
  computation refreshed on a schedule or trigger, or a computation over all reachable objects that still includes
  the orphaned original root, is exactly what we are asking. Creating or removing a branch varies neither the
  default branch nor object reachability, so this experiment cannot decide between those, and we do not present it
  as if it did.
* **Also not claimed:** anything from the Insights contributor list itself. During 33 seconds of observation
  (18:27:50Z → 18:28:23Z) the chart rendered but its contributor list never populated: 901 chars of text, zero
  user-link elements, and the positive control (either real username) **failed**. A zero from a region that was
  never read is not evidence, so that surface is excluded.

## 4. Surfaces deliberately kept out of the proof

* `GET /repos/…/graphs/contributors-data` — returned **200** with 30 + 5 earlier on 2026-10-22, then **404 Not
  Found** (106-byte error body) at 17:47Z, 18:10:35Z (request-id `8143:2B5D1F:21AA26D:305C3F3:6ADA515C`),
  18:24:47Z (request-id `6AF1:1922A7:22603C3:313763D:6ADA54F4`) and 19:13:06Z (request-id
  `1294:AC8E8:2316749:3262FF0:6ADA603F`). Same-day, same-repository, same request shape, four consecutive 404s: an
  intermittently unavailable endpoint is not cited as proof of anything. The stable
  `stats/contributors` route (V-5) is used instead. It needs `Accept: application/json`; text/html returns 400.
* **Not claimed:** that the orphaned commits are still retrievable by `git fetch`/`git cat-file` from the remote.
  Only HTTP 200 on the web pages and API responses was verified; object-level fetchability by SHA was not tested.
* **Not claimed:** that the card is a cache, or that any particular GitHub subsystem is at fault.
* **Not claimed:** that fixing the display requires deleting the repository, its issues or any branch. The owner
  wants the minimum GitHub-side action.
* **Not a licence or ownership dispute.** The repository's own audit record documents the correction:
  `docs/audit-register.md` Addendum 28 — which carries the old→new commit map and, in its own words, "the limits of
  what the correction proves" — and row A73, which records the reference hygiene the rewrite created.

## 5. Raw measurement files

* `scratch/attrib/gh_support_evidence.txt` / `_raw.txt` — E-1 refs, E-2 trailer census on both chains,
  E-3 all 34 orphan page statuses with request ids, E-4 orphan API sample.
* `scratch/attrib/main2_baseline_out.txt`, `main2_verify_T0.txt`, `main2_verify_T15.txt` — the three REST/git passes.
* `scratch/attrib/main2_measure_T0.md`, `main2_measure_T15.md` — baseline→T+0→T+15 tables for the rendered surfaces.
* `scratch/attrib/main2_verify_manual_T63m.txt` (17 PASS / 0 FAIL, impossible-SHA control included) and
  `main2_measure_manual_T63m.md` — the **manual fallback pass** at T+63m51s…T+65m16s, with its own UTC stamps and the
  positive controls listed per surface. It is cited as a fourth observation, not as the scheduled T+24 run.
* `scratch/attrib/main2_dom_probe.js` — the exact predicates used at every pass, so before/after numbers are
  comparable; browser captures are quoted inline above with UTC stamps, since an HTML fetch cannot substitute for
  the rendered view.
* `scratch/attrib/CHECKPOINT_2026-10-22T19-01Z.md` — the frozen handover record: verified refs, evidence fingerprints,
  and the manual procedure for the measurement that §7 records as missed.

**Where these files live.** Everything under `scratch/` is on this workstation, inside the Qoder chat workspace
`[USER_HOME]\Documents\Qoder\2026-10-14\e14afe8e` — **outside** the `Actizant/NativeVoiceEngine` repository
directory, so no measurement report here can be committed or pushed with repository work by accident. The paths quoted
above resolve only on this machine; they are cited as provenance for the figures, not as a link GitHub can follow.

---

## 6. FINALIZED REQUEST — edit only with the owner's approval; do not send before then

> **Subject:** Contributor card and orphaned commit pages still show a co-author removed by a history rewrite
>
> Hello GitHub Support,
>
> In repository `Actizant/NativeVoiceEngine` I removed an incorrect `Co-Authored-By: Claude` trailer from the root
> commit by a message-only rewrite of the history — no file content changed, and author, committer and dates were
> preserved — and force-pushed the corrected chain to the default branch `main` on 22 October 2026. `main` is now
> `8ba505b6c22e83e473187bb4c63b4ab71c8e3c49`; the rewritten root is `25661e2af9b0200f8ac0fca5cd57be7af64ea1fc`;
> the original root is `3c2857597f323227b4946ef10b073d760775a32a`. That push orphaned 34 commits, and no remote ref
> now names the old chain.
>
> **Verified correct, measured three times on 22 October (17:47, 18:10 and 18:24 UTC):**
>
> * Anonymous `git ls-remote` advertises only `main` (plus a test branch, see below) — never the old chain.
> * The REST API returns 35 commits for `main`, with a payload identical across all three passes, and no `claude`
>   author, committer or trailer anywhere in it.
> * `/repos/.../contributors` lists only my two accounts; `/repos/.../stats/contributors` totals 35 commits across
>   those same two accounts.
> * The page for the published root, `/commit/25661e2…`, shows one author and no co-author.
>
> **Still showing the removed attribution:**
>
> 1. The **Contributors card in the code-tab sidebar** reads `Contributors 3 (3)` and names `claude` (`/claude`) as
>    the third entry. It was unchanged at 17:49, 18:10 and 18:26 UTC.
> 2. The **direct pages of the orphaned commits still return HTTP 200** — all 34, checked individually — and
>    `/commit/3c28575…` still renders "ShreshthaDas-1999 and claude committed", a `claude` co-author chip and the
>    `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>` line, under the banner "This commit does not belong to
>    any branch on this repository."
>
> **What I have already ruled out myself.** I created and pushed a second branch, `main-2`, based on the published
> tip and carrying only a documentation commit. The sidebar card renders **identically on `/` and on
> `/tree/main-2`**, so it is a repository-level figure and not a per-branch one — no branch I add or remove can
> change it. I make no claim about why; I mention it so you do not suggest a step I have already measured. For the
> same reason I am not quoting a contributor count from the Insights ▸ Contributors page: that page identifies its
> own scope as "Contributions per week to main, excluding merge commits", but its contributor list did not populate
> while I observed it, so I have no rendered figure from it to compare against the sidebar card.
>
> **Questions:**
>
> * Is the sidebar Contributors card served from a cache or a separately computed index? If it can be refreshed or
>   invalidated for this repository, how do I request that?
> * If it is computed over all reachable objects rather than over `main`, would removing the orphaned objects change
>   it — and is there a supported request for that garbage collection or object removal, or is their continued
>   visibility at a direct SHA permanent by design?
> * If neither is self-service, which team handles this, so we can route the request correctly?
>
> I am not asking to delete the repository, its issues or any branch. I am asking for the minimum GitHub-side action
> that stops an attribution which this repository's own history no longer contains from being displayed.
>
> Thank you — Shreshtha, owner of `Actizant/NativeVoiceEngine`.

---

## 7. The scheduled T+24 pass — **NOT RUN, RECORDED AS MISSED**; manual fallback run instead (as at 2026-10-22 19:14:31 UTC host clock)

This pack contains **no T+24 figure**, and none will be added under the label "T+24". The owner returned the machine
at 19:01:49Z and both scheduled tasks were disabled first, so the pass could not execute unattended:

* automation `a385356c-c26f-4b2f-8df7-4175ecc3cf6a` (the T+24 measurement, trigger 2026-10-23T18:12:00Z) — final state
  `enabled: false`, `pauseReason: "manual"`, `nextRunAt: null`, **`lastRunAt: null`**, `lastRunStatus: null`,
  `lastError: null`, `consecutiveFailures: 0`;
* follow-up automation `67734def-21ef-45ec-8ebd-9593b6419fb5` (its verification pass, trigger 2026-10-23T20:00:00Z) —
  same state, `lastRunAt: null`;
* neither output file exists: `scratch/attrib/main2_verify_T24.txt`, `scratch/attrib/main2_measure_T24.md`.

So no observation exists for the 24-hour point on either surface, and nothing in §1 or §2 is asserted about
2026-10-23 or later. The card result in R-1 is stated as of its own capture minutes, not as a standing property.

**The manual fallback was then executed on this same host, before it was handed back.** It met the three-part
acceptance test exactly: (1) it is recorded at its own measured offset — REST pass ended 19:13:06Z, rendered surfaces
19:13:31.054Z–19:14:31.295Z, i.e. **T+63m51s … T+65m16s** from the 18:09:15Z push, and it is not called T+24 anywhere;
(2) `main2_verify_manual_T63m.txt` holds a 17 PASS / 0 FAIL block **including** the impossible-SHA control (`404`) and
the still-`404` unstable-graph note; (3) `main2_measure_manual_T63m.md` carries the rendered table **with its positive
controls** — both real usernames inside the read card region on two pages, and 36 `committed` rows plus DOM usernames
`ShreshthaDas-1999` 30 / `swarnajitdas108` 5 and the full corrected SHAs on the list page, which is what licenses the
`claude` 0 there.

**Outcome of that pass: nothing moved.** The card still reads `Contributors 3 (3)` naming `claude`, and `/commit/3c28575…`
is still served, still orphan-bannered, still carrying the trailer in both layers. Two fields the pass probed
additionally (`author_chips`, `/users/*/hovercard` on the commit pages) returned empty and are discarded as vacuous; the
conclusions rest only on the verbatim predicates and each page's own `"oid"`.

Because the card is unchanged, the pack edit is the timestamp extension already made in §2 R-1 and R-2 and the §5 file
list. **§6 has deliberately been left exactly as approved** — its request text still says "measured three times
(17:47, 18:10 and 18:24 UTC)" and "unchanged at 17:49, 18:10 and 18:26 UTC". If the owner wants the fourth observation
in the request itself, the change is two clauses (add 19:13 UTC to each list and "four times" for "three times"); it is
not applied until he approves it, since §6 is the text he reviews.

The procedure, the current verified ref set and the evidence fingerprints are recorded in
`scratch/attrib/CHECKPOINT_2026-10-22T19-01Z.md`.
