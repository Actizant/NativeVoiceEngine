// Reproducible DOM probe for the main-2 experiment. Run via the browser tool's
// evaluate_script, unchanged, at baseline and after the push - the before/after
// numbers only mean anything if the same predicate produced them.
// Returns counts of `claude` plus the positive controls that prove the region was
// actually read (a region with no usernames at all is the wrong layer, not a zero).

// ---- VARIANT 1: repository home, sidebar Contributors region ----
async () => {
  const out = { url: location.href, utc: new Date().toISOString(), variant: "repo-home-sidebar" };
  const heads = [...document.querySelectorAll("h2,h3,summary,span,div")].filter(
    (e) => /^Contributors\b/.test((e.innerText || "").trim()) && (e.innerText || "").length < 40
  );
  let region = heads[0];
  for (let i = 0; i < 5 && region && region.innerText.length < 60; i++) region = region.parentElement;
  const st = region ? region.innerText.replace(/\n{2,}/g, "\n").slice(0, 600) : null;
  out.sidebar_found = !!region;
  out.sidebar_text = st;
  out.sidebar_claude_hits = (st.match(/claude/gi) || []).length;
  out.sidebar_controls = {
    ShreshthaDas_1999: /ShreshthaDas-1999/.test(st),
    swarnajitdas108: /swarnajitdas108/.test(st),
  };
  out.count_label = (st.match(/Contributors\s*\n\s*(\d+)\s*\n\s*\((\d+)\)/) || []).slice(1);
  const dom = document.body.innerText;
  out.dom_len = dom.length;
  out.dom_claude_hits = (dom.match(/claude/gi) || []).length;
  const raw = await (await fetch(location.href, { cache: "no-store" })).text();
  out.raw_len = raw.length;
  out.raw_claude_hits = (raw.match(/claude/gi) || []).length;
  out.raw_username_hits = {
    ShreshthaDas_1999: (raw.match(/ShreshthaDas-1999/g) || []).length,
    swarnajitdas108: (raw.match(/swarnajitdas108/g) || []).length,
  };
  return out;
};

// ---- VARIANT 2: a direct commit page (use for 3c28575... orphan and 25661e2... control) ----
async () => {
  const out = { url: location.href, utc: new Date().toISOString(), variant: "commit-page" };
  const dom = document.body.innerText;
  out.dom_len = dom.length;
  out.dom_claude_hits = (dom.match(/claude/gi) || []).length;
  out.trailer_visible = /Co-Authored-By:\s*Claude/i.test(dom);
  out.orphan_banner = /does not belong to any branch/i.test(dom);
  out.claude_contexts = [...dom.matchAll(/claude/gi)].map((m) =>
    dom.slice(Math.max(0, m.index - 70), m.index + 40).replace(/\n/g, " | ")
  );
  const raw = await (await fetch(location.href, { cache: "no-store" })).text();
  out.raw_len = raw.length;
  out.raw_claude_hits = (raw.match(/claude/gi) || []).length;
  out.raw_trailer_hits = (raw.match(/Co-Authored-By/g) || []).length;
  out.og_commit_sha = (raw.match(/"oid":"([0-9a-f]{40})"/) || [])[1] || null;
  return out;
};

// ---- VARIANT 3: the branch selector popover, to prove main-2 is offered as a ref ----
// open the "Branch: main" widget first, then run:
async () => {
  const out = { url: location.href, utc: new Date().toISOString(), variant: "branch-menu" };
  const dom = document.body.innerText;
  out.main2_offered = /main-2/.test(dom);
  out.branch_entries = [...dom.matchAll(/\bmain(-2)?\b/g)].length;
  out.default_marker = (dom.match(/default/gi) || []).length;
  return out;
};

/*
BASELINE, 2026-10-22 (this host has no screenshot capture, so innerText is the evidence):
  repo home  17:49:52.967Z  sidebar "Contributors 3 (3)" = ShreshthaDas-1999, swarnajitdas108, claudeClaude
             claude hits 2; both real usernames present (controls PASS); raw HTML 365,806 B carries
             claude 0 and 0 usernames -> card is client-rendered, raw zero proves nothing here.
  /commit/3c28575  17:50:06.470Z  dom 14,537; claude hits 2 (byline chips joined "and", + trailer line);
             trailer_visible true; orphan_banner true; raw 1,037,579 B -> claude 10, Co-Authored-By 2,
             oid 3c2857597f323227b4946ef10b073d760775a32a (this page's raw layer DOES carry the fact).
API at the same minute: /contributors = 2 logins, claude 0. /stats/contributors = 2 entries
  (swarnajitdas108 5, ShreshthaDas-1999 30; total 35), claude 0. /graphs/contributors-data = 404 today
  though it returned 200 with the same 30+5 figures earlier the same day -> unstable, excluded from
  evidence. So the rendered card's third entry is the ONLY surface reporting claude; branch-derived
  surfaces already agree with the corrected history.
*/
