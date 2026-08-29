---
name: ci-watch
description: "One round of the CI loop: find the PR's GitHub Actions run, wait for it to settle, classify a failure by cause, fix only what this change caused, and -- when the caller authorizes it -- merge on green and decide whether to watch main. Used by the process-issues workflow's CI stage."
disable-model-invocation: true
---

You are one round of a bounded loop on GitHub Actions for a PR. The caller names the PR, the branch, the issue, the round number, and whether merging is authorized this round.

## STEP 1 -- find the run and wait for it to settle

- `gh run list --branch <branch> --limit 5 --json databaseId,status,conclusion,event,headSha,url`
- Take the newest run whose headSha matches the branch tip (`git rev-parse HEAD` after `git fetch origin`).
- Then wait for it with ONE foreground call -- do not hand-roll a sleep loop: `gh run watch <id> --interval 30 --exit-status` in a single Bash call with an explicit timeout of 1500000 ms (25 minutes); the run itself takes around 11. A non-zero exit means it did not succeed -- read the conclusion, do not retry the watch.
- If NO run appears within about 3 minutes, stop and set outcome="no-run". Say whether .github/workflows/ci.yml actually exists on the base branch -- if it does not, no push can ever trigger a run, and that is the finding, not a failure of this PR.

## STEP 2 -- green

If the conclusion is success, set outcome="green". If merging is authorized, continue to STEP 4 -- do not stop here. Otherwise you are done: change nothing.

## STEP 3 -- red

Read the actual log before touching anything: `gh run view <id> --log-failed` (fall back to `gh run view <id> --log` if that is empty). Record every failing job/step under failing_jobs with the error line, not a paraphrase.

Then judge WHY it failed, because the three causes need different responses:

a) OUR CODE is wrong -- a real failure this change caused. Fix the code.
b) The CI CONFIG is wrong, or the check is flaky/infra (a runner timeout, a network blip, a cache miss). Do NOT weaken the check to get green. Re-run it once with `gh run rerun <id> --failed` and see. If it fails the same way twice, it is not flaky -- report it as still-red with what you found.
c) The failure is PRE-EXISTING on the base branch and nothing to do with this change. Say so, and do not paper over it here.

Fix only case (a). Then commit and push: `git add -A`, commit with a concise lower-case imperative subject, no author trailer, hooks NOT bypassed, then `git push origin <branch>` (the caller states the pre-push hook's timeout rules -- follow them). Set outcome="fixed-and-pushed" so the next round picks up the new run. If you cannot fix it, outcome="still-red".

Anything you learn that is NOT this failure -- a check that should exist and does not, a flaky test, a config smell -- goes under "discoveries" for filing, not into this diff.

## STEP 4 -- merge (ONLY when the caller authorized it AND STEP 2 concluded green)

Never merge any other outcome.

a. `gh pr ready <pr>` -- take it out of draft, a draft PR cannot be merged.
b. `gh pr merge <pr> --squash --delete-branch`. Squash: one issue, one commit on main. If the merge is refused (conflict, protected branch, not mergeable), do NOT force anything: set merged=false and say exactly what GitHub refused.
c. Confirm the issue closed itself from the "Closes" line: `gh issue view <n> --json state`. If it is still open, leave it open and note that under issue_closed -- do not close it by hand, that hides a broken link.
d. Watch main's CI per the mode the caller names:
   - **auto**: decide from the trees, not from opinion. `git fetch origin`, then compare `git rev-parse origin/main^{tree}` against `git rev-parse HEAD^{tree}` (HEAD is the branch tip the PR's CI tested). IDENTICAL: set tree_matches_ci=true and STOP -- leave post_merge_green unset; the squash landed the exact tree the PR's CI just proved green, so main's run re-tests a proven tree and waiting for it buys nothing. DIFFERENT: set tree_matches_ci=false and watch as in "always".
   - **always**: the merge pushes to main, which triggers CI again. Find that run with `gh run list --branch main --limit 3 --json databaseId,status,conclusion,url`, wait for it with `gh run watch <id> --interval 30` (one foreground call, explicit 1500000 ms timeout) and report post_merge_green. If it went red, say so plainly and do NOT try to fix main here -- a red main is a human decision (revert or roll forward), not something to improvise at the end of an automated run.
   - **never**: do NOT wait for main's CI. Leave post_merge_green unset: nothing in this run acts on it, and the PR's own CI already went green on this exact tree. Stop once the merge is confirmed.

Do not tag, do not release, do not bump a version, do not upload anything.
