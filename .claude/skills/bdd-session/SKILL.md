---
name: bdd-session
description: Agree on WHAT the next session builds — example-map the story, then formulate it as a Gherkin feature file.
disable-model-invocation: true
---

The deliverable of this session is a `.feature` file we both agree states WHAT to build. This session formulates; the *next* session implements, starting by reading that file. Gherkin is our shared language — the file is the contract, not a test artifact.

## 1. Frame

State the story back to me in one sentence and wait for confirmation. If a *fact* can be found in the repo (existing behaviour, current code, prior features), look it up rather than asking. The *decisions* are mine.

## 2. Example-map

Interview me one question at a time, grilling style — one question, your recommended answer, wait. Cards, not syntax: no Gherkin is written during this step. Build three lists as we go:

- **Rules** — business rules / acceptance criteria (blue)
- **Examples** — concrete cases, each illustrating exactly one rule (green)
- **Red cards** — open questions and assumptions (red)

Read the map continuously and act on its signals:

- A rule with no example → under-specified: propose an example and ask me to accept or correct it.
- An example that fits no rule → a missing rule: name the rule it implies and put it to me.
- Red cards piling up → story not ready: stop, show me the red cards, ask which to resolve now and which to cut from scope.
- Rules piling up → story too big: propose a split and ask which slice this session's file covers.

This step is done when every rule has at least one example and the red-card list is empty — each one resolved into a rule/example or explicitly cut from scope.

## 3. Formulate

Now turn the map into Gherkin. **Declarative, not imperative** — every step describes behaviour; if the wording would change when the implementation changes, rewrite it (`When "Bob" logs in`, never fields and buttons).

- One `Feature:` per file; one `Rule:` block per blue card; one `Example:` per green card, titled by its outcome ("Decline without overdraft").
- Given = state, exactly one When = the trigger, Then = observable outcome. 3–5 steps; a longer scenario is hiding imperative steps or a second behaviour — split it.
- Data appears only when it illustrates the rule; incidental fixtures stay out.
- `Scenario Outline` only for the same rule at different boundaries, never as a loop over unrelated cases.

Show me the file. Iterate until I confirm it states exactly what to build — do not act on it before that confirmation.

## 4. Hand off

Write the agreed file to `features/<story-slug>.feature` (or wherever this repo already keeps feature files). Close by stating that the next session starts from this file; implementation belongs to that session, not this one.
