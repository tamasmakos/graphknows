# Feature Specification: Air-Gapped Delivery

**Feature Branch**: `001-air-gapped-delivery`

**Created**: 2026-09-02

**Status**: Draft

**Input**: User description: "Air-gapped delivery of GraphKnows to a client: ship a private versioned wheel as a GitHub Release asset instead of publishing to PyPI; add a client-built release image that installs the wheel by exact version and copies the operational scripts from the build context; enforce offline through image environment plus a boot-time check that the baked models are present; split an ontology extra off the LLM-extraction extra; and stamp graph namespaces with a schema version so a mismatched graph refuses to load rather than serving stale ids. Requested in issue #326."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Run the memory service with no network (Priority: P1)

An operator at a client site brings the deployment stack up on a host that cannot reach the
internet. The service starts, accepts a document, and answers a search over it. Nothing it
does requires an outbound connection, and nothing silently degrades because a resource could
not be fetched.

**Why this priority**: This is the product promise. Every other story here is delivery
plumbing around it; without this one there is nothing worth delivering. It is also the story
the client will test first, in the environment where failure is least recoverable.

**Independent Test**: Bring the stack up on a host with outbound network blocked, ingest a
document, run a search, and confirm hits come back — while observing that no external request
was attempted.

**Acceptance Scenarios**:

1. **Given** a host with all outbound network blocked except the graph database, **When** the
   operator starts the deployment stack, **Then** every service reaches a healthy state without
   attempting a single external connection.
2. **Given** the stack is healthy and offline, **When** the operator ingests a document and then
   searches for a phrase from it, **Then** the search returns hits drawn from that document.
3. **Given** the stack is offline, **When** any part of the memory pipeline needs a language
   model or corpus, **Then** it uses the copy delivered with the deployment and never reaches
   for a remote one.

---

### User Story 2 - Receive the software privately (Priority: P2)

The team cuts a release. The client receives a specific, named version of the software through
a channel only they can reach. At no point is the software listed on a public package index.

**Why this priority**: The client relationship depends on the software not being public. It
ranks below the offline guarantee only because a delivery channel with nothing working at the
end of it delivers nothing.

**Independent Test**: Cut a release, confirm the versioned artifact appears on the private
channel, and confirm a search of the public package index finds nothing.

**Acceptance Scenarios**:

1. **Given** a maintainer publishes a release for a specific version, **When** the release
   completes, **Then** the versioned artifact is available to authorized recipients and to
   nobody else.
2. **Given** a release has been published, **When** anyone searches the public package index for
   the project, **Then** no artifact for it is listed.
3. **Given** a release artifact and a version number, **When** the recipient builds the
   deployment, **Then** the running system reports exactly that version.

---

### User Story 3 - Be stopped before a stale graph is served (Priority: P3)

An operator upgrades to a newer version and points it at a graph built by an older one. Rather
than answering with records whose identity no longer means what it did, the system refuses to
open the graph and says what to do about it.

**Why this priority**: This is the only story here whose failure mode is silent and wrong
rather than loud and absent. A refused start is an afternoon; answers quietly drawn from a
mis-keyed graph can go unnoticed indefinitely and poison decisions made on them.

**Independent Test**: Build a graph on one version, upgrade, open the same graph, and confirm
the refusal names both versions and the remedy.

**Acceptance Scenarios**:

1. **Given** a graph written by a different schema version, **When** the operator starts the
   service against it, **Then** it refuses to proceed and names the stored version, the running
   version, and how to recover.
2. **Given** an empty namespace, **When** the service opens it for the first time, **Then** it
   records the schema version in use and proceeds normally.
3. **Given** a graph whose schema version matches the running software, **When** the service
   opens it, **Then** it proceeds without comment.
4. **Given** two processes open the same brand-new namespace at the same moment, **When** both
   attempt to record the schema version, **Then** exactly one recording survives and neither
   process fails.

---

### User Story 4 - Bring your own ontology without the LLM stack (Priority: P4)

An operator wants the system to use their organisation's own formal ontology. They install
ontology support and get it, without also installing the machinery for language-model-driven
extraction, which they cannot use offline and may not be permitted to hold.

**Why this priority**: It is the smallest of the four and the only one with a workaround
(install everything and ignore half). It still belongs in this release because the workaround
means shipping an air-gapped client the LLM stack they cannot run.

**Independent Test**: Install with ontology support in a clean environment, load an ontology
file, and confirm the LLM-extraction dependency is absent.

**Acceptance Scenarios**:

1. **Given** a clean environment, **When** the operator installs with ontology support, **Then**
   they can load and project their own ontology, and the LLM-extraction dependency is not
   installed.
2. **Given** a base installation with no extras, **When** something tries to use an ontology
   feature, **Then** the error names the ontology option specifically, not the
   LLM-extraction one.
3. **Given** an installation with LLM-extraction support, **When** the operator uses that path,
   **Then** it still works exactly as before.

---

### Edge Cases

- **A model or corpus is missing from the deployment.** Startup stops immediately and names
  what is absent, rather than starting healthy and failing later in the middle of an ingest.
- **The graph database is unreachable.** Startup fails inside the health-check window with a
  message naming the database — never a hang, and never an empty search result that looks like
  "no matches".
- **A graph exists but carries no schema version**, because it predates stamping. Treated as a
  mismatch and refused, not assumed compatible.
- **The operator enables a network-dependent capability** (LLM-assisted extraction, or a remote
  embedding service) inside an air-gapped deployment. The documentation states plainly that
  these sit outside the guarantee; enabling them is the operator's decision and their
  consequence.
- **The build host has no network.** The build is the one step that legitimately needs it.
  Failure here must say so, so an operator does not mistake it for a broken air-gap.

## Requirements *(mandatory)*

### Functional Requirements

**Offline operation**

- **FR-001**: The deployment MUST complete a full ingest and search cycle with no outbound
  network access other than to its own graph database.
- **FR-002**: The deployment MUST carry every language model and corpus it needs, delivered as
  part of the deployment rather than fetched when first required.
- **FR-003**: The offline guarantee MUST be enforced by the deployment's own configuration, not
  by convention at each point of use, so that a newly added code path cannot silently opt out
  of it.
- **FR-004**: The deployment MUST refuse to start when a required model or corpus is absent, and
  MUST name what is missing.
- **FR-005**: The startup check for missing models MUST add no more than a few seconds to
  startup, and MUST run once per start rather than on a repeating schedule — a repeated deep
  check has been measured to slow ingest several-fold.
- **FR-006**: The development environment MUST remain unaffected by the startup check, since it
  acquires its models lazily and holds none at first start.

**Private delivery**

- **FR-007**: Publishing a release MUST make the versioned artifact available only to authorized
  recipients, and MUST NOT list it on any public package index.
- **FR-008**: The deployment MUST be built from a named, exact version of the released artifact,
  so that what runs can be traced to what was released.
- **FR-009**: The running system MUST resolve to the installed release, not to a working copy of
  the source tree.
- **FR-010**: The recipient MUST be able to build the deployment from the released artifact and
  the accompanying deployment definition alone.
- **FR-010a**: Publishing a release MUST attach a deployment bundle that is self-contained — the
  deployment definition, the operational scripts the build needs, and the artifact — arranged so
  that the build behaves identically whether run from an unpacked bundle or from a source checkout.
  The recipient MUST NOT need access to the source repository.

**Schema safety**

- **FR-011**: Each namespace MUST record the schema version that created it.
- **FR-012**: The system MUST refuse to open a namespace whose recorded schema version differs
  from the running one, and MUST NOT read or write it.
- **FR-013**: A refusal MUST name the recorded version, the running version, and the recovery
  action available to the operator.
- **FR-014**: A namespace with no recorded schema version MUST be treated as a mismatch.
- **FR-015**: The recorded version MUST change whenever the graph's shape changes, without
  depending on anyone remembering to update it.
- **FR-016**: Concurrent first-time access to a new namespace MUST result in exactly one
  recorded version, with no process failing.

**Ontology support**

- **FR-017**: Ontology support MUST be installable without installing LLM-extraction support.
- **FR-018**: An error raised because ontology support is missing MUST name the ontology option,
  not another one.
- **FR-019**: Existing LLM-extraction behaviour MUST be unchanged by the separation.

**Documentation**

- **FR-020**: The deployment documentation MUST state that the build needs network access and
  that the offline guarantee applies to running the system, not building it.
- **FR-021**: The documentation MUST name the capabilities that fall outside the offline
  guarantee, so an operator can avoid them deliberately.
- **FR-022**: The documentation MUST describe the upgrade path, including that a schema change
  requires rebuilding the graph from its original sources.

### Key Entities

- **Release artifact**: A specific, versioned build of the software, delivered privately. Named
  by version; the thing a deployment is built from.
- **Deployment**: The running stack the client operates — the memory service and its graph
  database — built from a release artifact and a deployment definition.
- **Namespace**: An isolated graph belonging to one tenant. Carries the schema version that
  created it.
- **Schema version**: An identifier for the shape of a graph, derived from that shape rather
  than maintained by hand. Two namespaces with different schema versions are not interchangeable.
- **Baked resources**: The language models and corpora delivered inside the deployment, which
  make offline operation possible.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: On a host with outbound network blocked, an operator brings the stack up, ingests a
  document and retrieves it by search, with **zero** attempted external connections.
- **SC-002**: With a required model absent, the deployment fails to start and names the missing
  resource; it never reaches a state where it accepts work it cannot complete.
- **SC-003**: With the graph database stopped, startup fails within the health-check window with
  a message naming the database — no hang, and no empty result presented as a valid answer.
- **SC-004**: A published release is retrievable by an authorized recipient and returns no result
  in a public package index search.
- **SC-005**: Opening a graph written by a different schema version is refused **100%** of the
  time, and the message alone is enough for an operator to recover without support.
- **SC-006**: Installing ontology support in a clean environment yields a working ontology
  import with **no** LLM-extraction dependency present.
- **SC-007**: A recipient with only the release artifact, the deployment definition and the
  documentation can reach a working offline deployment without contacting the maintainers.

## Assumptions

**About the recipient and environment**

- The client operates the deployment themselves; the maintainers do not host it.
- The client's build host has network access once, at build time. The host that *runs* the
  deployment need never have it.
- The client keeps the sources of their ingested data outside the graph, so that rebuilding
  after a schema change is possible. This is a condition of the upgrade path, not an
  expectation the software can enforce.
- The client's engineers can run a container stack and read English documentation.

**Settled decisions carried from issue #326**

These were argued and closed before this specification; they are recorded so the planning phase
does not reopen them.

- Delivery is a private versioned artifact plus a deployment definition the client builds from.
  Not a public package index, not a hosted service, and no network-exposed interface.
- The client builds the deployment image locally. There is no pre-built image transfer, because
  the built image is far larger than the delivery channel's per-file limit.
- Because the exact tested bytes are not what the client runs, the build must be pinned — base
  image and artifact version both — so that two builds of the same release agree.
- Offline is enforced by the deployment's environment. Configuration may *read* that environment
  to produce a helpful error, which is not the same as owning the setting.
- Upgrades are a rebuild from source data, not a migration. No forward-migration machinery.
- LLM-assisted extraction and the remote embedding option remain available and remain outside
  the offline guarantee. They are not removed.

**Scope boundaries**

- The symbolic channel's correctness work and the library-wide silent-degradation policy are
  explicitly out of scope; both are tracked separately.
- No change to retrieval ranking, the evaluation harness, or the development container.

**Governance**

- The project constitution at `.specify/memory/constitution.md` was an unfilled template when this
  specification was written, so no project principles shaped it. It was ratified as **v1.0.0** on
  2026-09-02, and `plan.md` evaluates this feature against it: five principles pass, one is N/A,
  and one deviation is justified in Complexity Tracking. No requirement here changed as a result.
