# **STATEMENT OF WORK (SOW)**

Project Name: “Noa” Agentic Memory Graph POC  
Date: December 24, 2025 (Updated)  
Team: Tamás Diósi-Mákos (“Provider/Lead”), Sebi (“Algorithm Engineer”)

## **1\. Project Background & Vision**

**The Vision:** To build a “Life Graph” that gives the Noa agent autobiographical memory. The system will process timestamped text streams and GPS logs into a structured Knowledge Graph, enabling the agent to recall specific moments, conversations, and contexts.

**Technical Constraint:** The system must utilize a **Hybrid Architecture**:

* **Graph Structure (FalkorDB):** Stores the topology (Nodes/Edges), Time Backbone, and Ontology.

* **Content Store (PostgreSQL \+ pgvector):** Stores the heavy raw text (Chunks) and vector embeddings.

* **The Bridge:** HybridStore (Python Class) manages dual-write consistency:

  * **Nodes**: Uniquely indexed in FalkorDB.

  * **Content**: Text chunks stored in PostgreSQL (pgvector) with metadata pointers.

## **2\. Technical Scope of Work**

### **2.1. Phase 1: Scalability & Data Reality**

To validate economic viability, the Provider will perform a Stress Test on the Hybrid Architecture using the **provided “Brilliant GraphRAG” dataset**.

**Objective:** Ingest the provided CSV logs (Conversations, Scene Descriptions, Locations) to measure:

* **Data Density:** Handling rich node properties (e.g., Image column as Scene Context, Audio column parsed into Speaker/Dialogue nodes).

* **Multilingual Handling:** Processing mixed \[en\] and \[zh\] tags found in the source text.

* **Write Latency (“The Pulse”):** Calculating the cost of the “Ripple Effect” on complex dialogue structures rather than simple proxy data.

* **Read Latency:** Traversing deep ontology paths (Topic $\\\\to$ Entity $\\\\to$ Chunk) as the graph grows.

### **2.2. The Architecture**

**A. The Ingestion Pipeline (Queued & Sequential)**

* **Event-Driven:** A queue-based system ingests logs in strict temporal order.

* **The “Pulse”:** A background worker triggers graph maintenance (Leiden Community Detection, Centrality updates) at set intervals.

**B. The Lexical Backbone (Time)**

* DAY $\\\\to$ SEGMENT $\\\\to$ CONVERSATION $\\\\to$ CHUNK.

* **Locations:** Text-based locations (e.g., “Old Town, Dali”) are linked to

  From  SEGMENT nodes to PLACE nodes.

**C. The Semantic Ontology**

* Extracted entities mapped to: Person (Speakers), Place, Context (Image descriptions), Concept, Action.

### **2.3. Retrieval Pipeline (“Deep Context”)**

* **Top-Down Traversal:** Match Topic $\\\\to$ Expand to Entity/Ontology $\\\\to$ Ground in Chunk $\\\\to$ Fetch Content (Postgres).

* **Hybrid Search:** Combining Graph Traversal with Vector Similarity Search for maximum recall

## **3\. Assumptions & Dependencies**

* **API Costs:** The Provider is responsible for all third-party API costs incurred during development and operation. 

* **Data Privacy:** The Client warrants that all “Life Log” data provided for ingestion is consensual and compliant with relevant privacy regulations (GDPR/CCPA). The Provider disclaims liability for PII retained in the Knowledge Graph based on the provided source data.

* **Infrastructure:** The system is designed to run on a standard Linux environment (docker-compose). Cloud deployment costs (AWS/GCP) are the Client’s responsibility.

---

# **INDEPENDENT CONTRACTOR AGREEMENT**

This Independent Contractor Agreement (the “Agreement”) is entered into as of December 24, 2025 (the “Effective Date”), by and between:

**Client:** Brilliant Labs (*Implied*)  
**Provider:** Tamás Diósi-Mákos (“Contractor”)

**1\. SERVICES** Contractor shall provide the services and deliverables described in the **Statement of Work (SOW)** above (the “Services”).

**2\. COMPENSATION** Client shall pay Contractor a fixed fee of **$10,000.00 USD**, payable in four (4) equal installments of $2,500.00 USD upon the successful completion of each Weekly Deliverable defined in the SOW.

**3\. RELATIONSHIP OF PARTIES** Contractor is an independent contractor, not an employee of Client. Contractor is responsible for all taxes and benefits associated with the compensation.

**4\. INTELLECTUAL PROPERTY (IP)**

**4.1. Definitions** \* **“Background IP”**: Refers to Contractor’s pre-existing code, libraries, pipelines, generic algorithms and any improvements thereto that are of general utility and not specific to Client’s domain. **This constitutes the “Machine” or “Engine”.** \* **“Deliverables”**: Refers to the specific work product created for Client, including the specific Schema/Ontology, Parsers (e.g., CSVLogParser), Configuration files, and the populated Knowledge Graph database. **This constitutes the “Solution”.**

**4.2. Ownership of Deliverables** Client shall own all right, title, and interest in and to the **Deliverables**. Contractor hereby assigns to Client all copyright and other intellectual property rights in the Deliverables.

**4.3. Licensing of Background IP (The “Machine”)** Contractor retains all right, title, and interest in the **Background IP**. Contractor hereby grants to Client a perpetual, irrevocable, worldwide, royalty-free, non-exclusive license to use, reproduce, modify, and display the Background IP **solely as embedded within or necessary to operate the Deliverables** for Client’s internal business purposes and end-user products.

**4.4. Restrictions on Resale** Client acknowledges that the Background IP represents Contractor’s proprietary “Graph Engine.” **Client AGREES NOT TO separate the Background IP from the Deliverables for the purpose of reselling, sublicensing, or distributing the Background IP as a standalone software product (e.g., a generic “Knowledge Graph Generation Platform”) to third parties.**

**5\. CONFIDENTIALITY** Both parties agree to keep confidential all proprietary information exchanged during the term of this Agreement. Contractor shall not disclose Client’s “Life Log” data to any third party.

**6\. TERMINATION** \* **For Convenience:** Client may terminate this Agreement at any time upon written notice, provided that Client pays Contractor for all Services performed and Milestones completed up to the date of termination. \* **For Cause:** Either party may terminate for material breach if the breach is not cured within 10 days of notice.

**7\. WARRANTIES** Contractor represents and warrants that the Services will be performed in a professional manner and that the Deliverables will not infringe upon the intellectual property rights of any third party.

**8\. GOVERNING LAW** This Agreement shall be governed by the laws of \[Jurisdiction Preference\], without regard to its conflict of law principles.

---

**IN WITNESS WHEREOF**, the parties have signed this Agreement as of the Effective Date.

| Client (Brilliant Labs) | Provider (Tamás Diósi-Mákos) |
| :---- | :---- |
| Signature: \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ | Signature: \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ |
| Name: \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ | Name: **Tamás Diósi-Mákos** |
| Title: \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ | Title: **Lead Engineer** |
| Date: \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ | Date: \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ |

## **—**

**DETAILED 4-WEEK SPRINT PLAN**

### **Week 1: The Baseline (Ingestion & Parser Implementation)**

**Goal:** Establish the Hybrid DB, build the Queued Ingestion, and implement the specific CSVLogParser for the provided dataset.

| Role | Task Category | Specific Action Items |
| :---- | :---- | :---- |
| **Sebi** | Infrastructure | [x] 1. Dual-DB Setup: Deploy Docker Compose with FalkorDB (Graph) + Postgres (Vector). <br> [x] 2. The Bridge: Implement src/kg/storage/hybrid_store.py to handle atomic writes to both DBs. |
| **Sebi** | Ingestion | [x] 3. Queued Worker: Build RedisQueue consumer for sequential consistency. <br> [x] 4. The “Pulse”: Setup Cron/Celery task for periodic run_community_detection. |
| **Tamas** | Parser Dev | [x] **5. CSV LifeLog Parser:** Implement CSVLogParser in src/kg/graph/parsers/ inheriting from BaseDocumentParser. *Logic:* Strict datetime parsing; Regex split for [en] dialogue blocks; Store Image desc in Segment metadata. |
| **Tamas** | Schema Mapping | [x] 6. Ontology Alignment: Map CSV columns to Graph Schema: - Speaker ID $\to$ Person - Location $\to$ Place - Image $\to$ Context node. |
| **Deliverable** |  | **[x] Live Ingestion Pipeline utilizing CSVLogParser and HybridStore.** |

### **Week 2: The Backbone (Enrichment & Spatial Logic)**

**Goal:** Transition from basic ingestion to enriched “Life Data” with spatial and multilingual support.

| Role | Task Category | Specific Action Items |
| :---- | :---- | :---- |
| **Sebi** | Optimization | [x] 1. Bulk Loader: Optimize QueueWorker to batch inserts (N=50) for FalkorDB efficiency. |
| **Sebi** | Resolution | [x] 3. Entity Resolution: Implement merge_similar_nodes in src/kg/graph/resolution.py using fuzzy string matching on “Location”. |
| **Tamas** | Data Logic | [ ] 4. Multilingual Splitting: Detect language tags [zh] vs [en] in CSVLogParser and route to appropriate embedding model defined in config.yaml. |
| **Tamas** | Segmentation | [x] 5. Episode Segmentation: Implement group_segments_by_time logic in src/kg/graph/parsing.py (>5min gap = new Episode node). |
| **Deliverable** |  | **[x] Structured Graph with Episode nodes and resolved Place entities.** |

### **Week 3: The Brain (Ontology & Intelligence)**

**Goal:** Turn the raw graph into a Knowledge Graph. Implement “The Pulse” algorithms (Leiden/Pruning) on Life Data.

| Role | Task Category | Specific Action Items |
| :---- | :---- | :---- |
| **Sebi** | Algorithms | [x] 1. Community Detection: Tune Leiden parameters in src/kg/community/detection.py for small-cluster conversation grouping. <br> [x] 2. Pruning: Implement prune_disconnected_nodes in src/kg/graph/maintenance.py. |
| **Tamas** | Intelligence | [ ] 3. Context Extraction: Implement ImageContextExtractor to generate (Image)-[:DEPICTS]->(Context) facts. |
| **Tamas** | Extraction | [ ] 4. Triplet Extraction: Update src/kg/graph/extractors.py to extract (Person)-[:SPEAKS_AT]->(Place) from metadata. |
| **Deliverable** |  | **[/] Rich Knowledge Graph with Context nodes and Community clusters.** |

### **Week 4: The Recall (Deep Context & Handoff)**

**Goal:** Finalize the Retrieval Logic and prove success with “Golden Queries.”

| Role | Task Category | Specific Action Items |
| :---- | :---- | :---- |
| **Sebi** | Retrieval | [x] 1. Hybrid Query: Implement search_hybrid() combining Cypher traversal + Vector ANN. <br> [x] 2. Latency: Optimize redis-py connection pooling. <br> [x] 3. Deploy: Dockerize graph_service, worker, redis stack. |
| **Tamas** | Synthesis | [x] 4. Context Builder: Implement build_context_window() in src/kg/retrieval/context.py joining Image + Dialogue. <br> [ ] 5. Golden Query: Validate “What did I eat…?” query against ground truth. |
| **Deliverable** |  | **[/] Final POC Codebase (Dockerized) + Golden Query Report.** |

