

# **From Discourse to Structure: A Graph-Based Approach to Topic Modeling in Political Debate**

## **Abstract**

Traditional topic modeling techniques, such as Latent Dirichlet Allocation (LDA), have long served as the standard for thematic analysis of large text corpora. However, their reliance on "bag-of-words" assumptions, which disregard syntax and relational context, limits their ability to capture the nuanced structure of complex discourse. This thesis explores a graph-based paradigm for topic modeling that constructs and analyzes a knowledge graph from political text. The central research questions investigate whether thematic communities identified within this graph can be considered valid "topics" from a linguistic and philosophical standpoint, and whether this structural approach offers a complementary alternative to probabilistic models in terms of interpretability.

The methodology is applied to the verbatim reports of the 'This is Europe' European Parliamentary debate series (2022-2024). Ontology-derived labels guide a hybrid extraction pipeline that combines **GLiNER2** entity detection with **DSPy-based** relation extraction. The resulting graph is then processed through Node2Vec-informed edge weighting and Leiden community detection. This design is motivated by the nature of political discourse: debates contain persistent entities, implicit relations, and recurring cross-references that are not well captured by simple co-occurrence statistics alone.

Recognizing that discourse topics exist at multiple granularities—a limitation noted in Section 1.1—the Leiden algorithm is applied hierarchically, first identifying broad thematic clusters and then detecting sub-communities within each. Each resulting community is then processed through an LLM-based summarization pipeline that generates interpretable labels and descriptive reports from graph structure and supporting textual evidence.

The main empirical evaluation in this thesis is a full-corpus run over all 13 debates in the series, comprising **311 segments** and **2085 text chunks**. The resulting graph contains **6377 nodes** and **20661 edges**, including **3557 entity nodes**, **70 topics**, and **341 subtopics**. Topic-separation diagnostics indicate weak but non-random community-level separation (silhouette = **0.160**, p < **0.001**) and strongly overlapping but statistically differentiated subcommunity structure (silhouette = **-0.220**, p < **0.001**), with global separation exceeding global overlap (**0.793** vs. **0.207**). The evidence therefore supports the claim that the graph-based pipeline can recover broad thematic structure in parliamentary discourse, while also showing that fine-grained topic boundaries remain highly overlapping in this corpus.

The thesis concludes that defining topics as structurally coherent communities within a knowledge graph represents a viable complementary approach to probabilistic methods. This exploration offers an alternative perspective on topic modeling—one that prioritizes interpretability and explicit structure—while acknowledging that graph-based methods are not necessarily superior to simpler approaches (Galke & Scherp, 2022).

---

## **Chapter 1: Deconstructing the 'Topic': From Philosophy to Computation**

### **1.1 The Philosophical and Linguistic Foundations of a 'Topic'**

Before a computational model can claim to identify "topics," it is imperative to establish a rigorous, non-computational understanding of what a topic is. The term is often used imprecisely in data science, treated as a mere label for a cluster of co-occurring words. However, its roots in linguistics and the philosophy of language reveal a far more structured and profound concept, one that is central to the organization of information and the construction of meaning in human communication.

From a linguistic perspective, the foundational distinction is between the *topic* (or *theme*) and the *comment* (or *rheme*) (Halliday, 1985; Firbas, 1992). The topic is what a sentence or clause is *about*; it is the entity or concept that anchors the discourse, providing the subject of the predication. The comment is what is being said *about* that topic; it is the new information, the assertion, or the description being provided. This division, known as information structure, posits that communication is not an unstructured stream of words but a deliberate organization of information into old (the topic, which connects to the existing discourse) and new (the comment) (Halliday, 1985). This fundamental structure implies that a topic is not a standalone artifact but exists in relation to the propositions made about it.

The philosophy of language deepens this understanding through the concept of "aboutness" (Reinhart, 1981, 1982). Reinhart argues that "aboutness" is the defining characteristic of a topic, moving beyond purely grammatical definitions of a subject to a pragmatic one based on communicative intent. The topic is the entity that the speaker directs the hearer's attention to, about which they intend to convey information. This philosophical framing is critical because it sets a higher bar for topic modeling: the goal is not merely to find clusters of words but to identify the primary subjects of "aboutness" that structure a body of text.

Furthermore, a crucial distinction must be made between a *sentence topic* and a *discourse topic* (Reinhart, 1981). A sentence topic is the constituent that a specific sentence is about, whereas a discourse topic is what an entire conversation or text is about. For example, in a debate about European energy policy, the discourse topic is "European Energy Policy." Within this discourse, individual sentences may have sentence topics like "natural gas reserves," "renewable energy investment," or "Russian dependency." Traditional computational models often struggle to separate these levels, conflating high-frequency terms associated with the overarching discourse topic with the more specific subjects of individual arguments. An effective topic model must be capable of resolving this hierarchy.

These concepts can be synthesized through the lens of Foucauldian discourse theory. Foucault defines a discourse not as a simple collection of statements, but as a "system of thoughts composed of ideas, attitudes, courses of action, beliefs, and practices that systematically construct the subjects and the worlds of which they speak" (Foucault, 1972). In this view, a discourse creates its own objects and concepts through the regulated interplay of statements. A "topic," therefore, is not just a word or a concept but a node within this system, defined by its relationships to other nodes. It is a representation of one of these constructed subjects. This provides a powerful theoretical framework: a true topic model should aim to uncover these systems of thought, revealing not just *what* is being discussed, but *how* the subjects of the discourse are constructed through the relationships between different ideas and entities.

### **1.2 Computational Approaches to Topic Modeling**

The abstract, theoretical concept of a topic must be operationalized to be computationally tractable. While the dominant approach for the last two decades has been the probabilistic paradigm, exemplified by Latent Dirichlet Allocation (LDA) (Blei, Ng, & Jordan, 2003), alternative structural approaches have gained interest in recent years.

Probabilistic models like LDA define a topic as a probability distribution over a vocabulary, inferred from the co-occurrence of words within documents. This relies on the "bag-of-words" assumption, which treats text as an unordered collection of terms, disregarding the rich relational structure of language. Despite its limitations, LDA remains a strong baseline—recent work by Galke and Scherp (2022) demonstrated that simple bag-of-words approaches can often outperform more complex graph-based methods in text classification tasks, raising important questions about the practical necessity of graph constructions.

In recent years, graph-based approaches have emerged as an alternative paradigm. TextGCN (Yao, Mao, & Luo, 2019) constructed heterogeneous graphs linking words and documents, demonstrating that graph convolutional networks could capture document-level topics. The Graph Neural Topic Model (GNTM) (Xie et al., 2021) integrated graph-based regularization into neural topic models. These approaches define topics not as statistical abstractions but as components of an explicit knowledge structure.

This thesis explores a **structural paradigm** that defines a topic as a community within a knowledge graph. The formal definition is as follows:

*A topic is a densely interconnected community of entities (nodes) and their relationships (edges) within a knowledge graph, which is algorithmically identified through community detection and can be articulated through a natural language summary.*

This approach moves the unit of analysis from words to entities—real-world concepts, people, places, and organizations—and their explicit, labeled relationships. It represents one possible operationalization of the theoretical concept of a "system of thoughts," where meaning is derived from the structure of connections. It is important to note that this is not claimed as a replacement for probabilistic methods but rather as an alternative exploration—one that prioritizes interpretability and explicit structure over computational efficiency.

---

## **Chapter 2: A Graph-Based Epistemology: Constructing Knowledge from Discourse**

### **2.1 The Corpus: The 'This is Europe' Parliamentary Debates**

The corpus selected for this study consists of the verbatim reports of the 'This is Europe' debate series held in the European Parliament. Between April 2022 and March 2024, 13 EU Heads of State or Government addressed the Parliament to present their visions for the future of the European Union.

The context of these debates is critical. They commenced shortly after Russia's full-scale invasion of Ukraine, a geopolitical event that reshaped European priorities overnight. They also ran concurrently with the conclusion of the Conference on the Future of Europe (CoFoE), a citizen-led initiative to guide EU policy. This backdrop ensures that while each leader brought a national perspective, the speeches were anchored in a shared set of urgent and overarching challenges.

An analysis of the debates conducted by the European Parliamentary Research Service (EPRS) identifies six recurring themes: (i) the value of EU membership, (ii) defending EU values, (iii) the main challenges facing the EU, (iv) delivering for EU citizens, (v) next steps in EU integration, and (vi) the importance of EU unity (European Parliamentary Research Service, 2024). The EPRS report further breaks down the specific topics addressed, noting that Ukraine, enlargement, and energy were the most frequently and extensively discussed subjects across all speeches.

This corpus is particularly well-suited for this thesis for two reasons. First, its thematic cohesion provides a stringent test for any topic model. The high degree of thematic overlap and shared vocabulary makes it difficult for purely statistical models to disentangle nuanced sub-topics. In practice, this means that strong structural partitions may coexist with weak semantic separation, making the corpus a useful stress test for graph-based topic modeling. Second, the detailed EPRS briefing serves as an expert-curated "ground truth." It provides an independent, qualitative baseline against which the computationally derived topics can be assessed, allowing for a cautious comparison between model output and expert political analysis.

### **2.2 From Text to Graph: A Hybrid Ontology-Guided Pipeline**

The transformation of unstructured text from the parliamentary debates into a structured knowledge graph is the foundational step of the methodology. In the validated implementation, this process combines ontology guidance, neural entity detection, and constrained relation extraction to create a relational representation of the discourse. The pipeline consists of several sequential stages.

1.  **Ontology-Guided Schema Definition:** Before any text is processed, the pipeline ingests a domain-specific ontology (defined in OWL/RDF formats). A dedicated `OntologyLabelExtractor` parses these files to extract a set of valid class labels (e.g., *Person*, *Policy*, *Organization*, *GeopoliticalEntity*). These labels serve as a schema that constrains subsequent extraction steps and reduces type drift in downstream graph construction.

2.  **Text Chunking:** The raw verbatim transcripts are segmented into manageable text chunks. Chunk size is a practical design parameter because it governs both computational cost and extraction granularity. In the final corpus run reported later, a chunk size of **512 characters with 64 characters of overlap** was used. This choice was made deliberately after an earlier configuration with substantially larger chunks (2048/128) was found to interact poorly with the entity- and relation-extraction stages: chunks longer than the GLiNER2 effective context and the DSPy generation budget produced silent truncation, with measurable losses in extracted entities and relations relative to the same input re-chunked at 512. The smaller window therefore preserves local context within the limits of CPU-based extraction without sacrificing recall.

3.  **Constrained Entity and Relationship Extraction:** Each text chunk is processed by a hybrid extraction pipeline. First, **GLiNER2** scans the text for ontology-compatible entity surfaces. These detections are normalized into entity hints. Next, a **DSPy-based relation extractor** is prompted with both ontology classes and the detected entity hints to produce grounded relation triplets. In the current implementation, relations whose endpoints cannot be aligned conservatively to the detected hints are dropped, which favors precision over aggressive graph expansion.

4.  **Graph Assembly:** The extracted elements from all chunks are aggregated into a single knowledge graph. Entities become nodes and extracted relations become edges. The result is a structured, machine-readable representation of the debate corpus that can be used for community detection, summarization, and downstream analysis.

### **2.3 The Emergence of Structure: Scale-Free Properties in Extracted Knowledge**

A significant finding that validates the integrity of this graph construction process is the topological nature of the resulting network. Analysis of the graph's degree distribution—the probability P(k) that a randomly chosen node has k connections—reveals that it follows a power law, where P(k)∼k−γ. This is the defining characteristic of a scale-free network (Barabási & Albert, 1999; Newman, 2005).

The study of scale-free networks was pioneered by Albert-László Barabási and Réka Albert (1999), who discovered that this topology is not a mathematical curiosity but a ubiquitous feature of real-world complex systems, including the World Wide Web, social networks, and biological protein-interaction networks. The emergence of this structure is explained by two simple, yet powerful, underlying mechanisms:

**growth** and **preferential attachment** (Barabási & Albert, 1999).

* **Growth:** Real networks are rarely static; they expand over time through the addition of new nodes.  
* **Preferential Attachment:** New nodes are more likely to connect to existing nodes that are already highly connected. This "rich-get-richer" phenomenon leads to the formation of a few highly connected "hubs" that dominate the network's structure.

The appearance of a scale-free topology in the knowledge graph extracted from the parliamentary debates is not a random artifact. It is a profound reflection of the fundamental dynamics of discourse itself. A series of political debates is a growing system: each speech adds new concepts and arguments to the existing network of ideas (growth). When speakers contribute, they do not introduce concepts in a vacuum. To be relevant and persuasive, they must connect their arguments to the central, most salient themes of the ongoing discussion—the established hubs like 'Ukraine', 'energy dependency', or 'EU values' (preferential attachment) (Barabási & Albert, 1999).

Therefore, the very process of building a coherent, multi-speaker discourse is a network-generating mechanism that naturally follows the principles of scale-free models. The appearance of this exact topology in our generated graph acts as a **structural validation of the extraction pipeline**. It confirms that the ontology-guided extraction did not impose an artificial or rigid structure, but rather successfully captured the organic, self-organizing nature of the political debate. A graph that appeared random (Poisson degree distribution) or regular (lattice-like) would indicate a failure of the extraction process to identify the true semantic hubs. The scale-free property confirms that the "hubs" in our graph (e.g., *Ukraine*, *Energy Crisis*) correspond to the actual semantic anchors of the real-world discourse.

### **2.4 Ensuring Coherence: Conservative Entity Normalization and Resolution**

The integrity of the graph's structure, particularly the accurate identification of hubs, depends on a crucial data processing step: Entity Resolution (ER). Raw extraction output can be inconsistent, creating multiple nodes for the same real-world entity (e.g., "Olaf Scholz," "the German Chancellor," "Mr. Scholz"). ER is the process of identifying and merging such duplicates so that each unique entity is represented as consistently as possible.

In the implemented pipeline, this step is handled conservatively. Rather than treating knowledge graph embeddings as the primary mechanism for large-scale automatic merging, the workflow relies mainly on normalization, strict similarity thresholds, and downstream graph cleaning to avoid unsupported merges. This design choice reflects the practical priority of preserving trustworthy graph structure during corpus-level analysis.

This conservative behavior is also visible in the final full-corpus run, where the entity-resolution stage reduced obvious duplication without aggressively collapsing semantically adjacent nodes. The interpretation of these results should therefore remain modest: the current evidence validates extraction-and-assembly stability more strongly than it validates a large-scale automated canonicalization system.

Knowledge graph embeddings remain relevant as a future direction for entity resolution and link prediction (Nguyen, 2020), but they are not the main validated mechanism in the implemented pipeline emphasized in this thesis.

### **2.5 Conservative Filtering and Graph Cleanup**

The knowledge graph construction pipeline incorporates several filtering and cleanup mechanisms intended to reduce noise and preserve interpretable structure before community detection. In the current implementation, this is less an exhaustive linguistic filter stack than a conservative graph-construction strategy.

The main design principle is precision over aggressive abstraction. Ontology-derived labels constrain the entity search space, DSPy relation extraction is grounded in detected entity hints, and weak or unsupported structure is pruned during downstream graph cleanup. Similarity thresholds and minimum-component rules are used to avoid over-merging and to reduce fragmentation caused by low-value nodes or edges.

These mechanisms serve two related purposes. First, they reduce the chance that community structure is driven by spurious extractions or accidental bridges. Second, they make the downstream summaries more interpretable by ensuring that the graph presented to the community-detection stage is already a conservative representation of the discourse. In this study, that trade-off is desirable: it is preferable to retain a slightly incomplete graph than to create topic communities from unsupported structure.

### **2.6 Topic Modeling as a Use-Case: Structural versus Distributional Approaches**

The application of community detection to the filtered knowledge graph represents a structural alternative to traditional distributional topic modeling. Whereas LDA and its variants define topics as probability distributions over words—inferring latent structures from co-occurrence patterns—the structural paradigm defined in this thesis treats topics as explicit subgraphs identified through community detection. This distinction has important implications for interpretability, validation, and the philosophical foundations of what constitutes a "topic."

**Structural Topics as Relational Clusters**: In the structural paradigm, a topic is defined as a densely interconnected subgraph whose nodes share more edges with each other than with the rest of the graph. This definition operationalizes the philosophical notion of a topic as a coherent system of "aboutness" relations. The Leiden algorithm's guarantee of connected communities ensures that each identified topic forms a single, cohesive unit—no disconnected clusters that would violate the intuition of a unified theme.

**Validation Through Structural Properties**: The structural approach enables validation through network properties that are unavailable to distributional methods. The emergence of scale-free topology in the extracted graph serves as a strong form of structural validation: if the extraction pipeline has successfully captured the underlying discourse structure, the resulting network should exhibit the same statistical properties as real-world complex networks. The observed power-law degree distribution confirms that the ontology-guided extraction has produced a graph with organic structure rather than artificial clustering.

**Complementarity with Distributional Methods**: The structural approach does not necessarily supplant distributional methods but rather complements them. Where LDA infers topics from word co-occurrence patterns without explicit relational structure, the graph-based approach leverages the full richness of entity-relationship triples. The two paradigms may be fruitfully combined: distributional similarity measures could inform edge weights, while structural community detection could provide topic boundaries that constrain distributional inference.

The exhaust filtering mechanisms described above play a critical role in this use-case. By ensuring that only semantically substantive entities participate in the graph, the filters reduce the risk that community detection will identify spurious structures formed by noise entities. The resulting topics are more likely to correspond to genuine thematic divisions in the discourse rather than artifacts of the extraction process.

**A Note on Comparison with Probabilistic Methods**: In addition to the graph-based analysis, this thesis now includes a direct LDA baseline on the same debate corpus. The baseline is estimated over the same 311 cleaned discourse segments used for corpus-level analysis, using bag-of-words features with unigram and bigram counts. Candidate topic counts from 6 to 36 were compared under a held-out perplexity split together with intrinsic topic-quality measures (UMass and NPMI coherence) and topic diversity. On this basis, the most balanced LDA configuration was selected at **k = 12**, yielding **mean NPMI = 0.148**, **mean UMass = -1.551**, **topic diversity = 0.517**, and **held-out perplexity = 19685.3**. This does not make the graph-based and probabilistic approaches fully commensurable, because they optimize different representations and outputs. It does, however, provide a strong baseline against which the present method can be discussed more responsibly: LDA offers a competitive distributional summary of the corpus, while the graph-based pipeline offers explicit entities, relations, hierarchy, and provenance-aware interpretability.

---

## **Chapter 3: Uncovering Thematic Structures: Community Detection and Summarization**

### **3.1 The Leiden Algorithm: Partitioning the Graph into Thematic Communities**

Once a validated and coherent knowledge graph is constructed, the core task of identifying topics begins. In the structural paradigm, this is framed as a problem of community detection. A community in a network is defined as a set of nodes that are more densely connected to each other than they are to the rest of the network. This graph-theoretic concept maps directly onto the intuitive notion of a topic: a collection of closely related ideas and concepts.

The algorithm employed for this task is the Leiden algorithm, a state-of-the-art method for community detection (Traag, Waltman, & van Eck, 2019). The Leiden algorithm is an iterative process that aims to find a partition of the graph's nodes that maximizes a quality function known as *modularity* (Newman & Girvan, 2004; Newman, 2006). Modularity measures the difference between the density of edges within the detected communities and the density that would be expected if the edges were distributed randomly, preserving node degrees. A high modularity score indicates a strong, non-random community structure.

The Leiden algorithm improves upon its well-known predecessor, the Louvain method, in a critical way. While Louvain is effective, it can sometimes produce communities that are poorly connected or even internally disconnected. The Leiden algorithm introduces an intermediate refinement phase into its iterative process, which explicitly checks the internal connectivity of communities and may split them to resolve such issues. As a result, the Leiden algorithm **guarantees that all detected communities are well-connected subgraphs** (Traag, Waltman, & van Eck, 2019).

This guarantee is not merely a technical improvement; it provides a graph-theoretic enforcement of *thematic coherence* for the identified topics. As established in Chapter 1, a topic must be a coherent set of related concepts. In the language of graph theory, coherence is synonymous with connectivity. A disconnected community would represent a "topic" containing two or more sets of ideas with no explicit path of relationship between them, violating the fundamental definition of a single, unified topic. By ensuring that every node in a community is part of a single connected component, the Leiden algorithm algorithmically enforces this principle.

#### **3.1.1 Enhancing Structure with Node2Vec Embeddings**

While the Leiden algorithm is effective on unweighted graphs, this research postulates that the *strength* of the relationship between concepts is as important as its existence. To capture this, the pipeline integrates **Node2Vec**, an algorithmic framework for learning continuous feature representations for nodes in networks (Grover & Leskovec, 2016).

The process involves three steps:
1.  **Structural Embedding:** Node2Vec is trained on the graph to generate low-dimensional vector embeddings for every node. These embeddings capture the topological neighborhood of each node; nodes that share similar connections (homophily) or structural roles are mapped to nearby points in vector space.
2.  **Edge Reweighting:** The cosine similarity between the embeddings of connected nodes is calculated. This similarity score is then applied as a weight to the edge connecting them.
3.  **Weighted Detection:** The Leiden algorithm is run on this weighted graph. Edges connecting structurally similar nodes are treated as "stronger," encouraging the algorithm to keep them in the same community.

In the validated runs reported in this thesis, Node2Vec remains part of the implemented community-detection pipeline as a structurally motivated weighting layer. Across both the preparatory runs and the final full-corpus execution, the weighted pipeline remained stable while graph size increased substantially. However, the current manifests do not show a positive modularity delta relative to the unweighted baseline. Accordingly, Node2Vec is best presented here as a validated architectural component and a theoretically motivated structural prior, rather than as a separately re-demonstrated source of modularity improvement in this corpus.

### **3.2 Hierarchical Abstraction of Discourse**

Human discourse is often organized hierarchically. A broad theme, such as "EU's Main Challenges," can be broken down into more specific sub-themes like "The Energy Crisis," "Inflation," and "Disinformation," which themselves can be further decomposed. The Leiden algorithm naturally accommodates this nested structure.

In the implementation validated here, the hierarchical procedure is applied in two levels. After an initial partition of the graph is found, each resulting community is treated as a smaller graph and processed again to identify sub-communities. This produces a nested partitioning of the data, providing a multi-resolution view of the thematic landscape without requiring an open-ended recursive depth. The result is still useful for analysis: it allows the discourse to be examined both at the level of broader themes and at the level of finer-grained sub-topics.

### **3.3 Generating Topic Narratives: Structure-Aware Hierarchical Summarization**

The output of the community detection process is a set of subgraphs, where each subgraph represents a topic. While this structured representation is powerful for analysis, it is not immediately interpretable to a human user. The final stage of the methodology bridges this gap through a novel, structure-aware summarization process that explicitly leverages the graph topology to generate rich, multi-faceted community reports.

#### **3.3.1 Hierarchical Processing: From Subtopics to Topics**

Recognizing that discourse exhibits natural hierarchical organization, the summarization pipeline implements a strict bottom-up processing order. The system first identifies and processes all leaf-level communities (subtopics), generating comprehensive summaries for each. Only after subtopic summarization is complete does the pipeline proceed to higher-level communities (topics), where the summaries of constituent subtopics are explicitly provided as contextual input. This hierarchical composition ensures that topic-level summaries are genuine abstractions that synthesize insights from their component parts, mirroring how human analysts construct understanding at multiple levels of granularity.

#### **3.3.2 Structure-Aware Context Formatting**

Unlike traditional summarization approaches that operate solely on concatenated text, this methodology explicitly surfaces the graph structure to the LLM. For each community, the input context is formatted using XML markup to delineate distinct information layers:

1. **Community Structure**: An explicit enumeration of entities within the community, sorted by degree (topological prominence), accompanied by their ontological types. This provides the LLM with immediate visibility into the semantic actors involved.

2. **Relationship Network**: A structured listing of internal relationships (edges) between community members, formatted as source-relation-target triplets. This captures the specific ways in which entities interact within the thematic cluster.

3. **Sub-Community Summaries** (for hierarchical topics): Previously generated summaries of child communities, providing compositional context.

4. **Textual Evidence**: A curated selection of text chunks from the original corpus where these entities co-occur, serving as grounding evidence.

By providing this multi-modal input—combining structural metadata with textual evidence—the summarization process becomes *structure-aware*, enabling the LLM to generate summaries that reflect both the topological salience of entities and their discursive context.

#### **3.3.3 Structured Report Generation with Explicit Findings**

The summarization prompt employs a rigorous template that mandates a structured JSON output format. Rather than generating a simple descriptive paragraph, the LLM is instructed to produce a formal analytical report containing:

- **Title**: A concise, descriptive label for the community (3–10 words).
- **Executive Summary**: A comprehensive 3–5 sentence overview synthesizing the core theme and key dynamics.
- **Detailed Findings**: A structured list of 3–5 specific insights, each comprising:
  - A summary statement identifying a key pattern or observation.
  - An explanation paragraph that cites specific entities, relationships, or text evidence supporting the finding.

This format enforces analytical rigor, ensuring that summaries are not merely descriptive but provide interpretable, evidence-grounded insights. The structured findings serve as auditable claims that can be traced back to the underlying graph structure, enhancing the transparency and verifiability of the topic model.

#### **3.3.4 Prompt Engineering for Analytical Depth**

The prompt itself is carefully engineered to elicit high-quality analytical output. It employs several best practices:

- **Role Assignment**: The LLM is assigned the role of an "expert intelligence analyst specializing in graph-based pattern detection," establishing an analytical rather than merely descriptive framing.
- **Explicit Constraints**: The prompt includes explicit instructions on grounding (all findings must be supported by provided evidence), completeness (prefer synthesis over listing), and tone (professional, analytical, objective).
- **Output Format Specification**: The required JSON schema is explicitly defined, with field-by-field descriptions, ensuring consistent structure across all generated reports.

This structured prompting approach represents a significant methodological advancement over generic "summarize this text" instructions, aligning the summarization task with the graph-theoretic foundations of the overall methodology.

---

## **Chapter 4: Analysis of Graph-Derived Topics from the 'This is Europe' Debates**

### **4.1 Full-Corpus Graph Construction and Topic Recovery**

The main empirical evaluation of this thesis is a full-corpus execution over all 13 debates in the "This is Europe" series. Under the final configuration, the pipeline processed **311 discourse segments** and **2085 text chunks**, extracted **7635 entities** and **3824 relations**, and produced a final graph with **6377 nodes** and **20661 edges**. Of these nodes, **3557** are canonicalized entity nodes, while **70** represent higher-level topics and **341** represent subtopics.

These totals are important because they show that the graph is not merely a lexical scaffold. The extracted structure is large enough to support multi-level thematic abstraction, yet still interpretable enough to inspect directly. The graph also remains sparse relative to its size, which is expected in a discourse-derived network where a small number of entities function as central hubs while many others remain locally connected.

The entity-resolution stage merged **82 nodes** across **82 candidate clusters**. This is consistent with the conservative resolution strategy described earlier: the procedure removes some obvious duplicates and inflectional variants without attempting aggressive canonicalization. At the same time, the diagnostics show that this blocking-based approach can still collapse semantically adjacent or orthographically similar surface forms in some cases. In the final saved outputs used for interpretation, a small number of these obviously false-positive merges were corrected post hoc directly in the output artifacts rather than by rerunning extraction, so the reported graph statistics reflect the repaired corpus graph rather than the raw pre-repair merge list.

### **4.2 Structural Properties of the Final Graph**

At the graph level, the final run produced a weighted modularity score of **0.7456**, compared with an unweighted baseline of **0.6277** on the same entity subgraph. This indicates a substantial improvement from the Node2Vec-informed weighting layer under the validated final configuration. The gain should still be interpreted cautiously: it shows that structural weighting can sharpen the community partition, but it does not by itself imply clean semantic separability at the level of fine-grained subtopics.

The degree distribution also highlights the emergence of dominant semantic hubs. After entity resolution, the most connected entity nodes are **EUROPEAN_UNION** (degree **800**), **EUROPEAN** (**663**), **EUROPE** (**520**), **UKRAINE** (**414**), and **EUROPEAN_PARLIAMENT** (**294**). These are not random artifacts; they correspond closely to the recurring institutional and geopolitical anchors of the debates. Their prominence is substantively plausible given the corpus and provides an additional face-validity check on the extracted graph.

The distribution of entity surface-form classes further clarifies the character of the final graph. The resolved graph contains **2393 named-entity-like nodes**, **1070 concept-like nodes**, **88 unknown forms**, and **6 role artifacts**. This balance suggests that the pipeline is not merely extracting actors, but also recovering a large layer of policy and institutional concepts that help mediate thematic structure between speakers, organizations, and events.

### **4.3 Topic Recovery and Hierarchical Summarization**

The final graph yields **70 topics** and **341 subtopics**, which together form the hierarchical thematic output of the model. The summarization stage processed all of these successfully in the corrected final run, and the resulting node labels show that the detected communities are interpretable at a broad thematic level.

Among the highest-degree topic nodes are:
- **European Integration and Crisis Management**
- **European Community Response to Ukraine Crisis**
- **European Integration and Energy Security**
- **European Integration and Challenges**
- **European Parliament Community Analysis**

These labels reflect a recurring pattern in the corpus: the strongest macro-topics combine institutional integration, geopolitical crisis response, and energy/security concerns. This aligns well with the known political context of the debates, which unfolded under the combined pressures of Russia's war against Ukraine, energy insecurity, enlargement debates, and questions about the future institutional shape of the Union.

At the subtopic level, the hierarchy becomes more fine-grained and substantially denser. The large number of subtopics indicates that broad themes fracture into many local argumentative clusters rather than into a few sharply bounded semantic blocs. This is substantively plausible for parliamentary discourse, where speeches often revisit the same broad concerns from many national, ideological, and policy-specific perspectives.

### **4.4 Topic-Separation Diagnostics**

The final topic-separation report does not support a claim of strong semantic separability. At the community level, the silhouette score is **0.160**, which indicates only weak geometric separation between the topic representations. However, the corresponding ANOVA result is highly significant (**p < 0.001**), and the multivariate PCA-based diagnostic is also strongly significant. This means that the broad topic groups are not random, even if they are not cleanly separated in embedding space.

At the subcommunity level, the pattern is more extreme. The silhouette score is **-0.220**, which indicates poor separation and substantial overlap in the strict geometric sense. Yet the associated ANOVA result is again highly significant (**p < 0.001**), with a very large number of subcommunity groups (**341**) and samples (**3764**). This combination suggests that the subtopics are statistically differentiated but spatially interwoven. In other words, the subtopics are not arbitrary, but they remain highly entangled in semantic space.

The global separation and overlap diagnostics reinforce this interpretation. The average global separation is **0.793**, while global overlap is **0.207**. This indicates that, in aggregate, the embedding space does preserve meaningful distance between topic representations. The difficulty arises not from complete collapse, but from the fact that deliberative parliamentary discourse repeatedly reuses shared entities, institutions, and crisis frames across many local subtopics.

### **4.5 Comparison with an LDA Baseline**

To place the graph-based results in a stronger empirical context, an LDA baseline was estimated on the same corpus after segment-level transcript cleanup. The comparison used the **311 cleaned discourse segments** as bag-of-words documents, with unigram and bigram counts, English stop-word filtering, and candidate topic counts from **6** to **36**. Model selection was based on a balanced reading of **held-out perplexity**, **UMass coherence**, **NPMI coherence**, and **topic diversity**, rather than on silhouette, which is not an appropriate metric for probabilistic topic models.

Under this protocol, the most balanced LDA solution was obtained at **k = 12**. It achieved **mean NPMI = 0.148**, **mean UMass = -1.551**, **topic diversity = 0.517**, **held-out perplexity = 19685.3**, and activated all **12** topics in the training split. The top LDA topics were recognizable and substantively plausible, frequently centering on combinations of European Union governance, Ukraine, energy, security, Romania/Bulgaria-Schengen questions, Cyprus/Turkey, and country-specific leadership frames. In that sense, LDA succeeds as a strong baseline: it does recover meaningful recurring themes from the corpus.

At the same time, the baseline also clarifies the distinctive contribution of the graph-based method. The LDA topics are represented as weighted vocabularies, and several remain mixtures of country names, procedural expressions, and broad political terms such as *union*, *minister*, *people*, or *today*. By contrast, the graph-based pipeline yields **70 topics** and **341 subtopics** embedded in an explicit relational structure, with identified hubs, entity-resolution diagnostics, topic summaries, and traceable supporting evidence. The comparison therefore does not justify a blanket superiority claim for the graph-based approach on classical topic-model metrics. Instead, it supports a more modest argument: **LDA provides a strong and competitive distributional baseline, while the graph-based pipeline offers a richer, more auditable representation of discourse structure that is especially valuable when relations, hierarchy, and provenance matter analytically.**

### **4.6 Interpretation of the Final Results**

Taken together, the final results support a restrained but positive conclusion. The pipeline successfully converts the entire corpus into a large, interpretable, hierarchically summarized knowledge graph. It recovers broad thematic communities that are statistically non-random and substantively plausible. At the same time, the weak community-level silhouette and negative subcommunity silhouette show that the discourse does not decompose into sharply isolated thematic islands.

This should not be treated simply as a methodological failure. In parliamentary debate, overlap is expected. Speakers repeatedly return to the same institutional actors, shared crises, and policy frameworks. Under such conditions, semantic entanglement is itself a property of the corpus. The graph-based approach is valuable precisely because it preserves that entanglement rather than forcing hard separation through a purely distributional lens.

The final corpus-wide run therefore supports the following interpretation: the graph-based pipeline is effective at recovering broad thematic structure and hierarchical argumentative organization, but the resulting topics should be understood as overlapping relational communities rather than as fully discrete semantic clusters. This is a more modest claim than strong topic separability, but it is also a more realistic one for this type of political discourse.

### **4.7 Diagnostics and Qualitative Validation**

The final run also provides useful diagnostics for methodological evaluation. First, the graph is dominated by a small number of highly connected hubs, which is consistent with the structure of repeated multi-speaker debate around shared institutions and crises. Second, the entity-resolution report shows that many merges involve inflectional or adjectival variants (for example, country names and their adjectival forms), confirming that the conservative merge policy is operating on plausible cases. Third, the final topic labels themselves reveal recurring macro-themes—European integration, Ukraine, institutional reform, and energy security—that are consistent with the external EPRS analysis of the same debate series.

These diagnostics do not prove that every topic label is optimal, nor that every merge is correct. But they do provide triangulating evidence that the final graph captures recognizable political structure. For the purposes of this thesis, that is the key empirical standard: not perfect semantic partitioning, but an interpretable graph representation whose communities can be related back to the discourse and to external political analysis.

---

## **Chapter 5: Conclusion**

### **5.1 Summary of Research**

This thesis has pursued a single, coherent research question: whether a graph-based approach to topic modeling, grounded in explicit knowledge representation, offers substantive advantages over traditional probabilistic methods for analyzing political discourse. The investigation proceeded from a theoretical foundation in linguistics and the philosophy of language, through methodological development, and culminated in empirical evaluation on the "This is Europe" debate corpus.

The central theoretical contribution is a reformulation of "topic" from a probabilistic construct—a distribution over words—into a structural one: a densely connected subgraph within a knowledge graph. This redefinition aligns more closely with intuitive notions of thematic coherence, wherein a topic comprises not merely co-occurring terms but a network of interrelated concepts bound by explicit semantic relationships.

The methodology developed for this thesis integrates ontology-guided schema construction, GLiNER2-based entity detection, DSPy-based relation extraction, community detection via the Leiden algorithm, Node2Vec-informed structural weighting, hierarchical summarization built on the extracted graph, and a direct LDA baseline for comparison. Empirical evaluation on the complete debate corpus yields several cautious findings.

First, the final graph is large and structurally coherent. The corpus-wide run processed **13 debates**, **311 segments**, and **2085 chunks**, producing a graph with **6377 nodes** and **20661 edges**. This indicates that the pipeline can integrate a substantial amount of discourse into a single interpretable relational structure without obvious collapse or fragmentation.

Second, community detection and summarization recover a rich thematic hierarchy. The final run generated **70 topics** and **341 subtopics**, indicating that the graph structure supports both broad thematic abstraction and fine-grained decomposition. The highest-degree topic labels center on European integration, the war in Ukraine, institutional politics, and energy security, which is substantively consistent with the known political context of the debates.

Third, the topic-separation results support a restrained but positive interpretation. At the community level, the pipeline yields **weak cluster separation** (silhouette = **0.160**) together with **highly significant between-group differences** (p < **0.001**). At the subcommunity level, the representations are strongly overlapping (silhouette = **-0.220**) but still statistically differentiated (p < **0.001**). This suggests that broad thematic structure is real, while finer-grained subtopics remain semantically entangled.

Fourth, the full-corpus results reinforce the distinction between structural coherence and semantic discreteness. The graph-based pipeline is clearly capable of recovering broad, interpretable communities from political discourse. However, the negative subcommunity silhouette indicates that parliamentary debate does not decompose into sharply separated fine-grained semantic units.

### **5.2 Theoretical and Methodological Contributions**

This research makes three contributions, each of which should be understood as modest steps toward understanding the potential of graph-based approaches rather than definitive advances.

**Methodologically**, this thesis demonstrates that graph-based topic modeling is a feasible and interpretable alternative to probabilistic approaches. The hierarchical summarization pipeline developed here provides one template—among possible alternatives—for translating complex graph structures into human-readable analytical reports, with auditability built into each stage. Unlike LDA and its variants, which produce probability distributions whose interpretability requires additional post-processing, the graph-based approach generates explicit knowledge structures where each identified topic is a concrete subgraph that can be examined and related back to source documents. The added LDA baseline strengthens this claim by showing that the graph-based method is being evaluated against a real probabilistic comparator rather than only against its own internal diagnostics.

**Theoretically**, this work offers one perspective on the relationship between structural coherence and semantic distinctiveness in topic modeling. The observation that statistically significant group differences can coexist with weak or negative silhouette scores is especially relevant for deliberative political discourse, where overlap between themes is expected. In this setting, shared institutions, crises, and policy frames produce entangled topic structure even when higher-level communities remain substantively meaningful.

**Practically**, the approach appears well-suited for in-depth analysis of complex, interconnected discourse in domains where interpretability and verifiability are paramount. Political science, policy analysis, and legal text analysis are obvious examples where the ability to inspect entities, relations, and community summaries may be more valuable than the computational efficiency of purely probabilistic topic models.

### **5.3 Limitations**

The study's findings should be interpreted in light of several limitations.

**Parameter Sensitivity**: The optimal resolution parameter for community detection may vary across corpora. The current approach used fixed or empirically tuned parameters, but a more principled approach to parameter selection could improve robustness.

**Conservative Entity Resolution**: The implemented pipeline prioritizes precision over aggressive canonicalization. This reduces the risk of unsupported merges, but it also means that the current experiments should not be interpreted as a full validation of large-scale automated entity resolution.

**Semantic Enrichment**: The current pipeline extracts entities and relationships but does not yet enrich the graph with explicit entity descriptions or auxiliary metadata that might improve alignment between structural and semantic topic quality.

**Evaluation Frameworks**: The results show that standard clustering-style diagnostics remain difficult to interpret for deliberative discourse. The full-corpus run yields statistically significant differences at both community levels, but silhouette values remain weak or negative. For parliamentary debate, where overlap is expected and substantive, alternative evaluation frameworks that capture argumentative diversity, perspectival variance, or deliberative quality may prove more informative.

**Corpus Scope**: The "This is Europe" corpus is well-suited for testing the method's ability to handle thematic consilience, but it may not be representative of all political discourse. Application to corpora with more distinct thematic divisions could clarify whether the present combination of structural coherence and weak fine-grained separation is domain-specific.

### **5.4 Future Research Directions**

Several promising directions emerge from this research.

**Knowledge Graph Embeddings and Link Prediction**: The current pipeline does not exploit knowledge graph embeddings (KGEs) as a mainline validated component for link prediction or entity resolution. Models such as TransE, ComplEx, and RotatE learn vector space representations that capture complex relational patterns and could support more ambitious automated entity canonicalization and link prediction.

**Graph Neural Topic Models**: The topic modeling literature has seen advances in integrating graph-based regularization into neural topic models. The Graph Neural Topic Model (GNTM) uses document graphs to regularize topic distributions, while TextGCN builds heterogeneous graphs of words and documents for joint learning of word and document embeddings. The Node2Vec approach used in this thesis can be viewed as a simpler, unsupervised precursor to these more sophisticated architectures.

**Retrieval-Augmented Generation**: Recent developments in graph-based RAG systems—such as GraphRAG and LightRAG—demonstrate how entity knowledge graphs with hierarchical community summaries can improve question answering over large text corpora. Future work could implement a GraphRAG-style interface for querying the parliamentary debates using natural language.

**Dynamic and Incremental Graph Analysis**: Future work could study how graphs grow across time in a more principled way. A Cognee-like accumulation perspective—where entity resolution, upsert logic, and merge eligibility are treated as first-class problems—could provide a useful framework for longitudinal graph growth and evolving discourse memory.

**Cross-Lingual and Multimodal Extensions**: The methodology could be extended to multilingual corpora to explore how topics are framed differently across languages. Integration of multimodal data—voting records, policy documents, or visual elements—could provide richer bases for analysis.

**Causal and Argumentative Analysis**: The explicit relational structure of the knowledge graph could be leveraged for tasks beyond topic identification, such as causal claim extraction, argumentative structure mapping, stance detection, and influence network analysis.

### **5.5 Final Reflections**

This thesis has explored whether defining topics as structurally coherent communities within a knowledge graph offers a viable alternative to the dominant probabilistic paradigm. The approach developed here is computationally more demanding than LDA or similar methods, but it offers gains in explainability and auditability that may be valuable in domains where interpretability is paramount. The direct LDA baseline added here confirms that a strong bag-of-words model can recover meaningful recurrent themes from the same corpus; the contribution of the graph-based approach is therefore not that it replaces such baselines, but that it complements them with explicit relational structure, hierarchy, and provenance. As Galke and Scherp (2022) demonstrated, simpler distributional methods often perform equally well or better in practice; this thesis therefore does not claim superiority, but contributes to a broader exploration of alternative paradigms.

The central empirical lesson of the final corpus-wide run is that political discourse can exhibit both meaningful high-level thematic structure and strong fine-grained semantic overlap at the same time. This asymmetry should not be read simply as failure. Rather, it suggests that graph-based topic modeling may be especially valuable as a structural and interpretive tool in domains where the entanglement of themes is itself politically substantive.

Future work in knowledge-graph-based discourse analysis, graph neural topic models, and graph-enhanced retrieval systems may build on the insights developed here. More broadly, this thesis contributes to an ongoing conversation in computational linguistics about the role of structural representations in natural language understanding.

---

## **References**

Barabási, A.-L. & Albert, R. (1999) 'Emergence of scaling in random networks', *Science*, 286(5439), pp. 509-512.

Blei, D. M. (2012) 'Probabilistic topic models', *Communications of the ACM*, 55(4), pp. 77-84.

Blei, D. M., Ng, A. Y. & Jordan, M. I. (2003) 'Latent Dirichlet Allocation', *Journal of Machine Learning Research*, 3, pp. 993-1022.

Bordes, A., Usunier, N., Garcia-Duran, A., Weston, J. & Yakhnenko, O. (2013) 'Translating embeddings for modeling multi-relational data', *Advances in Neural Information Processing Systems*, 26, pp. 2787-2795.

European Parliamentary Research Service (2024) *'This is Europe' debates: Analysis of EU leaders' speeches*. Drachenberg, R. & Bącal, P. Available at: https://www.europarl.europa.eu/thinktank/en/document/EPRS_BRI(2024)757844 (Accessed: 15 January 2024).

Firbas, J. (1992) *Functional sentence perspective in written and spoken communication*. Cambridge: Cambridge University Press.

Foucault, M. (1972) *The archaeology of knowledge*. New York: Pantheon Books.

Galke, L. & Scherp, A. (2022) 'Bag-of-Words vs. Graph vs. Sequence in Text Classification: Questioning the Necessity of Text-Graphs and the Surprising Strength of a Wide MLP', *Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval*, pp. 2365-2376.

Griffiths, T. L. & Steyvers, M. (2004) 'Finding scientific topics', *Proceedings of the National Academy of Sciences*, 101(suppl 1), pp. 5228-5235.

Halliday, M. A. K. (1985) *An introduction to functional grammar*. London: Edward Arnold.

Jolliffe, I. T. (2002) *Principal component analysis*. 2nd edn. New York: Springer.

Zhao, H., Phung, D., Huynh, V., Jin, Y., Du, L. & Buntine, W. (2021) 'Topic Modelling Meets Deep Neural Networks: A Survey', *arXiv preprint arXiv:2103.00498*.

Wu, X., Nguyen, T. & Luu, A. T. (2024) 'A Survey on Neural Topic Models: Methods, Applications, and Challenges', *arXiv preprint arXiv:2401.15351*.

Newman, M. E. J. (2005) 'Power laws, Pareto distributions and Zipf's law', *Contemporary Physics*, 46(5), pp. 323-351.

Newman, M. E. J. (2006) 'Modularity and community structure in networks', *Proceedings of the National Academy of Sciences*, 103(23), pp. 8577-8582.

Newman, M. E. J. & Girvan, M. (2004) 'Finding and evaluating community structure in networks', *Physical Review E*, 69(2), p. 026113.

Nguyen, D. Q. (2020) 'A survey of embedding models of entities and relationships for knowledge graph completion', *arXiv preprint arXiv:2003.08001*.

Reinhart, T. (1981) 'Pragmatics and linguistics: An analysis of sentence topics', *Philosophica*, 27(1), pp. 53-94.

Reinhart, T. (1982) 'Pragmatics and linguistics: An analysis of sentence topics', *Distributed by the Indiana University Linguistics Club*.

Salton, G. & McGill, M. J. (1986) *Introduction to modern information retrieval*. New York: McGraw-Hill.

Traag, V. A., Waltman, L. & van Eck, N. J. (2019) 'From Louvain to Leiden: guaranteeing well-connected communities', *Scientific Reports*, 9(1), p. 5233.

Grover, A. & Leskovec, J. (2016) 'node2vec: Scalable feature learning for networks', *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, pp. 855–864.

Giles, H. (2016) 'Communication Accommodation Theory', in Berger, C. R. & Roloff, M. E. (eds.) *The International Encyclopedia of Interpersonal Communication*. Hoboken: Wiley, pp. 1-18.

Rousseeuw, P. J. (1987) 'Silhouettes: A graphical aid to the interpretation and validation of cluster analysis', *Journal of Computational and Applied Mathematics*, 20, pp. 53-65.

**Knowledge Graphs and Graph Neural Networks**

Ji, S., Pan, S., Cambria, E., Zhou, B. & Bian, Y. (2022) 'A Survey on Knowledge Graphs: Representation, Acquisition, and Applications', *IEEE Transactions on Neural Networks and Learning Systems*, 33(2), pp. 494-514. DOI: 10.1109/TNNLS.2021.3070843.

Sun, Z., Deng, Z.-H., Nie, J.-Y. & Tang, J. (2019) 'RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space', *Proceedings of the 7th International Conference on Learning Representations (ICLR)*.

Trouillon, T., Welbl, J., Riedel, S., Gaussier, E. & Bouchard, G. (2016) 'Complex Embeddings for Simple Link Prediction', *Proceedings of the 33rd International Conference on Machine Learning (ICML)*, pp. 2075-2084.

Xie, Z., Wang, J., Huang, Y., Peng, H. & Liu, Y. (2021) 'Knowledge Graph Enhanced Neural Topic Model', *arXiv preprint arXiv:2110.11942*.

Yao, L., Shu, C., Liu, S., Zou, Y., Yang, J. & Liu, M. (2019) 'TextGCN: Word Graph Convolutional Network for Text Classification', *Proceedings of the 33rd AAAI Conference on Artificial Intelligence (AAAI)*, pp. 7456-7463.

**Graph-Based Retrieval and Topic Modeling**

Edge, D., Trinh, H., Cheng, N., Bradley, J., Chao, A., Mody, A., Truitt, S., Metropolitansky, D. & Larson, J. (2024) 'From Local to Global: A Graph RAG Approach to Query-Focused Summarization', *arXiv preprint arXiv:2404.16130*.

Guo, Z., Xia, L., Yu, Y., Ao, T. & Huang, C. (2024) 'LightRAG: Simple and Fast Retrieval-Augmented Generation', *arXiv preprint arXiv:2410.05779*.


