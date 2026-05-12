---
title: EU Public Record Source Taxonomy
created: 2026-05-05
updated: 2026-05-05
type: concept
tags: [ingestion, sources, EU, taxonomy, scraping, adapters]
confidence: high
---

# EU Public Record Source Taxonomy

Complete taxonomy of publicly accessible EU institution documents suitable for political intelligence ingestion.
Guides adapter development priority in `services/search/adapters/`.

---

## Scoring Key
- **PIV**: Political Intelligence Value — HIGH / MED / LOW
- **Diff**: Scraping difficulty — EASY / MEDIUM / HARD

---

## 1. European Parliament

### 1.1 Plenary Debates
| Field | Value |
|---|---|
| Endpoint | `https://europarl.europa.eu/RegistreWeb/search/typedoc.htm?language=EN&typedoc=CRE` |
| API | OData + direct HTML |
| Format | HTML, XML |
| Frequency | ~80 sessions/year |
| Volume | ~500 debate docs/year |
| PIV | **HIGH** — verbatim speeches from 705 MEPs, real positions |
| Diff | EASY — existing adapter (`ep_debates.py`) |
| Example | `https://europarl.europa.eu/doceo/document/CRE-9-2024-03-13_EN.html` |

### 1.2 Resolutions & Adopted Texts
| Field | Value |
|---|---|
| Endpoint | `https://europarl.europa.eu/plenary/en/texts-adopted.html` |
| API | HTML scrape + OData `https://data.europarl.europa.eu/api/v1/documents?type=RESOLUTION_EP` |
| Format | HTML, PDF |
| Frequency | ~200 resolutions/year |
| PIV | **HIGH** — formal Parliament positions, legally significant |
| Diff | EASY — structured HTML, existing EP resolutions adapter |
| Example | `https://europarl.europa.eu/doceo/document/TA-9-2024-0137_EN.html` |

### 1.3 Committee Reports
| Field | Value |
|---|---|
| Endpoint | `https://europarl.europa.eu/committees/en/documents/search` |
| Format | HTML, PDF |
| Frequency | ~800 reports/year across 20+ committees |
| PIV | **HIGH** — detailed legislative analysis, committee positions, rapporteur opinions |
| Diff | MEDIUM — multiple committee sub-sites |
| Example | `https://europarl.europa.eu/doceo/document/A-9-2024-0097_EN.html` |

### 1.4 Amendments
| Field | Value |
|---|---|
| Endpoint | `https://data.europarl.europa.eu/api/v1/documents?type=AMCO` |
| Format | HTML, PDF |
| Volume | ~10,000/year |
| PIV | MED — granular legislative positions per MEP, useful for MEP profiling |
| Diff | MEDIUM |

### 1.5 Written Questions (Parliamentary Questions)
| Field | Value |
|---|---|
| Endpoint | `https://europarl.europa.eu/plenary/en/parliamentary-questions.html` |
| Format | HTML |
| Volume | ~5,000/year |
| PIV | MED — MEP concerns and Commission responses |
| Diff | EASY |

### 1.6 MEP Profiles & Voting Records
| Field | Value |
|---|---|
| Endpoint | `https://data.europarl.europa.eu/api/v1/meps` (REST JSON API) |
| Format | JSON |
| PIV | **HIGH** — voting records enable stance analysis, party alignment |
| Diff | EASY — clean JSON API |
| Example | `https://data.europarl.europa.eu/api/v1/meps/197450/votes` |

---

## 2. European Council

### 2.1 Summit Conclusions
| Field | Value |
|---|---|
| Endpoint | `https://www.consilium.europa.eu/en/meetings/european-council/` |
| Format | HTML, PDF |
| Frequency | 4–6 summits/year |
| PIV | **HIGH** — highest-level EU political direction, all 27 heads of state agreement |
| Diff | EASY — well-structured PDFs |
| Example | `https://www.consilium.europa.eu/media/68967/2024-10-euco-conclusions-en.pdf` |

### 2.2 Statements & Press Releases
| Field | Value |
|---|---|
| Endpoint | `https://www.consilium.europa.eu/en/press/press-releases/` |
| Format | HTML |
| Frequency | ~200/year |
| PIV | MED |
| Diff | EASY |

---

## 3. Council of the European Union

### 3.1 Legislative Acts & Decisions
| Field | Value |
|---|---|
| Endpoint | `https://data.consilium.europa.eu/doc/document/search` (REST API available) |
| Format | HTML, PDF, XML |
| Frequency | ~1,000 acts/year |
| PIV | **HIGH** — binding legislation, member-state positions |
| Diff | MEDIUM — Council document register has API |

### 3.2 Meeting Agendas & Provisional Agendas
| Field | Value |
|---|---|
| Endpoint | `https://www.consilium.europa.eu/en/meetings/calendar/` |
| Format | HTML, PDF |
| PIV | MED — signals upcoming policy priorities |
| Diff | EASY |

### 3.3 Press Releases & Outcome Documents
| Field | Value |
|---|---|
| Endpoint | `https://www.consilium.europa.eu/en/press/press-releases/` |
| Format | HTML |
| Frequency | ~500/year |
| PIV | MED |
| Diff | EASY |

---

## 4. European Commission

### 4.1 COM Documents (Communications, Proposals)
| Field | Value |
|---|---|
| Endpoint | `https://eur-lex.europa.eu/search.html?type=advanced&qid=&DTS_SUBDOM=COM_PRELEGISLATIVE` |
| API | EUR-Lex SPARQL / CELLAR API |
| Format | HTML, PDF |
| Frequency | ~500 COM docs/year |
| PIV | **HIGH** — legislative proposals, policy papers, impact assessments |
| Diff | MEDIUM — EUR-Lex search well-documented |
| Example | `https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=COM:2024:0146:FIN` |

### 4.2 Directives & Regulations (EUR-Lex OJ L)
| Field | Value |
|---|---|
| Endpoint | `https://eur-lex.europa.eu/oj/direct-access.html` — OJ L series |
| API | CELLAR SPARQL: `http://publications.europa.eu/webapi/rdf/sparql` |
| Format | HTML, XML (Akoma Ntoso), PDF |
| Frequency | ~1,200 legal acts/year |
| PIV | **HIGH** — binding law texts, definitively track regulatory change |
| Diff | MEDIUM — CELLAR API is powerful but complex |

### 4.3 Decisions
| Field | Value |
|---|---|
| Endpoint | EUR-Lex OJ L / CELLAR |
| Format | HTML, PDF |
| PIV | **HIGH** — individual binding acts (e.g. sanctions, competition decisions) |
| Diff | MEDIUM |

### 4.4 Press Releases & Speeches
| Field | Value |
|---|---|
| Endpoint | `https://ec.europa.eu/commission/presscorner/home/en` |
| Format | HTML |
| Frequency | ~5,000/year |
| PIV | MED — political signals, Commissioner positions |
| Diff | EASY — clean HTML |

### 4.5 DG RELEX / Foreign Policy Documents
| Field | Value |
|---|---|
| Endpoint | `https://eeas.europa.eu/topics/eu-foreign-affairs-council_en` |
| Format | HTML, PDF |
| PIV | **HIGH** for foreign policy analysis |
| Diff | MEDIUM |

---

## 5. EUR-Lex — Cross-Cutting

### 5.1 CELLAR SPARQL API
| Field | Value |
|---|---|
| Endpoint | `http://publications.europa.eu/webapi/rdf/sparql` |
| Format | SPARQL → JSON/XML |
| Coverage | All OJ content since 1952 |
| PIV | **HIGH** — programmatic access to entire EU legal corpus |
| Diff | MEDIUM — requires SPARQL knowledge |
| Adapter | `eurlex.py` (existing, extend with CELLAR SPARQL) |

### 5.2 OJ C Series (Information & Notices)
| Field | Value |
|---|---|
| Endpoint | EUR-Lex OJ C |
| Format | HTML, PDF |
| Content | State aid decisions, procurement notices, committee positions |
| PIV | MED |
| Diff | MEDIUM |

---

## 6. European Central Bank

### 6.1 Economic Bulletin
| Field | Value |
|---|---|
| Endpoint | `https://www.ecb.europa.eu/pub/economic-bulletin/html/index.en.html` |
| Format | HTML, PDF |
| Frequency | 8 issues/year |
| PIV | **HIGH** for economic policy analysis |
| Diff | EASY |
| Example | `https://www.ecb.europa.eu/pub/economic-bulletin/html/eb202402.en.html` |

### 6.2 ECB Opinions on EU Legislation
| Field | Value |
|---|---|
| Endpoint | `https://www.ecb.europa.eu/pub/consultations/html/index.en.html` |
| Format | HTML, PDF |
| Frequency | ~30/year |
| PIV | **HIGH** — ECB positions on proposed legislation |
| Diff | EASY |

### 6.3 Speeches
| Field | Value |
|---|---|
| Endpoint | `https://www.ecb.europa.eu/press/key/html/index.en.html` |
| Format | HTML |
| Frequency | ~200/year |
| PIV | MED |
| Diff | EASY |

---

## 7. European Court of Auditors

### 7.1 Special Reports
| Field | Value |
|---|---|
| Endpoint | `https://www.eca.europa.eu/en/publications?category=special_report` |
| Format | HTML, PDF |
| Frequency | ~25/year |
| PIV | **HIGH** — independent audit findings on EU spending effectiveness |
| Diff | EASY — clean structured PDFs |
| Example | `https://www.eca.europa.eu/en/publications/SR-2024-03` |

### 7.2 Annual Reports
| Field | Value |
|---|---|
| Endpoint | `https://www.eca.europa.eu/en/publications?category=annual_report` |
| Format | PDF, HTML |
| Frequency | 1/year |
| PIV | HIGH |
| Diff | EASY |

---

## 8. Court of Justice of the EU (CURIA)

### 8.1 Judgments
| Field | Value |
|---|---|
| Endpoint | `https://curia.europa.eu/juris/recherche.jsf` — case law search |
| Format | HTML, PDF |
| Frequency | ~800 judgments/year |
| PIV | **HIGH** — binding interpretations of EU law |
| Diff | MEDIUM — form-based search, requires session handling |
| API | `https://curia.europa.eu/juris/liste.jsf?language=en&jur=C,T,F&num=&dates=&docnodecision=docnodecision` |

### 8.2 Advocate General Opinions
| Field | Value |
|---|---|
| Endpoint | CURIA search (type=AG) |
| Format | HTML, PDF |
| Frequency | ~500/year |
| PIV | HIGH — predictive of ECJ direction |
| Diff | MEDIUM |

---

## 9. EP Research Service (EPRS)

### 9.1 Briefings & Studies
| Field | Value |
|---|---|
| Endpoint | `https://www.europarl.europa.eu/thinktank/en/search.html` |
| Format | HTML, PDF |
| Frequency | ~300 publications/year |
| PIV | **HIGH** — expert analysis on EU policy for MEPs |
| Diff | EASY — open access, clean PDFs |
| Example | `https://www.europarl.europa.eu/thinktank/en/document/EPRS_BRI(2024)762367` |

---

## 10. Committee of the Regions & EESC

### 10.1 Committee of the Regions Opinions
| Field | Value |
|---|---|
| Endpoint | `https://cor.europa.eu/en/our-work/opinions` |
| Format | HTML, PDF |
| Frequency | ~60/year |
| PIV | MED — sub-national government positions |
| Diff | EASY |

### 10.2 EESC Opinions
| Field | Value |
|---|---|
| Endpoint | `https://www.eesc.europa.eu/en/our-work/opinions-information-reports/opinions` |
| Format | HTML, PDF |
| Frequency | ~150/year |
| PIV | MED — civil society/employer/trade union positions |
| Diff | EASY |

---

## 11. Europol & Eurojust

### 11.1 Europol Public Reports
| Field | Value |
|---|---|
| Endpoint | `https://www.europol.europa.eu/publications-documents` |
| Format | PDF |
| Content | IOCTA (cyber threat), TE-SAT (terrorism), SOCTA (serious crime) |
| Frequency | ~20 public reports/year |
| PIV | **HIGH** for security/crime policy analysis |
| Diff | EASY — public PDFs |

---

## 12. EU Open Data Portal

### 12.1 datasets.europa.eu
| Field | Value |
|---|---|
| Endpoint | `https://data.europa.eu/api/hub/search/search?q=&filter=dataset&limit=10` (CKAN-compatible API) |
| Format | JSON metadata, CSV/RDF/XML data |
| Coverage | 1.4M+ datasets from all EU institutions |
| PIV | MED — structured data rather than political text |
| Diff | EASY — CKAN API well-documented |

---

## Adapter Development Priority

| Priority | Source | Reason |
|---|---|---|
| 1 | EP Resolutions | HIGH PIV, EASY, new adapter already exists |
| 2 | Council Conclusions | HIGH PIV, EASY, most politically significant docs |
| 3 | EC Press Releases + Speeches | HIGH volume, EASY, real-time political signals |
| 4 | EUR-Lex CELLAR SPARQL | Unlocks entire EU legal corpus programmatically |
| 5 | ECB Economic Bulletin | HIGH PIV for economic intelligence |
| 6 | ECA Special Reports | Independent audit view, HIGH PIV |
| 7 | EPRS Briefings | Policy analysis, HIGH PIV |
| 8 | CURIA Judgments | Legal binding text, MEDIUM diff |
| 9 | MEP Voting Records | Enables stance/alignment analysis |
| 10 | EP Committee Reports | Comprehensive, MEDIUM diff |
