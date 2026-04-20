# GraphKnows Admin Dashboard

A modern, high-performance dashboard for managing the Knowledge Graph pipeline and testing Agentic RAG.

## Features

- **🚀 Pipeline Control**: Trigger KG generation runs with configurable limits and clean-start options. Real-time log streaming via WebSockets.
- **🧠 Agent Chat**: Test the LlamaIndex agent with full visibility into the **Reasoning Chain**, **Tool Calls**, and **Retrieved Subgraphs**.
- **🌐 Graph Explorer**: Interactive 2D visualization of the knowledge graph with node inspection and neighbor expansion.
- **📊 Label Analytics**: Distribution of nodes across entity types and system health monitoring.

## Tech Stack

- **Frontend**: React, TypeScript, Vite, Tailwind CSS, Lucide Icons, Framer Motion.
- **Visualization**: `react-force-graph-2d` (D3-powered physics).
- **Backend**: FastAPI, WebSockets, Subprocess execution for the KG pipeline.

## Getting Started

### 1. Start the Management Backend
```bash
uvicorn src.dashboard.backend.main:app --host 0.0.0.0 --port 8001 --reload
```

### 2. Start the Frontend
```bash
cd src/dashboard/frontend
npm run dev
```

### 3. Ensure Agent Service is running
The dashboard communicates with the main agent service on port 8000.
```bash
uvicorn src.app.main:app --host 0.0.0.0 --port 8000 --reload
```

## Usage Tips

- **Agentic Mode**: Toggle Agentic mode in Chat to see the agent proactively use tools like `search_entities` or `get_timeline`.
- **Visualize Context**: After an agent response, click "Visualize Context" to see the exact subgraph the agent retrieved to answer your question.
- **Expand Nodes**: In the Graph Explorer, click any node to view its properties and expand its local neighborhood.
