CREATE TABLE IF NOT EXISTS ltm_topics (
    community_id   INTEGER PRIMARY KEY,
    summary        TEXT NOT NULL,
    embedding      vector(384) NOT NULL,
    entity_count   INTEGER,
    hub_entities   TEXT,   -- JSON array of top entity names
    updated_at     TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS ltm_topics_embedding_idx
    ON ltm_topics USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 10);
