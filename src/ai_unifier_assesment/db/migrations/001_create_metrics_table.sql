CREATE TABLE IF NOT EXISTS metrics (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    endpoint VARCHAR(50) NOT NULL,
    session_id VARCHAR(255),
    prompt_tokens INTEGER NOT NULL,
    completion_tokens INTEGER NOT NULL,
    total_tokens INTEGER NOT NULL,
    cost DECIMAL(10, 8) NOT NULL,
    latency_ms DECIMAL(10, 2) NOT NULL,
    metadata JSONB
);

CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON metrics(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_metrics_endpoint ON metrics(endpoint);
CREATE INDEX IF NOT EXISTS idx_metrics_session_id ON metrics(session_id);
