-- Reference schema (for documentation / manual setup).
-- At runtime the backend uses Hibernate ddl-auto=update, which creates these
-- tables automatically. Run this file manually only if you prefer to manage
-- the schema yourself (then set spring.jpa.hibernate.ddl-auto=validate).

CREATE TABLE IF NOT EXISTS ml_model (
    id         BIGSERIAL PRIMARY KEY,
    name       VARCHAR(255) NOT NULL,
    version    VARCHAR(255) NOT NULL,
    framework  VARCHAR(255),
    input_dim  INTEGER,
    threshold  DOUBLE PRECISION,
    trained_at TIMESTAMPTZ,
    accuracy   DOUBLE PRECISION,
    notes      TEXT,
    CONSTRAINT uq_model_name_version UNIQUE (name, version)
);

CREATE TABLE IF NOT EXISTS analysis (
    id                   UUID PRIMARY KEY,
    model_id             BIGINT REFERENCES ml_model(id),
    created_at           TIMESTAMPTZ NOT NULL,
    original_filename    VARCHAR(255),
    content_type         VARCHAR(255),
    file_size_bytes      BIGINT,
    image_width          INTEGER,
    image_height         INTEGER,
    image_path           VARCHAR(1024),
    predicted_class      VARCHAR(64),
    probability_melanoma DOUBLE PRECISION,
    threshold_used       DOUBLE PRECISION,
    inference_status     VARCHAR(32) NOT NULL,
    inference_ms         INTEGER,
    error_message        TEXT
);

CREATE TABLE IF NOT EXISTS analysis_feature (
    id            BIGSERIAL PRIMARY KEY,
    analysis_id   UUID NOT NULL REFERENCES analysis(id) ON DELETE CASCADE,
    feature_name  VARCHAR(255) NOT NULL,
    feature_value DOUBLE PRECISION NOT NULL,
    feature_index INTEGER
);

CREATE INDEX IF NOT EXISTS idx_feature_analysis ON analysis_feature(analysis_id);
