CREATE TABLE model_registry (
    id BIGSERIAL PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    model_category VARCHAR(50) NOT NULL,
    nutrient_group VARCHAR(50) NOT NULL,
    target_nutrient VARCHAR(100),
    summary TEXT,
    tuned_parameters JSONB,
    model_file_path TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


CREATE TABLE model_metrics (
    id BIGSERIAL PRIMARY KEY,

    model_id BIGINT NOT NULL
        REFERENCES model_registry(id)
        ON DELETE CASCADE,

    mae DOUBLE PRECISION,
    rmse DOUBLE PRECISION,
    r2 DOUBLE PRECISION,
    mape DOUBLE PRECISION,
    nse DOUBLE PRECISION,
    bias DOUBLE PRECISION,
    explained_variance DOUBLE PRECISION,

    training_time_seconds DOUBLE PRECISION,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


