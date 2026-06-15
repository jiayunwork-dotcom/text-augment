import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from sqlalchemy import create_engine, text, inspect, MetaData, Table, Column, Integer, String, JSON, Boolean, ForeignKey, DateTime, Float, Enum, Text, UniqueConstraint
from sqlalchemy.schema import CreateTable
from app.config import DATABASE_URL


def main():
    print("Starting database migration...")

    sync_url = DATABASE_URL.replace("sqlite+aiosqlite://", "sqlite://")
    engine = create_engine(sync_url)
    inspector = inspect(engine)

    existing_tables = inspector.get_table_names()
    print(f"Existing tables: {existing_tables}")

    with engine.begin() as conn:
        if "augmentation_steps" not in existing_tables:
            print("Creating augmentation_steps table...")
            metadata = MetaData()
            augmentation_steps = Table(
                "augmentation_steps",
                metadata,
                Column("id", Integer, primary_key=True, autoincrement=True),
                Column("task_id", Integer, ForeignKey("augmentation_tasks.id", ondelete="CASCADE"), nullable=False),
                Column("step_order", Integer, nullable=False),
                Column("strategy", String(50), nullable=False),
                Column("strategy_params", JSON, default=dict),
                Column("input_count", Integer, default=0),
                Column("success_count", Integer, default=0),
                Column("skipped_count", Integer, default=0),
            )
            conn.execute(CreateTable(augmentation_steps))
            print("  augmentation_steps table created.")
        else:
            print("  augmentation_steps table already exists.")

        if "augmentation_tasks" in existing_tables:
            columns = [col["name"] for col in inspector.get_columns("augmentation_tasks")]
            print(f"Existing columns in augmentation_tasks: {columns}")

            new_columns = [
                ("is_composite", "BOOLEAN DEFAULT 0"),
                ("current_step_index", "INTEGER DEFAULT 0"),
                ("step_stats", "TEXT DEFAULT '[]'"),
            ]

            for col_name, col_def in new_columns:
                if col_name not in columns:
                    print(f"Adding column {col_name}...")
                    conn.execute(text(f"ALTER TABLE augmentation_tasks ADD COLUMN {col_name} {col_def}"))
                    print(f"  Column {col_name} added.")
                else:
                    print(f"  Column {col_name} already exists.")

        if "annotation_queues" not in existing_tables:
            print("Creating annotation_queues table...")
            conn.execute(text("""
                CREATE TABLE annotation_queues (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    version_id INTEGER NOT NULL,
                    name VARCHAR(255) NOT NULL,
                    status VARCHAR(20) DEFAULT 'pending',
                    capacity INTEGER DEFAULT 100,
                    review_mode VARCHAR(20) DEFAULT 'single',
                    num_reviewers INTEGER DEFAULT 1,
                    lock_timeout_minutes INTEGER DEFAULT 30,
                    created_by VARCHAR(100),
                    created_at DATETIME,
                    started_at DATETIME,
                    completed_at DATETIME,
                    applied_at DATETIME,
                    target_version_id INTEGER,
                    FOREIGN KEY (version_id) REFERENCES dataset_versions(id) ON DELETE CASCADE,
                    FOREIGN KEY (target_version_id) REFERENCES dataset_versions(id)
                )
            """))
            print("  annotation_queues table created.")
        else:
            print("  annotation_queues table already exists.")

        if "annotation_items" not in existing_tables:
            print("Creating annotation_items table...")
            conn.execute(text("""
                CREATE TABLE annotation_items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    queue_id INTEGER NOT NULL,
                    sample_id INTEGER NOT NULL,
                    uncertainty_score FLOAT DEFAULT 0.0,
                    status VARCHAR(20) DEFAULT 'pending',
                    locked_by VARCHAR(100),
                    locked_at DATETIME,
                    final_decision VARCHAR(20),
                    final_label VARCHAR(100),
                    arbitrated_by VARCHAR(100),
                    arbitrated_at DATETIME,
                    FOREIGN KEY (queue_id) REFERENCES annotation_queues(id) ON DELETE CASCADE,
                    FOREIGN KEY (sample_id) REFERENCES samples(id) ON DELETE CASCADE,
                    UNIQUE (queue_id, sample_id)
                )
            """))
            print("  annotation_items table created.")
        else:
            print("  annotation_items table already exists.")

        if "annotation_records" not in existing_tables:
            print("Creating annotation_records table...")
            conn.execute(text("""
                CREATE TABLE annotation_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    item_id INTEGER NOT NULL,
                    annotator_id VARCHAR(100) NOT NULL,
                    decision VARCHAR(20) NOT NULL,
                    new_label VARCHAR(100),
                    comment TEXT,
                    created_at DATETIME,
                    FOREIGN KEY (item_id) REFERENCES annotation_items(id) ON DELETE CASCADE,
                    UNIQUE (item_id, annotator_id)
                )
            """))
            print("  annotation_records table created.")
        else:
            print("  annotation_records table already exists.")

        if "annotation_queues" in existing_tables:
            columns = [col["name"] for col in inspector.get_columns("annotation_queues")]
            print(f"Existing columns in annotation_queues: {columns}")

            new_columns = [
                ("priority_strategy", "VARCHAR(20) DEFAULT 'uncertainty'"),
                ("webhook_url", "VARCHAR(500)"),
                ("webhook_thresholds", "TEXT DEFAULT '[]'"),
                ("triggered_thresholds", "TEXT DEFAULT '[]'"),
            ]

            for col_name, col_def in new_columns:
                if col_name not in columns:
                    print(f"Adding column {col_name}...")
                    conn.execute(text(f"ALTER TABLE annotation_queues ADD COLUMN {col_name} {col_def}"))
                    print(f"  Column {col_name} added.")
                else:
                    print(f"  Column {col_name} already exists.")

        if "annotation_records" in existing_tables:
            columns = [col["name"] for col in inspector.get_columns("annotation_records")]
            print(f"Existing columns in annotation_records: {columns}")

            new_columns = [
                ("locked_at", "DATETIME"),
                ("submitted_at", "DATETIME"),
                ("annotation_duration_seconds", "FLOAT"),
                ("is_final_decision", "BOOLEAN DEFAULT 0"),
            ]

            for col_name, col_def in new_columns:
                if col_name not in columns:
                    print(f"Adding column {col_name}...")
                    conn.execute(text(f"ALTER TABLE annotation_records ADD COLUMN {col_name} {col_def}"))
                    print(f"  Column {col_name} added.")
                else:
                    print(f"  Column {col_name} already exists.")

        if "webhook_logs" not in existing_tables:
            print("Creating webhook_logs table...")
            conn.execute(text("""
                CREATE TABLE webhook_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    queue_id INTEGER NOT NULL,
                    threshold FLOAT NOT NULL,
                    url VARCHAR(500) NOT NULL,
                    status_code INTEGER,
                    success BOOLEAN DEFAULT 0,
                    response_body TEXT,
                    error_message TEXT,
                    created_at DATETIME,
                    FOREIGN KEY (queue_id) REFERENCES annotation_queues(id) ON DELETE CASCADE
                )
            """))
            print("  webhook_logs table created.")
        else:
            print("  webhook_logs table already exists.")

        if "recommended_filter_configs" not in existing_tables:
            print("Creating recommended_filter_configs table...")
            conn.execute(text("""
                CREATE TABLE recommended_filter_configs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    version_id INTEGER NOT NULL,
                    queue_id INTEGER,
                    source_config_name VARCHAR(50) DEFAULT 'standard',
                    ppl_multiplier FLOAT,
                    similarity_threshold FLOAT,
                    jaccard_threshold FLOAT,
                    label_confidence_threshold FLOAT,
                    adjustments TEXT DEFAULT '{}',
                    reasoning TEXT,
                    created_at DATETIME,
                    FOREIGN KEY (version_id) REFERENCES dataset_versions(id) ON DELETE CASCADE,
                    FOREIGN KEY (queue_id) REFERENCES annotation_queues(id)
                )
            """))
            print("  recommended_filter_configs table created.")
        else:
            print("  recommended_filter_configs table already exists.")

        if "ml_training_tasks" not in existing_tables:
            print("Creating ml_training_tasks table...")
            conn.execute(text("""
                CREATE TABLE ml_training_tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_name VARCHAR(255) NOT NULL,
                    dataset_id INTEGER NOT NULL,
                    annotated_version_id INTEGER NOT NULL,
                    model_type VARCHAR(50) NOT NULL,
                    hyperparams TEXT DEFAULT '{}',
                    split_ratios TEXT DEFAULT '{}',
                    status VARCHAR(20) DEFAULT 'pending',
                    train_loss_history TEXT DEFAULT '[]',
                    train_acc_history TEXT DEFAULT '[]',
                    val_loss_history TEXT DEFAULT '[]',
                    val_acc_history TEXT DEFAULT '[]',
                    model_path VARCHAR(500),
                    model_size_bytes INTEGER,
                    training_duration_seconds FLOAT,
                    error_message TEXT,
                    started_at DATETIME,
                    completed_at DATETIME,
                    created_at DATETIME,
                    FOREIGN KEY (dataset_id) REFERENCES datasets(id) ON DELETE CASCADE,
                    FOREIGN KEY (annotated_version_id) REFERENCES dataset_versions(id)
                )
            """))
            print("  ml_training_tasks table created.")
        else:
            print("  ml_training_tasks table already exists.")

        if "ml_training_reports" not in existing_tables:
            print("Creating ml_training_reports table...")
            conn.execute(text("""
                CREATE TABLE ml_training_reports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id INTEGER NOT NULL UNIQUE,
                    accuracy FLOAT DEFAULT 0.0,
                    weighted_f1 FLOAT DEFAULT 0.0,
                    per_class_metrics TEXT DEFAULT '{}',
                    confusion_matrix TEXT DEFAULT '[]',
                    class_names TEXT DEFAULT '[]',
                    created_at DATETIME,
                    FOREIGN KEY (task_id) REFERENCES ml_training_tasks(id) ON DELETE CASCADE
                )
            """))
            print("  ml_training_reports table created.")
        else:
            print("  ml_training_reports table already exists.")

    print("Migration completed successfully!")


if __name__ == "__main__":
    main()
