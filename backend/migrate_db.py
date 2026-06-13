import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from sqlalchemy import create_engine, text, inspect, MetaData, Table, Column, Integer, String, JSON, Boolean, ForeignKey
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

    print("Migration completed successfully!")


if __name__ == "__main__":
    main()
