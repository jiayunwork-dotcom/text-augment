import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from app.database import engine, Base
from app.models import db_models
from sqlalchemy import text, inspect
from sqlalchemy.ext.asyncio import AsyncConnection


async def migrate():
    print("Starting database migration...")

    async with engine.begin() as conn:
        inspector = await conn.run_sync(lambda c: inspect(c))

        existing_tables = inspector.get_table_names()
        print(f"Existing tables: {existing_tables}")

        if "augmentation_steps" not in existing_tables:
            print("Creating augmentation_steps table...")
            from sqlalchemy.schema import CreateTable
            await conn.execute(CreateTable(db_models.AugmentationStep.__table__))
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
                    await conn.execute(text(f"ALTER TABLE augmentation_tasks ADD COLUMN {col_name} {col_def}"))
                    print(f"  Column {col_name} added.")
                else:
                    print(f"  Column {col_name} already exists.")

        print("Migration completed successfully!")


if __name__ == "__main__":
    asyncio.run(migrate())
