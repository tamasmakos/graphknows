"""Idempotent migration: ensure chunks table has session_id and metadata columns.

Connects via PGVECTOR_DSN env var. Adds columns and indexes if missing.
Run standalone:
    python scripts/migrate_pgvector_add_session_id.py
    python scripts/migrate_pgvector_add_session_id.py --dry-run
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

import psycopg2

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("migrate_pgvector")


def check_column(cur, table: str, column: str) -> bool:
    """Check if a column exists in the given table.

    Args:
        cur: Database cursor.
        table: Table name.
        column: Column name to check.

    Returns:
        True if the column exists.
    """
    cur.execute(
        "SELECT COUNT(*) FROM information_schema.columns "
        "WHERE table_name = %s AND column_name = %s",
        (table, column),
    )
    return cur.fetchone()[0] > 0


def migrate(dsn: str, dry_run: bool = False) -> None:
    """Run the idempotent migration.

    Args:
        dsn: PostgreSQL DSN.
        dry_run: If True, print actions without executing.

    Raises:
        SystemExit: If the migration fails.
    """
    conn = psycopg2.connect(dsn)
    try:
        cur = conn.cursor()

        # Ensure vector extension exists
        if not dry_run:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.commit()
            logger.info("Vector extension ensured")
        else:
            logger.info("[DRY-RUN] CREATE EXTENSION IF NOT EXISTS vector")

        # Check if chunks table exists
        cur.execute(
            "SELECT COUNT(*) FROM information_schema.tables "
            "WHERE table_name = 'chunks'"
        )
        if cur.fetchone()[0] == 0:
            logger.warning("chunks table does not exist — nothing to migrate")
            cur.close()
            conn.close()
            return

        actions: list[str] = []

        if not check_column(cur, "chunks", "session_id"):
            actions.append(
                "ALTER TABLE chunks ADD COLUMN session_id TEXT NOT NULL DEFAULT 'legacy'"
            )

        if not check_column(cur, "chunks", "metadata"):
            actions.append(
                "ALTER TABLE chunks ADD COLUMN metadata JSONB"
            )

        if actions:
            for stmt in actions:
                logger.info("Executing: %s", stmt)
                if not dry_run:
                    cur.execute(stmt)
            if not dry_run:
                conn.commit()
                logger.info("Migration applied: %d column(s) added", len(actions))
            else:
                logger.info("[DRY-RUN] Would execute %d statement(s)", len(actions))
        else:
            logger.info("No migration needed — all columns already present")

        # Index on session_id
        has_idx = False
        cur.execute(
            "SELECT COUNT(*) FROM pg_indexes "
            "WHERE tablename = 'chunks' AND indexname = 'chunks_session_id_idx'"
        )
        if cur.fetchone()[0] == 0:
            idx_stmt = "CREATE INDEX IF NOT EXISTS chunks_session_id_idx ON chunks (session_id)"
            logger.info("Executing: %s", idx_stmt)
            if not dry_run:
                cur.execute(idx_stmt)
                conn.commit()
                logger.info("Session_id index created")
            else:
                logger.info("[DRY-RUN] Would create session_id index")
        else:
            logger.info("Session_id index already exists")

        cur.close()
    except Exception:
        logger.exception("Migration failed")
        conn.rollback()
        raise
    finally:
        conn.close()


def main() -> None:
    """Parse args and run migration."""
    parser = argparse.ArgumentParser(
        description="Migrate pgvector chunks table: add session_id and metadata columns"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without executing them",
    )
    args = parser.parse_args()

    dsn = os.environ.get("PGVECTOR_DSN")
    if not dsn:
        logger.error("PGVECTOR_DSN environment variable is not set")
        sys.exit(1)

    migrate(dsn, dry_run=args.dry_run)
    logger.info("Migration complete")


if __name__ == "__main__":
    main()
