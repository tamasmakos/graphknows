#!/usr/bin/env python3
"""Shared debug runner for database-oriented OpenCode tools."""
import json
import os
import re
import socket
import sys
from typing import Any

import psycopg2
from neo4j import GraphDatabase
from psycopg2.extras import RealDictCursor


def load_dotenv(path: str) -> None:
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as file_handle:
        for raw_line in file_handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    return repr(value)


def is_read_only_cypher(query: str) -> bool:
    forbidden = [
        r"\bcreate\b",
        r"\bmerge\b",
        r"\bdelete\b",
        r"\bdetach\s+delete\b",
        r"\bset\b",
        r"\bdrop\b",
        r"\bremove\b",
        r"\bload\s+csv\b",
        r"\bcall\s+dbms\b",
        r"\bcall\s+apoc\b",
    ]
    lowered = query.lower()
    return not any(re.search(pattern, lowered) for pattern in forbidden)


def is_read_only_sql(query: str) -> bool:
    lowered = query.strip().lower()
    if not lowered:
        return False
    if ";" in lowered[:-1]:
        return False
    forbidden = [
        r"\binsert\b",
        r"\bupdate\b",
        r"\bdelete\b",
        r"\bdrop\b",
        r"\balter\b",
        r"\bcreate\b",
        r"\btruncate\b",
        r"\bgrant\b",
        r"\brevoke\b",
        r"\bcomment\b",
        r"\bcopy\b",
    ]
    if any(re.search(pattern, lowered) for pattern in forbidden):
        return False
    return lowered.startswith("select") or lowered.startswith("with") or lowered.startswith("explain")


def neo4j_config() -> tuple[str, str, str]:
    return (
        os.environ.get("NEO4J_URI", "bolt://172.18.0.1:7687"),
        os.environ.get("NEO4J_USER", "neo4j"),
        os.environ.get("NEO4J_PASSWORD", "testpassword123"),
    )


def pgvector_config() -> dict[str, Any]:
    dsn = os.environ.get("PGVECTOR_DSN")
    if dsn:
        return {"dsn": dsn}
    return {
        "host": os.environ.get("PGVECTOR_HOST", "172.18.0.1"),
        "port": int(os.environ.get("PGVECTOR_PORT", 5432)),
        "dbname": os.environ.get("PGVECTOR_DB", "vectordb"),
        "user": os.environ.get("PGVECTOR_USER", "postgres"),
        "password": os.environ.get("PGVECTOR_PASSWORD", "pgvectorpass123"),
    }


def run_neo4j_query(argv: list[str]) -> dict[str, Any]:
    if len(argv) < 1:
        raise ValueError("missing query argument")
    query = argv[0].strip()
    database = argv[1].strip() if len(argv) > 1 else "neo4j"
    max_rows = int(argv[2]) if len(argv) > 2 else 25
    if not query:
        raise ValueError("query cannot be empty")
    if not is_read_only_cypher(query):
        raise ValueError("only read-only Cypher is allowed")

    uri, user, password = neo4j_config()
    with GraphDatabase.driver(uri, auth=(user, password)) as driver:
        result = driver.execute_query(query, database_=database)
        records = result.records[: max(1, max_rows)]

    rows = [{key: to_jsonable(record[key]) for key in record.keys()} for record in records]
    return {
        "uri": uri,
        "database": database,
        "returned_rows": len(rows),
        "rows": rows,
    }


def run_memgraph_query(argv: list[str]) -> dict[str, Any]:
    if len(argv) < 1:
        raise ValueError("missing query argument")
    query = argv[0].strip()
    max_rows = int(argv[1]) if len(argv) > 1 else 25
    if not query:
        raise ValueError("query cannot be empty")
    if not is_read_only_cypher(query):
        raise ValueError("only read-only Cypher is allowed")

    uri = os.environ.get("MEMGRAPH_URI", "bolt://172.18.0.1:7688")
    with GraphDatabase.driver(uri, auth=None) as driver:
        with driver.session() as session:
            result = session.run(query)
            rows = []
            for index, record in enumerate(result):
                if index >= max(1, max_rows):
                    break
                rows.append({key: to_jsonable(record[key]) for key in record.keys()})

    return {
        "uri": uri,
        "returned_rows": len(rows),
        "rows": rows,
    }


def run_pgvector_query(argv: list[str]) -> dict[str, Any]:
    if len(argv) < 1:
        raise ValueError("missing query argument")
    query = argv[0].strip()
    max_rows = int(argv[1]) if len(argv) > 1 else 25
    if not query:
        raise ValueError("query cannot be empty")
    if not is_read_only_sql(query):
        raise ValueError("only read-only SQL is allowed")

    kwargs = pgvector_config()
    with psycopg2.connect(**kwargs) as connection:
        with connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(query)
            rows = cursor.fetchmany(max(1, max_rows))

    json_rows = [{key: to_jsonable(value) for key, value in row.items()} for row in rows]
    return {
        "database": kwargs.get("dbname", "from_dsn"),
        "host": kwargs.get("host", "from_dsn"),
        "returned_rows": len(json_rows),
        "rows": json_rows,
    }


def run_neo4j_status(_argv: list[str]) -> dict[str, Any]:
    uri, user, password = neo4j_config()
    host = "unknown"
    port = "unknown"
    if uri.startswith("bolt://"):
        host_port = uri.removeprefix("bolt://")
        if ":" in host_port:
            host, port = host_port.rsplit(":", 1)
    info = {
        "NEO4J_URI": uri,
        "NEO4J_USER": user,
        "NEO4J_PASSWORD": "***" if password else "not set",
        "host": host,
        "port": port,
    }

    try:
        with GraphDatabase.driver(uri, auth=(user, password)) as driver:
            driver.verify_connectivity()
        info["status"] = "reachable"
    except Exception as error:
        info["status"] = "unreachable"
        info["error"] = str(error)
        try:
            with socket.create_connection((host, int(port)), timeout=1):
                info["socket"] = "open"
        except Exception:
            info["socket"] = "closed"
    return info


COMMANDS = {
    "neo4j-query": run_neo4j_query,
    "memgraph-query": run_memgraph_query,
    "pgvector-query": run_pgvector_query,
    "neo4j-status": run_neo4j_status,
}


def main() -> int:
    load_dotenv(os.path.join(os.getcwd(), ".env"))
    if len(sys.argv) < 2:
        print(json.dumps({"error": "missing command"}))
        return 1

    command = sys.argv[1]
    handler = COMMANDS.get(command)
    if handler is None:
        print(json.dumps({"error": f"unknown command: {command}"}))
        return 1

    try:
        payload = handler(sys.argv[2:])
    except Exception as error:
        print(json.dumps({"error": str(error), "command": command}, indent=2))
        return 1

    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())