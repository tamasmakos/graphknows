#!/usr/bin/env python3
"""
Smoke test: verify FalkorDB and pgvector (PostgreSQL) connectivity.

NOTE: This project uses FalkorDB (Redis-compatible Cypher graph DB), not Neo4j.
      There is no Neo4j service configured in docker-compose.yaml.

Run from workspace root with:
    python smoke_test.py
"""

import os
import sys
import socket


# ---------------------------------------------------------------------------
# Load .env from workspace root
# ---------------------------------------------------------------------------
def _load_dotenv(path: str = ".env") -> None:
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                val = val.strip().strip('"').strip("'")
                os.environ.setdefault(key.strip(), val)
    except FileNotFoundError:
        pass


_load_dotenv()

FALKORDB_HOST = os.environ.get("FALKORDB_HOST", "localhost")
FALKORDB_PORT = int(os.environ.get("FALKORDB_PORT", 6379))
POSTGRES_HOST = os.environ.get("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.environ.get("POSTGRES_PORT", 5432))
POSTGRES_DB = os.environ.get("POSTGRES_DB", "graphknows")
POSTGRES_USER = os.environ.get("POSTGRES_USER", "postgres")
POSTGRES_PASS = os.environ.get("POSTGRES_PASSWORD", "password")
NEO4J_HOST = os.environ.get("NEO4J_HOST", "localhost")
NEO4J_PORT = int(os.environ.get("NEO4J_PORT", 7687))
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASS = os.environ.get("NEO4J_PASSWORD", "password")

PASS = "\033[32m PASS\033[0m"
FAIL = "\033[31m FAIL\033[0m"

results: list[tuple[str, bool, str]] = []


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _tcp_reachable(host: str, port: int, timeout: int = 5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


# ---------------------------------------------------------------------------
# FalkorDB checks
# ---------------------------------------------------------------------------
print(f"\n=== FalkorDB  ({FALKORDB_HOST}:{FALKORDB_PORT}) ===")


def test_falkordb_tcp():
    ok = _tcp_reachable(FALKORDB_HOST, FALKORDB_PORT)
    results.append(("FalkorDB TCP reachable", ok, ""))
    print(f"  {'[PASS]' if ok else '[FAIL]'} TCP connect to {FALKORDB_HOST}:{FALKORDB_PORT}")
    return ok


def test_falkordb_ping():
    try:
        from falkordb import FalkorDB

        db = FalkorDB(host=FALKORDB_HOST, port=FALKORDB_PORT)
        db.connection.ping()
        results.append(("FalkorDB PING", True, ""))
        print("  [PASS] PING")
        return db
    except Exception as e:
        results.append(("FalkorDB PING", False, str(e)))
        print(f"  [FAIL] PING: {e}")
        return None


def test_falkordb_cypher(db):
    try:
        g = db.select_graph("smoke_test_probe")
        res = g.query("RETURN 1 AS n")
        val = res.result_set[0][0]
        ok = val == 1
        results.append(("FalkorDB Cypher RETURN 1", ok, ""))
        print(f"  {'[PASS]' if ok else '[FAIL]'} Cypher RETURN 1 → {val}")
    except Exception as e:
        results.append(("FalkorDB Cypher RETURN 1", False, str(e)))
        print(f"  [FAIL] Cypher: {e}")


def test_falkordb_list_graphs(db):
    try:
        graphs = db.list_graphs()
        results.append(("FalkorDB list graphs", True, ""))
        print(f"  [PASS] list_graphs() → {graphs}")
    except Exception as e:
        results.append(("FalkorDB list graphs", False, str(e)))
        print(f"  [FAIL] list_graphs(): {e}")


if test_falkordb_tcp():
    fdb = test_falkordb_ping()
    if fdb:
        test_falkordb_cypher(fdb)
        test_falkordb_list_graphs(fdb)
else:
    results.append(("FalkorDB PING", False, "TCP unreachable"))
    results.append(("FalkorDB Cypher RETURN 1", False, "TCP unreachable"))
    results.append(("FalkorDB list graphs", False, "TCP unreachable"))

# ---------------------------------------------------------------------------
# pgvector / PostgreSQL checks
# ---------------------------------------------------------------------------
print(f"\n=== pgvector / PostgreSQL  ({POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}) ===")


def test_pg_tcp():
    ok = _tcp_reachable(POSTGRES_HOST, POSTGRES_PORT)
    results.append(("pgvector TCP reachable", ok, ""))
    print(f"  {'[PASS]' if ok else '[FAIL]'} TCP connect to {POSTGRES_HOST}:{POSTGRES_PORT}")
    return ok


def test_pg_connect():
    try:
        import psycopg2

        conn = psycopg2.connect(
            host=POSTGRES_HOST,
            port=POSTGRES_PORT,
            dbname=POSTGRES_DB,
            user=POSTGRES_USER,
            password=POSTGRES_PASS,
            connect_timeout=5,
        )
        results.append(("pgvector DB connect", True, ""))
        print(f"  [PASS] connect to db='{POSTGRES_DB}'")
        return conn
    except Exception as e:
        results.append(("pgvector DB connect", False, str(e)))
        print(f"  [FAIL] connect: {e}")
        return None


def test_pg_server_version(conn):
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT version();")
            ver = cur.fetchone()[0]
        results.append(("pgvector server version", True, ""))
        print(f"  [PASS] server version: {ver.split(',')[0]}")
    except Exception as e:
        results.append(("pgvector server version", False, str(e)))
        print(f"  [FAIL] version query: {e}")


def test_pgvector_extension(conn):
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector';")
            row = cur.fetchone()
        if row:
            results.append(("pgvector extension present", True, ""))
            print(f"  [PASS] pgvector extension v{row[0]} is installed")
        else:
            # Auto-create it — the image ships with the extension, just needs enabling
            # Must use a fresh connection with autocommit to run CREATE EXTENSION
            import psycopg2 as _pg2

            _conn2 = _pg2.connect(
                host=POSTGRES_HOST,
                port=POSTGRES_PORT,
                dbname=POSTGRES_DB,
                user=POSTGRES_USER,
                password=POSTGRES_PASS,
                connect_timeout=5,
            )
            _conn2.autocommit = True
            with _conn2.cursor() as _cur:
                _cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            _conn2.close()
            results.append(("pgvector extension present", True, "(just enabled)"))
            print("  [PASS] pgvector extension enabled (CREATE EXTENSION vector)")
    except Exception as e:
        results.append(("pgvector extension present", False, str(e)))
        print(f"  [FAIL] extension check: {e}")


def test_pg_embeddings_table(conn):
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT to_regclass('hybrid_embeddings')::text;")
            exists = cur.fetchone()[0]
        if exists:
            cur2 = conn.cursor()
            cur2.execute("SELECT COUNT(*) FROM hybrid_embeddings;")
            count = cur2.fetchone()[0]
            cur2.close()
            results.append(("hybrid_embeddings table", True, ""))
            print(f"  [PASS] 'hybrid_embeddings' table exists ({count} rows)")
        else:
            # Not a failure — table is created on first pipeline run
            results.append(("hybrid_embeddings table", True, "not yet created"))
            print(
                "  [WARN] 'hybrid_embeddings' table not found yet (created on first pipeline run)"
            )
    except Exception as e:
        results.append(("hybrid_embeddings table", False, str(e)))
        print(f"  [FAIL] table check: {e}")


if test_pg_tcp():
    pg_conn = test_pg_connect()
    if pg_conn:
        test_pg_server_version(pg_conn)
        test_pgvector_extension(pg_conn)
        test_pg_embeddings_table(pg_conn)
        pg_conn.close()
else:
    results.append(("pgvector DB connect", False, "TCP unreachable"))
    results.append(("pgvector server version", False, "TCP unreachable"))
    results.append(("pgvector extension present", False, "TCP unreachable"))

# ---------------------------------------------------------------------------
# Neo4j checks
# ---------------------------------------------------------------------------
print(f"\n=== Neo4j  ({NEO4J_HOST}:{NEO4J_PORT}) ===")


def test_neo4j_tcp():
    ok = _tcp_reachable(NEO4J_HOST, NEO4J_PORT)
    results.append(("Neo4j TCP reachable", ok, ""))
    print(f"  {'[PASS]' if ok else '[FAIL]'} TCP connect to {NEO4J_HOST}:{NEO4J_PORT}")
    return ok


def test_neo4j_bolt():
    try:
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(
            f"bolt://{NEO4J_HOST}:{NEO4J_PORT}",
            auth=(NEO4J_USER, NEO4J_PASS),
        )
        driver.verify_connectivity()
        results.append(("Neo4j Bolt auth", True, ""))
        print(f"  [PASS] Bolt auth as '{NEO4J_USER}'")
        return driver
    except Exception as e:
        results.append(("Neo4j Bolt auth", False, str(e)))
        print(f"  [FAIL] Bolt auth: {e}")
        return None


def test_neo4j_cypher(driver):
    try:
        with driver.session() as session:
            result = session.run("RETURN 1 AS n")
            val = result.single()["n"]
        ok = val == 1
        results.append(("Neo4j Cypher RETURN 1", ok, ""))
        print(f"  {'[PASS]' if ok else '[FAIL]'} Cypher RETURN 1 → {val}")
    except Exception as e:
        results.append(("Neo4j Cypher RETURN 1", False, str(e)))
        print(f"  [FAIL] Cypher: {e}")


def test_neo4j_version(driver):
    try:
        with driver.session() as session:
            result = session.run(
                "CALL dbms.components() YIELD name, versions RETURN name, versions"
            )
            row = result.single()
            ver = row["versions"][0] if row else "unknown"
        results.append(("Neo4j version query", True, ""))
        print(f"  [PASS] version: {ver}")
    except Exception as e:
        results.append(("Neo4j version query", False, str(e)))
        print(f"  [FAIL] version: {e}")


if test_neo4j_tcp():
    neo4j_driver = test_neo4j_bolt()
    if neo4j_driver:
        test_neo4j_cypher(neo4j_driver)
        test_neo4j_version(neo4j_driver)
        neo4j_driver.close()
else:
    results.append(("Neo4j Bolt auth", False, "TCP unreachable"))
    results.append(("Neo4j Cypher RETURN 1", False, "TCP unreachable"))
    results.append(("Neo4j version query", False, "TCP unreachable"))

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n=== Summary ===")
passed = sum(1 for _, ok, _ in results if ok)
total = len(results)
for name, ok, detail in results:
    status = "[PASS]" if ok else "[FAIL]"
    suffix = f"  ({detail})" if detail and not ok else ""
    print(f"  {status} {name}{suffix}")

print(f"\n{passed}/{total} checks passed.")
if passed < total:
    sys.exit(1)
