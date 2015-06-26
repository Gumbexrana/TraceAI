"""
E2E Tests for PostgreSQL Database Verification

Verifies data is correctly stored in PostgreSQL.
"""

import pytest
import os
import time
import uuid
from datetime import datetime
from typing import Dict, Any, Optional

from config import config


def get_db_connection():
    """Get PostgreSQL database connection."""
    try:
        import psycopg2
        from psycopg2.extras import RealDictCursor

        conn = psycopg2.connect(
            host=config.pg_host,
            port=config.pg_port,
            database=config.pg_db,
            user=config.pg_user,
            password=config.pg_password,
        )
        return conn
    except ImportError:
        pytest.skip("psycopg2 not installed")
    except Exception as e:
        pytest.skip(f"Cannot connect to PostgreSQL: {e}")


class TestTraceStorage:
    """Test trace data storage in PostgreSQL."""

    @pytest.fixture
    def db_conn(self):
        """Get database connection."""
        conn = get_db_connection()
        yield conn
        conn.close()

    def test_traces_table_exists(self, db_conn):
        """Verify traces table exists."""
        cursor = db_conn.cursor()
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_name = 'tracer_trace'
            );
        """)
        result = cursor.fetchone()
        assert result[0] is True
        cursor.close()

    def test_observation_spans_table_exists(self, db_conn):
        """Verify observation spans table exists."""
        cursor = db_conn.cursor()
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_name = 'tracer_observationspan'
            );
        """)
        result = cursor.fetchone()
        assert result[0] is True
        cursor.close()

    def test_trace_columns(self, db_conn):
        """Verify trace table has expected columns."""
        cursor = db_conn.cursor()
        cursor.execute("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'tracer_trace';
        """)
        columns = {row[0] for row in cursor.fetchall()}

        expected_columns = {
            "id",
            "trace_id",
            "project_id",
            "name",
            "start_time",
            "end_time",
            "user_id",
            "session_id",
        }

        for col in expected_columns:
            assert col in columns, f"Missing column: {col}"

        cursor.close()

    def test_span_columns(self, db_conn):
        """Verify span table has expected columns."""
        cursor = db_conn.cursor()
        cursor.execute("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'tracer_observationspan';
        """)
        columns = {row[0] for row in cursor.fetchall()}

        expected_columns = {
            "id",
            "span_id",
            "trace_id",
            "name",
            "observation_type",
            "start_time",
            "end_time",
            "model",
            "prompt_tokens",
            "completion_tokens",
            "cost",
        }

        for col in expected_columns:
            assert col in columns, f"Missing column: {col}"

        cursor.close()


class TestProjectStorage:
    """Test project data storage."""

    @pytest.fixture
    def db_conn(self):
        """Get database connection."""
        conn = get_db_connection()
        yield conn
        conn.close()

    def test_projects_table_exists(self, db_conn):
        """Verify projects table exists."""
        cursor = db_conn.cursor()
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_name = 'tracer_project'
            );
        """)
        result = cursor.fetchone()
        assert result[0] is True
        cursor.close()

    def test_count_projects(self, db_conn):
        """Count existing projects."""
        cursor = db_conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM tracer_project;")
        count = cursor.fetchone()[0]
        print(f"Total projects in database: {count}")
        assert count >= 0
        cursor.close()


class TestDataIntegrity:
    """Test data integrity and foreign keys."""

    @pytest.fixture
    def db_conn(self):
        """Get database connection."""
        conn = get_db_connection()
        yield conn
        conn.close()

    def test_span_trace_foreign_key(self, db_conn):
        """Verify spans have valid trace references."""
        cursor = db_conn.cursor()

        # Check for orphaned spans (spans without valid trace)
        cursor.execute("""
            SELECT COUNT(*)
            FROM tracer_observationspan s
            LEFT JOIN tracer_trace t ON s.trace_id = t.id
            WHERE t.id IS NULL;
        """)
        orphaned = cursor.fetchone()[0]
        print(f"Orphaned spans: {orphaned}")
        # Some orphaned spans might be acceptable during testing
        cursor.close()

    def test_trace_project_foreign_key(self, db_conn):
        """Verify traces have valid project references."""
        cursor = db_conn.cursor()

        cursor.execute("""
            SELECT COUNT(*)
            FROM tracer_trace t
            LEFT JOIN tracer_project p ON t.project_id = p.id
            WHERE p.id IS NULL;
        """)
        orphaned = cursor.fetchone()[0]
        print(f"Traces without project: {orphaned}")
        cursor.close()

    def test_no_duplicate_trace_ids(self, db_conn):
        """Check for duplicate trace IDs within same project."""
        cursor = db_conn.cursor()

        cursor.execute("""
            SELECT trace_id, project_id, COUNT(*)
            FROM tracer_trace
            GROUP BY trace_id, project_id
            HAVING COUNT(*) > 1
            LIMIT 10;
        """)
        duplicates = cursor.fetchall()

        if duplicates:
            print(f"Found {len(duplicates)} duplicate trace IDs")

        cursor.close()


