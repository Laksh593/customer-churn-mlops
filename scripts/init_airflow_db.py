"""
scripts/init_airflow_db.py
~~~~~~~~~~~~~~~~~~~~~~~~~~
Idempotent database provisioning script for Airflow metadata on PostgreSQL.

Creates a dedicated PostgreSQL role (airflow_user) and database (airflow_db)
with least-privilege access, ensuring complete isolation from the production
application database (churn_db).
"""

from __future__ import annotations

import logging
import os
import sys
import time

import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("init_airflow_db")


def validate_airflow_web_credentials() -> None:
    """Validate Airflow web admin credentials to ensure safe configuration."""
    env = os.environ.get("ENVIRONMENT", "development").lower()
    www_user = os.environ.get("_AIRFLOW_WWW_USER_USERNAME", "airflow_admin").strip()
    www_pass = os.environ.get("_AIRFLOW_WWW_USER_PASSWORD", "").strip()

    if not www_pass:
        raise ValueError(
            "Missing Airflow web admin password. "
            "Please configure _AIRFLOW_WWW_USER_PASSWORD in your environment or .env file."
        )

    if www_user.lower() == "admin" and www_pass.lower() == "admin":
        raise ValueError(
            "Insecure default Airflow credentials ('admin' / 'admin') are prohibited. "
            "Please configure unique values for _AIRFLOW_WWW_USER_USERNAME and _AIRFLOW_WWW_USER_PASSWORD."
        )

    if env == "production" and (
        www_pass == "airflow_dev_password" or len(www_pass) < 8
    ):
        raise ValueError(
            "Production environment requires an explicit, secure _AIRFLOW_WWW_USER_PASSWORD (minimum 8 characters)."
        )

    logger.info(
        "Airflow web admin credentials validated successfully (user: '%s').", www_user
    )


def provision_airflow_database() -> None:
    """Connect as administrative user and provision dedicated Airflow database and role."""
    host = os.environ.get("POSTGRES_HOST", "postgres")
    port = int(os.environ.get("POSTGRES_PORT", "5432"))
    admin_user = os.environ.get("POSTGRES_USER", "churn_user")
    admin_password = os.environ.get("POSTGRES_PASSWORD", "")
    default_db = os.environ.get("POSTGRES_DB", "churn_db")
    target_db = os.environ.get("AIRFLOW_DB", "airflow_db")

    airflow_user = os.environ.get("AIRFLOW_DB_USER", "airflow_user")
    airflow_password = os.environ.get("AIRFLOW_DB_PASSWORD", "airflow_db_dev_password")

    logger.info(
        "Connecting to PostgreSQL as admin '%s' at %s:%d to provision dedicated Airflow role...",
        admin_user,
        host,
        port,
    )

    max_retries = 30
    retry_interval = 2

    for attempt in range(1, max_retries + 1):
        try:
            conn = psycopg2.connect(
                host=host,
                port=port,
                user=admin_user,
                password=admin_password,
                dbname=default_db,
                connect_timeout=5,
            )
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            cur = conn.cursor()

            # 1. Provision dedicated role for Airflow
            cur.execute("SELECT 1 FROM pg_roles WHERE rolname = %s;", (airflow_user,))
            if not cur.fetchone():
                logger.info(
                    "Creating dedicated Airflow PostgreSQL role '%s'...", airflow_user
                )
                cur.execute(
                    f'CREATE ROLE "{airflow_user}" WITH LOGIN PASSWORD %s;',
                    (airflow_password,),
                )
            else:
                logger.info(
                    "Dedicated Airflow PostgreSQL role '%s' already exists. Synchronizing password...",
                    airflow_user,
                )
                cur.execute(
                    f'ALTER ROLE "{airflow_user}" WITH PASSWORD %s;',
                    (airflow_password,),
                )

            # 2. Provision dedicated database for Airflow
            cur.execute("SELECT 1 FROM pg_database WHERE datname = %s;", (target_db,))
            if not cur.fetchone():
                logger.info(
                    "Creating dedicated Airflow database '%s' owned by '%s'...",
                    target_db,
                    airflow_user,
                )
                cur.execute(f'CREATE DATABASE "{target_db}" OWNER "{airflow_user}";')
            else:
                logger.info(
                    "Database '%s' exists. Ensuring ownership is set to '%s'...",
                    target_db,
                    airflow_user,
                )
                cur.execute(f'ALTER DATABASE "{target_db}" OWNER TO "{airflow_user}";')

            # 3. Restrict access: Ensure only application owner can connect to application database
            cur.execute(f'REVOKE CONNECT ON DATABASE "{default_db}" FROM PUBLIC;')
            cur.execute(f'GRANT CONNECT ON DATABASE "{default_db}" TO "{admin_user}";')
            cur.execute(f'REVOKE ALL ON DATABASE "{default_db}" FROM "{airflow_user}";')
            logger.info(
                "Restricted access on application database '%s' strictly to '%s'.",
                default_db,
                admin_user,
            )

            cur.close()
            conn.close()

            # 4. Connect to target database and grant full schema permissions
            conn_target = psycopg2.connect(
                host=host,
                port=port,
                user=admin_user,
                password=admin_password,
                dbname=target_db,
                connect_timeout=5,
            )
            conn_target.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            cur_target = conn_target.cursor()

            cur_target.execute(f'GRANT ALL ON SCHEMA public TO "{airflow_user}";')
            cur_target.execute(f'ALTER SCHEMA public OWNER TO "{airflow_user}";')

            # Ensure all existing tables in airflow_db are owned by airflow_user
            cur_target.execute(
                f"""
                DO $$
                DECLARE r RECORD;
                BEGIN
                    FOR r IN (SELECT tablename FROM pg_tables WHERE schemaname = 'public') LOOP
                        EXECUTE 'ALTER TABLE public.' || quote_ident(r.tablename) || ' OWNER TO "{airflow_user}"';
                    END LOOP;
                END $$;
            """
            )
            logger.info(
                "Schema ownership and table permissions on '%s' verified for '%s'.",
                target_db,
                airflow_user,
            )

            cur_target.close()
            conn_target.close()
            return
        except Exception as exc:
            logger.warning(
                "Attempt %d/%d failed: %s. Retrying in %ds...",
                attempt,
                max_retries,
                exc,
                retry_interval,
            )
            time.sleep(retry_interval)

    logger.error("Failed to provision Airflow database after %d attempts.", max_retries)
    sys.exit(1)


if __name__ == "__main__":
    validate_airflow_web_credentials()
    provision_airflow_database()
