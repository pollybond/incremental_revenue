import os
import psycopg
from contextlib import contextmanager
from typing import Generator
import logging

logger = logging.getLogger(__name__)

@contextmanager
def get_gp_connection(
    host: str = os.getenv("GP_HOST"),
    port: int = int(os.getenv("GP_PORT", 5432)),
    dbname: str = os.getenv("GP_DB"),
    user: str = os.getenv("GP_USER"),
    password: str = os.getenv("GP_PASSWORD"),
) -> Generator[psycopg.Connection, None, None]:
    """
    Контекстный менеджер для подключения к Greenplum/PostgreSQL.
    Автоматически закрывает соединение при выходе из блока with.
    """
    if not all([host, dbname, user, password]):
        raise ValueError("❌ DB credentials are missing in .env")
        
    logger.info(f"🔌 Connecting to DB: {host}:{port}/{dbname}")
    conn = psycopg.connect(
        host=host,
        port=port,
        dbname=dbname,
        user=user,
        password=password,
        autocommit=True
    )
    try:
        yield conn
    finally:
        conn.close()
        logger.info("🔌 Connection closed")