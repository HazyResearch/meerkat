"""Postgres cache."""
import hashlib
import logging
from typing import Any, Dict, Union
import uuid

logger = logging.getLogger("postgresql")
logger.setLevel(logging.WARNING)

from .abstract import WatchLogger

import sqlalchemy  # type: ignore
from sqlalchemy.engine import Engine
# from google.cloud.sql.connector import Connector  # type: ignore
from sqlalchemy import Column, String, Integer, Identity  # type: ignore
from sqlalchemy.ext.declarative import declarative_base  # type: ignore
from sqlalchemy.orm import sessionmaker  # type: ignore

Base = declarative_base()


class Query(Base):  # type: ignore
    """The request table."""

    __tablename__ = "queries"
    id = Column(String, primary_key=True)
    input = Column(String)
    response = Column(String)
    engine = Column(String)


class SQLAlchemyWatchLogger(WatchLogger):
    """A PostgreSQL cache for request/response pairs."""


    def __init__(self, engine: Engine):
        """
        Connect to client.

        Args:
            connection_str: connection string.
            cache_args: arguments for cache should include the following fields:
                {
                    "cache_user": "",
                    "cache_password": "",
                    "cache_db": ""
                }
        """
        db_exists = len(sqlalchemy.inspect(engine).get_table_names()) > 0
        if not db_exists:
            logger.info("Creating database...")
        Base.metadata.create_all(engine)

        self.session = sessionmaker(bind=engine)()

    def close(self) -> None:
        """Close the client."""
        self.session.close()

    def log(self, input: str, response: str, engine: str) -> None:
        """
        Set the value for the key.

        Will override old value.

        Args:
            key: key for cache.
            value: new value for key.
            table: table to set key in.
        """
        self.session.add(Query(
            id=str(uuid.uuid4()),
            input=input, response=response, engine=engine
        ))
        self.commit()

    def commit(self) -> None:
        """Commit any results."""
        self.session.commit()
