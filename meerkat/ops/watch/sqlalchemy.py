"""Postgres cache."""
import hashlib
import logging
import uuid
from typing import Any, Dict, Union

logger = logging.getLogger("postgresql")
logger.setLevel(logging.WARNING)

import sqlalchemy  # type: ignore
from sqlalchemy import Column, Identity, Integer, Float, String, create_engine  # type: ignore
from sqlalchemy.engine import Engine
from sqlalchemy.ext.declarative import declarative_base  # type: ignore
from sqlalchemy.orm import sessionmaker  # type: ignore

from .abstract import WatchLogger

Base = declarative_base()


class EngineRun(Base):
    """The request table."""

    __tablename__ = "engine_runs"
    id = Column(String, primary_key=True)
    errand_run_id = Column(String)
    input = Column(String)
    output = Column(String)
    engine = Column(String)
    response = Column(String)
    cost = Column(Float)
    input_tokens = Column(Integer)
    output_tokens = Column(Integer)


class ErrandRun(Base):
    """Table of errand runs."""

    __tablename__ = "errand_runs"
    id = Column(String, primary_key=True)
    errand_id = Column(String)
    key = Column(String)
    parent_id = Column(String)
    engine = Column(String)


class Errand(Base):
    """The errands table."""

    __tablename__ = "errands"
    id = Column(String, primary_key=True)
    code = Column(String)
    name = Column(String)
    module = Column(String)


class ErrandRunInput(Base):
    """Inputs to errand runs."""

    __tablename__ = "errand_run_inputs"
    id = Column(String, primary_key=True)
    object_id = Column(String)
    errand_run_id = Column(String)
    key = Column(String)


class ErrandRunOutput(Base):
    """Outputs of errand runs."""

    __tablename__ = "errand_run_outputs"
    id = Column(String, primary_key=True)
    object_id = Column(String)
    errand_run_id = Column(String)
    key = Column(String)


class Object(Base):
    """The objects table."""

    __tablename__ = "objects"
    id = Column(String, primary_key=True)
    value = Column(String)


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

    def log_errand(
        self,
        code: str,
        name: str,
        module: str,
    ):
        # Query the `errands` table to see if the errand is already there
        # If not, add it
        response = self.session.query(Errand).filter_by(code=code).first()
        if response is not None:
            return response.id

        errand = Errand(
            id=str(uuid.uuid4()),
            code=code,
            name=name,
            module=module,
        )
        self.session.add(errand)
        self.commit()
        return errand.id

    def log_errand_start(
        self,
        errand_id: str,
        inputs: Dict[str, Any],
        engine: str,
    ) -> str:

        # Log to Tables: errand_runs, errand_run_inputs, objects

        # Add the errand run
        errand_run = ErrandRun(
            id=str(uuid.uuid4()),
            errand_id=errand_id,
            parent_id=None,
            engine=engine,
        )
        self.session.add(errand_run)

        # Add the inputs
        for key, value in inputs.items():
            obj = Object(id=str(uuid.uuid4()), value=value)
            self.session.add(obj)
            errand_run_input = ErrandRunInput(
                id=str(uuid.uuid4()),
                object_id=obj.id,
                errand_run_id=errand_run.id,
                key=key,
            )
            self.session.add(errand_run_input)
        print("here")
        self.commit()
        return errand_run.id

    def log_engine_run(
        self,
        errand_run_id: str,
        input: str,
        output: str,
        engine: str,
        response: str,
        cost: float,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        # Log to Tables: engine_runs
        self.session.add(
            EngineRun(
                id=str(uuid.uuid4()),
                errand_run_id=errand_run_id,
                input=input,

                engine=engine,
                response=response,
                cost=cost,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
        )
        self.commit()

    def log_errand_end(
        self,
        errand_run_id: str,
        outputs: Dict[str, Any],
    ) -> None:
        # Log to Tables: errand_runs, errand_run_outputs, objects

        # Add the outputs
        for key, value in outputs.items():
            obj = Object(id=str(uuid.uuid4()), value=value)
            self.session.add(obj)

            errand_run_output = ErrandRunOutput(
                id=str(uuid.uuid4()),
                object_id=obj.id,
                errand_run_id=errand_run_id,
                key=key,
            )
            self.session.add(errand_run_output)

        self.commit()

    def commit(self) -> None:
        """Commit any results."""
        self.session.commit()

    @classmethod
    def from_snowflake(cls, user: str, password: str, account_identifier: str):

        engine = create_engine(
            f"snowflake://{user}:{password}@{account_identifier}/meerkatlogs/public"
        )
        return cls(engine=engine)

    @classmethod
    def from_bigquery(cls, project: str, dataset: str):
        engine = create_engine(f"bigquery://{project}/{dataset}")
        return cls(engine=engine)
