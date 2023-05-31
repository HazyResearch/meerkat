"""Postgres cache."""
import datetime
import logging
import threading
import uuid
from datetime import datetime
from typing import Any, Dict, Union, Optional

from meerkat import DataFrame

logger = logging.getLogger("postgresql")
logger.setLevel(logging.WARNING)

import sqlalchemy  # type: ignore, noqa: E402
from sqlalchemy import Column, Float, Integer, String, create_engine  # type: ignore
from sqlalchemy.engine import Engine  # noqa: E402
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
    cost = Column(Float)
    input_tokens = Column(Integer)
    output_tokens = Column(Integer)
    created_at = Column(String)
    configuration = Column(String)
    hash = Column(String)


class ErrandRun(Base):
    """Table of errand runs."""

    __tablename__ = "errand_runs"
    id = Column(String, primary_key=True)
    errand_id = Column(String)
    parent_id = Column(String)
    engine = Column(String)
    time = Column(String)


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

    def __init__(self, *args: Any, value: Any, **kwargs: Any) -> None:
        value = str(value)
        super().__init__(*args, value=value, **kwargs)


TABLE_TO_MODEL = {
    "engine_runs": EngineRun,
    "errand_runs": ErrandRun,
    "errands": Errand,
    "errand_run_inputs": ErrandRunInput,
    "errand_run_outputs": ErrandRunOutput,
    "objects": Object,
}


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
        self.sql_engine = engine
        db_exists = len(sqlalchemy.inspect(engine).get_table_names()) > 0
        if not db_exists:
            logger.info("Creating database...")
        Base.metadata.create_all(engine)

        self.sessionmaker = sessionmaker(bind=engine)
        self.thread_id = threading.get_ident()

    def log_errand(
        self,
        code: str,
        name: str,
        module: str,
    ):
        # Query the `errands` table to see if the errand is already there
        # If not, add it
        session = self.session
        with self.sessionmaker():
            response = session.query(Errand).filter_by(code=code).first()
            if response is not None:
                return response.id

            errand = Errand(
                id=str(uuid.uuid4()),
                code=code,
                name=name,
                module=module,
            )
            session.add(errand)
            session.commit()
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
            time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )
        with self.sessionmaker() as session:
            session.add(errand_run)

            # Add the inputs
            for key, value in inputs.items():
                obj = Object(id=str(uuid.uuid4()), value=value)
                session.add(obj)
                errand_run_input = ErrandRunInput(
                    id=str(uuid.uuid4()),
                    object_id=obj.id,
                    errand_run_id=errand_run.id,
                    key=key,
                )
                session.add(errand_run_input)
            session.commit()
        return errand_run.id

    def _hash_run(self, input: str, engine: str, configuration: Dict[str, Any]):
        from hashlib import sha256
        import json

        return sha256(
            json.dumps(
                {"input": input, "engine": engine, "configuration": configuration}
            ).encode("utf-8")
        ).hexdigest()

    def retrieve_engine_run(
        self, input: str, engine: str, configuration: Dict[str, Any]
    ) -> Optional[EngineRun]:
        """Retrieve the most recent engine run for a given input and engine."""
        # Query the `engine_runs` table to see if the engine run is already there
        # If not, return None
        hashed = self._hash_run(input=input, configuration=configuration, engine=engine)
        with self.sessionmaker() as session:
            response = (
                session.query(EngineRun)
                .filter_by(hash=hashed)
                .order_by(EngineRun.created_at.desc())
                .first()
            )
            if response is not None:
                return response
            else:
                return None

    def log_engine_run(
        self,
        errand_run_id: str,
        input: str,
        output: str,
        engine: str,
        cost: float,
        input_tokens: int,
        output_tokens: int,
        configuration: Dict[str, Any],
    ) -> None:
        hashed = self._hash_run(input=input, configuration=configuration, engine=engine)

        # Log to Tables: engine_runs
        with self.sessionmaker() as session:
            session.add(
                EngineRun(
                    id=str(uuid.uuid4()),
                    errand_run_id=errand_run_id,
                    input=input,
                    output=output,
                    engine=engine,
                    cost=cost,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    configuration=configuration,
                    hash=hashed,
                )
            )
            session.commit()

    def log_errand_end(
        self,
        errand_run_id: str,
        outputs: Dict[str, Any],
    ) -> None:
        # Log to Tables: errand_runs, errand_run_outputs, objects

        # Add the outputs
        with self.sessionmaker() as session:
            for key, value in outputs.items():
                obj = Object(id=str(uuid.uuid4()), value=value)
                session.add(obj)

                errand_run_output = ErrandRunOutput(
                    id=str(uuid.uuid4()),
                    object_id=obj.id,
                    errand_run_id=errand_run_id,
                    key=key,
                )
                session.add(errand_run_output)

            session.commit()

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

    @classmethod
    def from_sqlite(cls, path: str):
        engine = create_engine(f"sqlite:///{path}")
        return cls(engine=engine)

    @classmethod
    def from_google_cloud_sql(
        cls,
        instance_connection_string: str,
        username: str,
        password: str,
        database_name: str,
    ):
        """Create a new instance of the class from a Google Cloud SQL connection.

        This method establishes a connection to a Google Cloud SQL database instance
        and returns a new instance of the class, configured to use this connection.

        Args:
            instance_connection_string (str): The connection string of the Google Cloud SQL instance.
                This should include the instance's project, region, and name.
            username (str): The username to authenticate with the database instance.
            password (str): The password to authenticate with the database instance.
            database_name (str): The name of the specific database to connect to within the instance.

        Returns:
            cls: A new instance of the class, configured to use the provided Google Cloud SQL connection.
        """
        from google.cloud.sql.connector import Connector

        # initialize Cloud SQL Python Connector object
        connector = Connector()

        def getconn():
            conn = connector.connect(
                instance_connection_string,
                "pg8000",
                user=username,
                password=password,
                db=database_name,
            )
            return conn

        engine = sqlalchemy.create_engine(
            "postgresql+pg8000://",
            creator=getconn,
            pool_size=5,
            max_overflow=-1,
        )
        return cls(engine=engine)

    def get_table(self, model: Union[type, str]):
        if isinstance(model, str):
            model = TABLE_TO_MODEL[model]

        with self.sessionmaker() as session:
            result = session.query(model).all()

            records = [
                {c.name: getattr(row, c.name) for c in model.__table__.columns}
                for row in result
            ]
            if len(records) == 0:
                raise ValueError("No records found.")
            return DataFrame(records)
