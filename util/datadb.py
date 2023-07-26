"""
Module for helper functions to create sqlite3 database and tables from directory
of CSV files.
"""
from pandas import DataFrame, read_csv, read_sql
from pathlib import Path
from sqlite3 import Connection, Cursor, connect
from typing import Any

from .path import ProjectPath


DATADB_FILENAME = "data.db"


def _data_csv_paths() -> list[Path]:
    paths = ProjectPath()
    assert hasattr(paths, "data")

    csv_paths = []
    for path in paths.data.iterdir():
        if ".csv" in path.name:
            csv_paths.append(path)

    return csv_paths


def _csv_to_table(csv_path: Path, connection: Connection) -> None:
    table_name = csv_path.stem.replace("-", "_")

    dataframe = read_csv(csv_path)
    dataframe.to_sql(table_name, connection, if_exists="replace", index=False)


class DataDB:
    """
    Class for SQLite3 database with data from CSV files in the project's `data`
    directory.

    Attributes:
        datadb_path (Path): Absolute path to `data.db` database file in projects
            root directory if it exists else None.
        connection (sqlite3.Connection): Connection to sqlite3 database.
        cursor (sqlite3.Cursor): Cursor for sqlite3 database.
        tables (list[str]): List of table names in sqlite3 database.
    """

    def __init__(self):
        self.datadb_path = None
        self.connection = None
        self.cursor = None
        self._tables = None

    def _datadb_path(self) -> Path:
        """
        Returns absolute path to `data.db` database file in projects root
        directory if it exists else None.
        """
        paths = ProjectPath()
        db_file_path = paths.root.joinpath(DATA_DB_FILENAME)

        return db_file_path if db_file_path.exists() else None

    def set_connection(self, use_file: bool = False, reset: bool = False) -> Connection:
        """
        Creates a sqlite3 database with data from CSV files in the project's
        `/data` directory and returns the connection. The default is to create
        the database in memory. If `use_file=True`, the database is created
        as a `data.db` file in the project's root directory if it doesn't
        already exist.

        After the database is established, the connection is stored in the
        `connection` class attribute

        Args:
            use_file (bool): Connect to `data.db` file in project's root directory if
                it exists or create a new `data.db` file in project's root directory
                from CSV files in projects `/data` directory.

            reset (bool): If True, reset the connection to the database.
        """
        if not self.connection or reset:
            paths = ProjectPath()
            db_file_path = paths.root.joinpath(DATADB_FILENAME)

            if use_file:
                connection = connect(db_file_path)
                self.datadb_path = self._datadb_path()
            else:
                connection = connect(":memory:")

            for path in _data_csv_paths():
                _csv_to_table(path, connection)

            self.connection = connection

    def set_cursor(self) -> None:
        """
        Returns sqlite3 cursor for database connection.

        Returns:
            Sqlite.Cursor instance.
        """
        if not self.cursor:
            if not self.connection:
                raise ValueError(
                    "No connection to database. Call `set_connection` first."
                )

            self.cursor = self.connection.cursor()

    def fetchall(self, sql: str) -> list[Any]:
        """
        Returns all rows from a SQL query as a list of tuples.

        Args:
            sql (str): SQL query string.

        Returns:
            List of tuples.
        """
        if not self.cursor:
            self.set_cursor()

        self.cursor.execute(sql)

        return self.cursor.fetchall()

    @property
    def tables(self) -> list[str]:
        """
        A list of table names in sqlite3 database.

        Returns:
            List of table names in sqlite3 database.
        """
        if self._tables:
            return self._tables

        table_names_query = "SELECT name FROM sqlite_master WHERE type='table';"
        self._tables = [table[0] for table in self.fetchall(table_names_query)]

        return self._tables

    def table_data(self, table_name: str) -> DataFrame:
        """
        Returns Pandas DataFrame with all table data.

        Args:
            table_name: Name of table in database.

        Returns:
            Pandas DataFrame with table data.
        """
        if table_name not in self.tables:
            raise ValueError(f"Table '{table_name}' not in database.")

        return read_sql(f"SELECT * FROM {table_name}", self.connection)

    def read_sql(self, sql_query: str) -> DataFrame:
        """
        Returns Pandas DataFrame with data from SQL query.

        Args:
            sql_query (str): SQL query string.

        Returns:
            Pandas DataFrame with data from SQL query.
        """
        return read_sql(sql_query, self.connection)
