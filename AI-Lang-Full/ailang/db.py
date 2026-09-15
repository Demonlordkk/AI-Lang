"""Persistent storage: SQL over SQLite, and a document store.

Files and JSON are fine until data outgrows memory or two writers collide.
These builtins give real durable storage with transactions, without adding a
dependency -- SQLite ships with the host.

    let d := db_open("app.db").
    db_exec(d, "create table if not exists users(id integer, name text)").
    db_exec(d, "insert into users values(?, ?)", [1, "ann"]).
    emit db_query(d, "select * from users where id = ?", [1]).

Rows come back as Maps keyed by column name, so a query result flows straight
into `map`, `filter` and `group_by` like any other list.

Parameters are always bound, never interpolated, so a value carrying a quote
cannot alter the statement.
"""
from __future__ import annotations

import json
import sqlite3
import threading

from .errors import VMError
from .values import MapKey, RecordValue, unmap_key

_id_lock = threading.Lock()
_next = [1]


class _DB:
    __slots__ = ("conn", "path", "id", "depth", "lock", "closed")

    def __init__(self, conn, path):
        self.conn = conn
        self.path = path
        # Each connection has its own re-entrant lock.  A process-global lock
        # unnecessarily serialized independent databases, while releasing the
        # lock during a transaction let another thread interleave statements
        # into the transaction's uncommitted state.
        self.lock = threading.RLock()
        # nesting level of explicit db_transaction blocks; while it is
        # non-zero individual statements must not commit on their own or a
        # later rollback would have nothing left to undo
        self.depth = 0
        self.closed = False
        with _id_lock:
            self.id = _next[0]
            _next[0] += 1


def _check(db, where):
    if not isinstance(db, _DB):
        raise VMError(f"{where}: first argument must be a database from db_open")
    if db.closed:
        raise VMError(f"{where}: database is closed")
    return db


def _jsonable(v):
    """Convert runtime containers to values accepted by ``json.dumps``."""
    if isinstance(v, MapKey):
        return _jsonable(unmap_key(v))
    if isinstance(v, RecordValue):
        return {str(k): _jsonable(x) for k, x in v.data.items()}
    if isinstance(v, list):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(unmap_key(k)): _jsonable(x) for k, x in v.items()}
    if v is None or isinstance(v, (int, float, str, bool)):
        return v
    raise TypeError(f"{type(v).__name__} is not JSON serializable")


def _to_sql(v, where):
    """AI-Lang values that SQLite can store directly; others are JSON."""
    if v is None or isinstance(v, (int, float, str, bool)):
        return v
    if isinstance(v, (list, dict, RecordValue)):
        try:
            return json.dumps(_jsonable(v), ensure_ascii=False)
        except (TypeError, ValueError) as e:
            raise VMError(f"{where}: cannot store value: {e}") from None
    raise VMError(f"{where}: cannot store a value of type {type(v).__name__}")


def db_open(path=":memory:"):
    """Open (or create) a database. ':memory:' is a temporary one."""
    if not isinstance(path, str):
        raise VMError("db_open: path must be Text")
    try:
        conn = sqlite3.connect(path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("pragma journal_mode=WAL")
        conn.execute("pragma foreign_keys=ON")
    except sqlite3.Error as e:
        raise VMError(f"db_open: cannot open '{path}': {e}") from e
    return _DB(conn, str(path))


def _params(args, where):
    if args is None:
        return []
    if isinstance(args, dict):
        return {str(k): _to_sql(v, where) for k, v in args.items()}
    if not isinstance(args, list):
        raise VMError(f"{where}: parameters must be a List or Map")
    return [_to_sql(v, where) for v in args]


def db_exec(db, sql, args=None):
    """Run a statement that does not return rows; gives the rows affected."""
    db = _check(db, "db_exec")
    if not isinstance(sql, str):
        raise VMError("db_exec: sql must be Text")
    try:
        with db.lock:
            cur = db.conn.execute(sql, _params(args, "db_exec"))
            if db.depth == 0:
                db.conn.commit()
            return cur.rowcount if cur.rowcount >= 0 else 0
    except sqlite3.Error as e:
        raise VMError(f"db_exec: {e}\n  statement: {sql}") from e


def db_query(db, sql, args=None):
    """Run a query; gives a List of Maps keyed by column name."""
    db = _check(db, "db_query")
    if not isinstance(sql, str):
        raise VMError("db_query: sql must be Text")
    try:
        with db.lock:
            cur = db.conn.execute(sql, _params(args, "db_query"))
            return [dict(row) for row in cur.fetchall()]
    except sqlite3.Error as e:
        raise VMError(f"db_query: {e}\n  statement: {sql}") from e


def db_one(db, sql, args=None):
    """Run a query expected to match a single row; gives nothing if none."""
    rows = db_query(db, sql, args)
    if not rows:
        return None
    return rows[0]


def db_many(db, sql, rows):
    """Run one statement over many parameter sets, in a single transaction."""
    db = _check(db, "db_many")
    if not isinstance(sql, str):
        raise VMError("db_many: sql must be Text")
    if not isinstance(rows, list):
        raise VMError("db_many: third argument must be a List of parameter Lists")
    try:
        with db.lock:
            db.conn.executemany(sql, [_params(r, "db_many") for r in rows])
            if db.depth == 0:
                db.conn.commit()
    except sqlite3.Error as e:
        raise VMError(f"db_many: {e}\n  statement: {sql}") from e
    return len(rows)


def db_transaction(db, fn):
    """Run `fn` inside a transaction, rolling back if it raises.

    The connection lock is held for the whole callback.  It is re-entrant for
    statements made by that callback, but prevents another thread from
    interleaving work on the same SQLite connection or observing its temporary
    transaction depth.
    """
    db = _check(db, "db_transaction")
    if not callable(fn):
        raise VMError("db_transaction: second argument must be a function")
    with db.lock:
        outer = db.depth == 0
        entered = False
        try:
            if outer:
                db.conn.execute("begin")
            db.depth += 1
            entered = True
            result = fn()
        except BaseException as e:
            if entered:
                db.depth -= 1
            if outer and entered:
                try:
                    db.conn.rollback()
                except sqlite3.Error:
                    pass
            if isinstance(e, sqlite3.Error):
                raise VMError(f"db_transaction: {e}") from e
            raise
        else:
            db.depth -= 1
            if outer:
                try:
                    db.conn.commit()
                except sqlite3.Error as e:
                    try:
                        db.conn.rollback()
                    except sqlite3.Error:
                        pass
                    raise VMError(f"db_transaction: commit failed: {e}") from e
            return result


def db_tables(db):
    """List the tables in the database."""
    _check(db, "db_tables")
    rows = db_query(
        db, "select name from sqlite_master where type='table' order by name"
    )
    return [r["name"] for r in rows]


def db_close(db):
    """Close the database."""
    db = _check(db, "db_close")
    with db.lock:
        try:
            db.conn.close()
        except sqlite3.Error:
            pass
        finally:
            db.closed = True
    return None


# ------------------------------------------------------------ document store
def store_open(path):
    """A durable key/value store for documents, backed by the same engine."""
    db = db_open(path)
    db_exec(db, "create table if not exists documents(key text primary key, value text)")
    return db


def store_put(db, key, value):
    """Save a document under a key."""
    _check(db, "store_put")
    db_exec(
        db,
        "insert into documents(key, value) values(?, ?) "
        "on conflict(key) do update set value=excluded.value",
        [str(key), json.dumps(_jsonable(value), ensure_ascii=False)],
    )
    return None


def store_get(db, key, default=None):
    """Read a document; gives `default` when the key is absent."""
    _check(db, "store_get")
    row = db_one(db, "select value from documents where key = ?", [str(key)])
    if row is None:
        return default
    return json.loads(row["value"])


def store_delete(db, key):
    """Remove a document; reports whether it existed."""
    _check(db, "store_delete")
    return db_exec(db, "delete from documents where key = ?", [str(key)]) > 0


def store_keys(db, prefix=""):
    """List keys, optionally restricted to a prefix."""
    _check(db, "store_keys")
    if prefix:
        # a prefix is literal text, not a pattern: neutralise the LIKE
        # wildcards so "a_b" or "50%" cannot widen the match
        escaped = (
            str(prefix)
            .replace("\\", "\\\\")
            .replace("%", "\\%")
            .replace("_", "\\_")
        )
        rows = db_query(
            db,
            "select key from documents where key like ? escape '\\' order by key",
            [escaped + "%"],
        )
    else:
        rows = db_query(db, "select key from documents order by key")
    return [r["key"] for r in rows]
