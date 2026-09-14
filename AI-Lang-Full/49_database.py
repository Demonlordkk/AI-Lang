"""Phase 49: portable SQL baseline via SQLite."""
import sqlite3
class Database:
    def __init__(self, path=":memory:"): self.conn=sqlite3.connect(path)
    def execute(self, sql, params=()): return self.conn.execute(sql, params)
    def commit(self): self.conn.commit()
    def close(self): self.conn.close()
