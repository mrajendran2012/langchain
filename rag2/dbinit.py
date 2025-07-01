import sqlite3
import requests
import os

SQLITE_DB_PATH = "Chinook.db"
SQLITE_SQL_URL = "https://raw.githubusercontent.com/lerocha/chinook-database/master/ChinookDatabase/DataSources/Chinook_Sqlite.sql"

def initialize_sqlite_db():
    if not os.path.exists(SQLITE_DB_PATH):
        print("Downloading and initializing Chinook.db...")
        sql = requests.get(SQLITE_SQL_URL).text
        conn = sqlite3.connect(SQLITE_DB_PATH)
        try:
            conn.executescript(sql)
            print("Chinook.db created successfully.")
        finally:
            conn.close()
    else:
        print("Chinook.db already exists.")

initialize_sqlite_db()