"""SQLite persistence for prediction history."""
from pathlib import Path
import json, sqlite3
from datetime import datetime, timezone

DB_PATH = Path(__file__).resolve().parents[1] / "database" / "maintenance.db"

def init_db(path: Path = DB_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as con:
        con.execute("""CREATE TABLE IF NOT EXISTS maintenance_history (id INTEGER PRIMARY KEY AUTOINCREMENT, vehicle_id TEXT NOT NULL, timestamp TEXT NOT NULL, parameters TEXT NOT NULL, risk_probability REAL NOT NULL, maintenance_prediction INTEGER NOT NULL, diagnosis TEXT NOT NULL, recommendation TEXT NOT NULL, priority TEXT NOT NULL)""")

def save_prediction(vehicle_id: str, parameters: dict, risk: float, prediction: int, diagnosis: list[str], recommendation: dict, path: Path = DB_PATH) -> None:
    init_db(path)
    with sqlite3.connect(path) as con:
        con.execute("INSERT INTO maintenance_history(vehicle_id,timestamp,parameters,risk_probability,maintenance_prediction,diagnosis,recommendation,priority) VALUES (?,?,?,?,?,?,?,?)", (vehicle_id, datetime.now(timezone.utc).isoformat(), json.dumps(parameters), risk, prediction, json.dumps(diagnosis), recommendation["recommendation"], recommendation["priority"]))

def get_history(vehicle_id: str, path: Path = DB_PATH) -> list[dict]:
    init_db(path)
    with sqlite3.connect(path) as con:
        con.row_factory = sqlite3.Row
        return [dict(row) for row in con.execute("SELECT * FROM maintenance_history WHERE vehicle_id=? ORDER BY timestamp DESC", (vehicle_id,)).fetchall()]

def count_summary(path: Path = DB_PATH) -> tuple[int, int, int, float]:
    init_db(path)
    with sqlite3.connect(path) as con:
        row = con.execute("SELECT COUNT(*), COALESCE(SUM(maintenance_prediction),0), COALESCE(SUM(priority='HIGH'),0), COALESCE(AVG(risk_probability),0) FROM maintenance_history").fetchone()
    return tuple(row)
