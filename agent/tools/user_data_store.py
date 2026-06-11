from __future__ import annotations

import csv
import sqlite3
from pathlib import Path
from typing import Any

from utils.config_handler import agent_conf
from utils.logger_handler import logger
from utils.path_tool import get_abs_path


class UserUsageStore:
    def __init__(
        self,
        db_path: str | None = None,
        seed_csv_path: str | None = None,
    ) -> None:
        self.db_path = Path(get_abs_path(db_path or agent_conf["external_data_db_path"]))
        self.seed_csv_path = Path(get_abs_path(seed_csv_path or agent_conf["external_data_path"]))

    def get_record(self, user_id: str, month: str) -> dict[str, Any] | None:
        self.ensure_initialized()
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT
                    user_id,
                    month,
                    feature_profile,
                    cleaning_efficiency,
                    consumables_status,
                    comparison_summary
                FROM user_usage_records
                WHERE user_id = ? AND month = ?
                """,
                (user_id, month),
            ).fetchone()

        if row is None:
            return None
        return {key: self._clean(value) for key, value in dict(row).items()}

    def get_schema_summary(self) -> dict[str, Any]:
        self.ensure_initialized()
        with self._connect() as conn:
            record_count = conn.execute("SELECT COUNT(*) FROM user_usage_records").fetchone()[0]
            user_count = conn.execute(
                "SELECT COUNT(DISTINCT user_id) FROM user_usage_records"
            ).fetchone()[0]
            month_rows = conn.execute(
                """
                SELECT MIN(month) AS min_month, MAX(month) AS max_month
                FROM user_usage_records
                """
            ).fetchone()

        return {
            "database": str(self.db_path),
            "table": "user_usage_records",
            "record_count": record_count,
            "user_count": user_count,
            "month_range": {
                "min": month_rows["min_month"],
                "max": month_rows["max_month"],
            },
            "fields": {
                "user_id": "脱敏用户 ID，数字字符串",
                "month": "记录月份，格式 YYYY-MM",
                "feature_profile": "家庭面积、地面材质、宠物等设备使用特征",
                "cleaning_efficiency": "覆盖率、日均清扫面积、漏扫或避障表现",
                "consumables_status": "主刷、边刷、滤网、尘盒等耗材状态",
                "comparison_summary": "同类用户或同户型用户的使用对比",
            },
        }

    def ensure_initialized(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            self._create_table(conn)
            count = conn.execute("SELECT COUNT(*) FROM user_usage_records").fetchone()[0]
            if count == 0:
                inserted = self._seed_from_csv(conn)
                logger.info(f"[user data store] seeded {inserted} records into {self.db_path}")

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def _create_table(conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS user_usage_records (
                user_id TEXT NOT NULL,
                month TEXT NOT NULL,
                feature_profile TEXT NOT NULL,
                cleaning_efficiency TEXT NOT NULL,
                consumables_status TEXT NOT NULL,
                comparison_summary TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (user_id, month)
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_user_usage_records_month
            ON user_usage_records(month)
            """
        )

    def _seed_from_csv(self, conn: sqlite3.Connection) -> int:
        if not self.seed_csv_path.exists():
            logger.warning(f"[user data store] seed csv not found: {self.seed_csv_path}")
            return 0

        inserted = 0
        with self.seed_csv_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO user_usage_records (
                        user_id,
                        month,
                        feature_profile,
                        cleaning_efficiency,
                        consumables_status,
                        comparison_summary
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        self._clean(row.get("用户ID")),
                        self._clean(row.get("时间")),
                        self._clean(row.get("特征")),
                        self._clean(row.get("清洁效率")),
                        self._clean(row.get("耗材")),
                        self._clean(row.get("对比")),
                    ),
                )
                inserted += 1
        return inserted

    @staticmethod
    def _clean(value: Any) -> str:
        return str(value or "").replace("\\n", "\n").strip()
