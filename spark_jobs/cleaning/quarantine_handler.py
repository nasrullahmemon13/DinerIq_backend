"""
DineIQ Analytics - Quarantine Handler (SRS Step 5)
Handles isolating, logging, and persisting corrupt or invalid records
to processed_data/quarantine/ per documented data-quality rules.
"""
import os
import json
import time
from datetime import datetime
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
QUARANTINE_DIR = os.path.join(PROJECT_ROOT, "processed_data", "quarantine")

class QuarantineHandler:
    def __init__(self, output_dir: str = QUARANTINE_DIR):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.quarantine_registry = []

    def quarantine_records(
        self,
        df_invalid: pd.DataFrame,
        entity_name: str,
        rule_id: str,
        rule_name: str,
        reason: str
    ) -> int:
        """
        Persist invalid records to a dedicated quarantine file with audit metadata.
        """
        count = len(df_invalid)
        if count == 0:
            return 0

        quarantine_df = df_invalid.copy()
        timestamp_str = datetime.now().isoformat()
        quarantine_df["_quarantine_timestamp"] = timestamp_str
        quarantine_df["_quarantine_rule_id"] = rule_id
        quarantine_df["_quarantine_rule_name"] = rule_name
        quarantine_df["_quarantine_reason"] = reason

        file_prefix = f"{entity_name}_{rule_id.lower().replace('-', '_')}"
        csv_file = os.path.join(self.output_dir, f"{file_prefix}.csv")
        parquet_file = os.path.join(self.output_dir, f"{file_prefix}.parquet")

        quarantine_df.to_csv(csv_file, index=False, encoding="utf-8")
        quarantine_df.to_parquet(parquet_file, compression="snappy", index=False)

        entry = {
            "rule_id": rule_id,
            "rule_name": rule_name,
            "entity": entity_name,
            "quarantined_records": count,
            "csv_path": csv_file,
            "parquet_path": parquet_file,
            "reason": reason,
            "timestamp": timestamp_str
        }
        self.quarantine_registry.append(entry)
        print(f"  [QUARANTINE] {rule_id} ({rule_name}): Quarantined {count:,} records -> {os.path.basename(csv_file)}")
        return count

    def get_summary(self) -> list:
        return self.quarantine_registry

    def export_quarantine_manifest(self) -> str:
        """Export an index manifest of all quarantined batches."""
        manifest_path = os.path.join(self.output_dir, "quarantine_manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump({
                "generated_at": datetime.now().isoformat(),
                "total_quarantined_batches": len(self.quarantine_registry),
                "total_quarantined_records": sum(item["quarantined_records"] for item in self.quarantine_registry),
                "batches": self.quarantine_registry
            }, f, indent=2)
        return manifest_path
