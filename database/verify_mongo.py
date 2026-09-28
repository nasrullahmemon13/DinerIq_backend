"""
MongoDB Connection & Health Diagnostic Utility for DineIQ Analytics
Tests live Atlas connectivity, database statistics, and local NoSQL export integrity.
"""
import os
import sys
import json
from dotenv import load_dotenv

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from mongodb_manager import check_mongo_connection, MONGO_EXPORT_DIR


def main():
    print("=" * 70)
    print("DineIQ Analytics — MongoDB Atlas Diagnostic & Health Audit")
    print("=" * 70)

    status = check_mongo_connection()
    print(f"\n1. Atlas Cluster Status: [{status.get('status').upper()}]")
    if status.get("status") == "connected":
        print(f"   Database:        {status.get('database')}")
        print(f"   Server Version:  {status.get('server_version')}")
        print(f"   Cluster Host:    {status.get('cluster_host')}")
        print(f"   Collections:     {', '.join(status.get('existing_collections', []))}")
    elif status.get("status") == "pending_credentials":
        print(f"   Notice:          {status.get('message')}")
        print(f"   Cluster Target:  {status.get('uri_masked')}")
        print(f"   Action Required: Edit .env file and set your real password:")
        print(f"                    MONGODB_URI=mongodb+srv://nasrullahdilshad0_db_user:<YOUR_PASSWORD>@cluster0.gfo196r.mongodb.net/?appName=Cluster0")
    else:
        print(f"   Error:           {status.get('message')}")

    print("\n2. Local MongoDB Export Collections (JSON Lines):")
    if os.path.exists(MONGO_EXPORT_DIR):
        files = [f for f in os.listdir(MONGO_EXPORT_DIR) if f.endswith(".json")]
        total_size_mb = 0
        for f in sorted(files):
            fp = os.path.join(MONGO_EXPORT_DIR, f)
            sz = os.path.getsize(fp) / (1024 * 1024)
            total_size_mb += sz
            with open(fp, "r", encoding="utf-8") as fl:
                lines = sum(1 for line in fl if line.strip())
            print(f"   - {f:<28}: {lines:>8,} records ({sz:.2f} MB)")
        print(f"   Total Local NoSQL Lake Size: {total_size_mb:.2f} MB ({len(files)} collections)")
    else:
        print("   No exports found yet. Run 'python database/mongodb_manager.py' to generate.")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
