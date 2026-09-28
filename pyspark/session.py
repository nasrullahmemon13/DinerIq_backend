"""
DineIQ Analytics — Centralized PySpark Session Provider
Configures and initializes Apache Spark with:
- Windows 8.3 short-path resolution for python worker sockets
- Adaptive Query Execution (AQE) enabled
- Configurable driver/executor memory and shuffle partitions
- Production logging levels
"""
import os
import sys
import ctypes
from typing import Optional


def get_short_path(path: str) -> str:
    """Returns 8.3 short path on Windows to avoid whitespace issues in worker process execution."""
    if os.name == "nt" and os.path.exists(path):
        buf = ctypes.create_unicode_buffer(500)
        ret = ctypes.windll.kernel32.GetShortPathNameW(path, buf, 500)
        if ret > 0 and os.path.exists(buf.value):
            return buf.value
    return path


def get_spark_session(
    app_name: str = "DineIQ-PySpark-Engine",
    master: Optional[str] = None,
    driver_memory: Optional[str] = None,
    shuffle_partitions: Optional[int] = None,
    log_level: str = "WARN"
):
    """
    Creates or retrieves the active SparkSession with optimized enterprise settings.
    """
    safe_python = get_short_path(sys.executable)
    os.environ["PYSPARK_PYTHON"] = safe_python
    os.environ["PYSPARK_DRIVER_PYTHON"] = safe_python

    master_url = master or os.environ.get("SPARK_MASTER", "local[*]")
    mem = driver_memory or os.environ.get("SPARK_DRIVER_MEMORY", "4g")
    partitions = str(shuffle_partitions or os.environ.get("SPARK_SHUFFLE_PARTITIONS", "8"))

    from pyspark.sql import SparkSession

    builder = SparkSession.builder \
        .appName(app_name) \
        .master(master_url) \
        .config("spark.driver.memory", mem) \
        .config("spark.sql.shuffle.partitions", partitions) \
        .config("spark.default.parallelism", partitions) \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .config("spark.python.worker.reuse", "true")

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel(log_level)
    return spark


if __name__ == "__main__":
    print("Testing PySpark Session Initialization...")
    sp = get_spark_session("PySpark-Session-SmokeTest")
    print(f"Active Spark Version: {sp.version}")
    print(f"Master: {sp.sparkContext.master}")
    print(f"AQE Enabled: {sp.conf.get('spark.sql.adaptive.enabled')}")
    df = sp.createDataFrame([(1, "DineIQ"), (2, "PySpark")], ["id", "platform"])
    print(f"Smoke Test Count: {df.count()}")
    sp.stop()
    print("Session stopped cleanly.")
