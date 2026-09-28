"""
DineIQ Analytics — Centralized SparkSession Provider
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
    app_name: str = "DineIQ-Analytics",
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

    if os.name == "nt":
        socket_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tmp", "java")
        os.makedirs(socket_dir, exist_ok=True)
        option = "-Djdk.net.unixdomain.tmpdir=" + get_short_path(socket_dir)
        if "jdk.net.unixdomain.tmpdir" not in os.environ.get("JAVA_TOOL_OPTIONS", ""):
            os.environ["JAVA_TOOL_OPTIONS"] = (os.environ.get("JAVA_TOOL_OPTIONS", "") + " " + option).strip()

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
