SPARK_APP_NAME = "DineIQ-BigData-Analytics"
SPARK_MASTER = "local[*]"
SPARK_DRIVER_MEMORY = "4g"
SPARK_EXECUTOR_MEMORY = "4g"
SPARK_SQL_SHUFFLE_PARTITIONS = 8
SPARK_DEFAULT_PARALLELISM = 8

SPARK_CONF = {
    "spark.app.name": SPARK_APP_NAME,
    "spark.master": SPARK_MASTER,
    "spark.driver.memory": SPARK_DRIVER_MEMORY,
    "spark.executor.memory": SPARK_EXECUTOR_MEMORY,
    "spark.sql.shuffle.partitions": str(SPARK_SQL_SHUFFLE_PARTITIONS),
    "spark.default.parallelism": str(SPARK_DEFAULT_PARALLELISM),
    "spark.sql.execution.arrow.pyspark.enabled": "true",
    "spark.sql.parquet.compression.codec": "snappy"
}
