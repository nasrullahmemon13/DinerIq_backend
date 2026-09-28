"""
DineIQ Analytics - Explicit Schema CSV Ingestion Engine (SRS Step 3)
Demonstrates Capabilities:
- Explicit Schema Definition (using schema_definitions.py)
- Large-File Loading (1,001,500 order items & 100,200 orders)
- Partition Handling (getNumPartitions, glom, repartition, coalesce, partitionBy write)
"""
import os
import sys
import time

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "raw_data")
PARQUET_DATA_DIR = os.path.join(PROJECT_ROOT, "parquet_data")

sys.path.append(CURRENT_DIR)
import schema_definitions as sd
from spark_compat import get_spark_session

def create_spark_session(app_name: str = "DineIQ-ExplicitSchemaIngestion"):
    """Initialize a tuned SparkSession for local high-throughput processing."""
    return get_spark_session(app_name)

def load_with_explicit_schema(spark, table_name: str, file_path: str, schema):
    """Load a CSV file with an explicit PySpark StructType schema."""
    print(f"\n[Loading] Table: {table_name} from {os.path.basename(file_path)}")
    start_time = time.perf_counter()

    df = spark.read \
        .format("csv") \
        .option("header", "true") \
        .option("mode", "PERMISSIVE") \
        .option("columnNameOfCorruptRecord", "_corrupt_record") \
        .option("dateFormat", "yyyy-MM-dd") \
        .option("timestampFormat", "yyyy-MM-dd HH:mm:ss") \
        .schema(schema) \
        .load(file_path)

    row_count = df.count()
    elapsed = time.perf_counter() - start_time
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    throughput_rows_sec = row_count / elapsed if elapsed > 0 else 0
    throughput_mb_sec = file_size_mb / elapsed if elapsed > 0 else 0

    print(f"  Rows loaded: {row_count:,}")
    print(f"  File size: {file_size_mb:.2f} MB")
    print(f"  Ingestion time: {elapsed:.3f} seconds")
    print(f"  Throughput: {throughput_rows_sec:,.0f} rows/sec ({throughput_mb_sec:.2f} MB/sec)")
    return df, elapsed

def demonstrate_partition_handling(df, table_name: str):
    """
    Demonstrate Spark Partition Handling:
    1. Check default partition count
    2. Inspect distribution across partitions (glom)
    3. Repartition by column (e.g. location_id)
    4. Coalesce partitions to minimize file fragmentation
    5. Save partitioned dataset by location
    """
    print(f"\n--- Demonstrating Partition Handling for {table_name} ---")

    default_partitions = df.rdd.getNumPartitions()
    print(f"1. Default Spark Partitions: {default_partitions}")

    try:
        partition_sizes = df.rdd.glom().map(len).collect()
        print(f"   Record distribution across partitions: {partition_sizes}")
    except Exception as e:
        print(f"   Partition inspection: {e}")

    if "location_id" in df.columns:
        print("\n2. Repartitioning by 'location_id' into 8 balanced shuffle partitions...")
        repartitioned_df = df.repartition(8, "location_id")
        print(f"   New Partition Count: {repartitioned_df.rdd.getNumPartitions()}")

        print("\n3. Coalescing partitions to 2 for downstream archiving...")
        coalesced_df = repartitioned_df.coalesce(2)
        print(f"   Coalesced Partition Count: {coalesced_df.rdd.getNumPartitions()}")

        output_dir = os.path.join(PARQUET_DATA_DIR, "partitioned", f"{table_name}_by_location")
        print(f"\n4. Writing partitioned Parquet dataset to: {output_dir}")
        t0 = time.perf_counter()
        df.write.mode("overwrite").partitionBy("location_id").parquet(output_dir)
        print(f"   [OK] Partitioned write completed in {time.perf_counter() - t0:.2f}s")

        if os.path.exists(output_dir):
            partition_dirs = [d for d in os.listdir(output_dir) if d.startswith("location_id=")]
            print(f"   Created {len(partition_dirs)} physical partition folders (e.g., {partition_dirs[:3]}...)")

def main():
    print("=" * 75)
    print("DineIQ Analytics - SRS Step 3: Explicit Schema CSV Ingestion Engine")
    print("=" * 75)

    spark = create_spark_session()

    try:
        orders_csv = os.path.join(RAW_DATA_DIR, "orders", "orders.csv")
        orders_schema = sd.get_orders_schema()
        df_orders, _ = load_with_explicit_schema(spark, "orders", orders_csv, orders_schema)

        print("\n--- Orders Schema Confirmation ---")
        df_orders.printSchema()
        print("\n--- Orders Sample Records ---")
        df_orders.select("order_id", "customer_id", "location_id", "order_date", "total_amount", "order_status").show(5, truncate=False)

        items_csv = os.path.join(RAW_DATA_DIR, "order_items", "order_items.csv")
        items_schema = sd.get_order_items_schema()
        df_items, _ = load_with_explicit_schema(spark, "order_items", items_csv, items_schema)

        print("\n--- Order Items Schema Confirmation ---")
        df_items.printSchema()
        print("\n--- Order Items Sample Records ---")
        df_items.show(5, truncate=False)

        cust_csv = os.path.join(RAW_DATA_DIR, "customers", "customers.csv")
        df_cust, _ = load_with_explicit_schema(spark, "customers", cust_csv, sd.get_customers_schema())

        rest_csv = os.path.join(RAW_DATA_DIR, "restaurants", "restaurants.csv")
        df_rest, _ = load_with_explicit_schema(spark, "restaurants", rest_csv, sd.get_restaurants_schema())

        demonstrate_partition_handling(df_orders, "orders")

        print("\n" + "=" * 75)
        print("Explicit Schema Ingestion & Partition Handling Demonstrated Successfully!")
        print("=" * 75)

    finally:
        spark.stop()

if __name__ == "__main__":
    main()
