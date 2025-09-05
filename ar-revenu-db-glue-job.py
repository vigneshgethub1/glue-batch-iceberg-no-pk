import sys
import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import concat_ws, md5
from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.utils import getResolvedOptions

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)  

class IcebergJob:
    def __init__(self, spark, glue_context, args):  
        self.spark = spark
        self.glue_context = glue_context
        self.args = args
        self.database_name = args['database_name'].strip()
        self.table_names = [t.strip() for t in args['iceberg_table_name'].split(',')]
        self.s3_paths = [p.strip() for p in args['s3_input_path'].split(',')]
        self.warehouse_path = args['warehouse_name'].strip()
        self.partition_column = args['partition_column'].strip() if 'partition_column' in args else None
        self.surrogate_key_columns = []
        if 'surrogate_key_columns' in args:
            surrogate_arg = args['surrogate_key_columns'].strip()
            self.surrogate_key_columns = [cols.split('|') for cols in surrogate_arg.split(',')]
            if len(self.surrogate_key_columns) != len(self.table_names):
                raise ValueError("Number of surrogate_key_columns entries must match number of tables")
        if len(self.table_names) != len(self.s3_paths):
            raise ValueError("Number of tables and S3 paths must be equal")

    def run(self):
        logger.info(f"Starting ETL job for database '{self.database_name}' with warehouse '{self.warehouse_path}'")
        logger.info(f"Processing tables: {self.table_names}")
        logger.info(f"Corresponding S3 input paths: {self.s3_paths}")
        self._create_database_if_not_exists(self.database_name)
        success_count = 0
        failure_count = 0
        for idx, (table_name, s3_path, warehouse_path) in enumerate(zip(self.table_names, self.s3_paths, self.warehouse_path)):
            try:
                logger.info(f"Processing table '{table_name}' with data from '{s3_path}'")
                df = self._read_data(s3_path)
                self._validate_partition_column(df)
                surrogate_cols = None
                if self.surrogate_key_columns:
                    surrogate_cols = self.surrogate_key_columns[idx]
                    df = self._add_surrogate_key(df, surrogate_cols)
                    logger.info(f"Added surrogate key column using columns: {surrogate_cols}")
                    df = df.dropDuplicates(["surrogate_key"])
                table_exists = self._check_table_exists(self.database_name, table_name)
                if table_exists:
                    logger.info(f"Table {self.database_name}.{table_name} exists, merging data to avoid duplicates")
                    if surrogate_cols:
                        self._merge_data(df, table_name)
                    else:
                        logger.info("Surrogate key columns not specified, falling back to append")
                        self._append_data(df, table_name)
                else:
                    logger.info(f"Table {self.database_name}.{table_name} does not exist, creating table")
                    self._create_table(df, table_name, warehouse_path)
                self._run_optimizations(table_name)
                success_count += 1
                logger.info(f"Completed processing for table '{table_name}'")
            except Exception as ex:
                logger.error(f"Error processing table '{table_name}': {ex}", exc_info=True)
                failure_count += 1
                continue
        logger.info(f"Job finished: {success_count} succeeded, {failure_count} failed")

    def _create_database_if_not_exists(self, database_name):
        query = f"CREATE DATABASE IF NOT EXISTS {database_name}"
        self.spark.sql(query)
        logger.info(f"Ensured database exists: {database_name}")

    def _read_data(self, s3_path):
        df = self.spark.read.format("parquet").load(s3_path)
        row_count = df.count()
        logger.info(f"Read {row_count} rows from {s3_path}")
        df.printSchema()
        print("DataFrame columns:", df.columns)
        if row_count == 0:
            raise ValueError(f"No data found at path: {s3_path}")
        return df

    def _validate_partition_column(self, df):
        if self.partition_column:
            if self.partition_column not in df.columns:
                raise ValueError(f"Partition column '{self.partition_column}' not found in input data")
            else:
                logger.info(f"Partition column '{self.partition_column}' validated")

    def _check_table_exists(self, database_name, table_name):
        full_table_name = f"glue_catalog.{database_name}.{table_name}"
        
        try:
            # Checking if table is listed in the catalog
            logger.debug(f"Checking existence of table: {full_table_name}")
            tables_df = self.spark.sql(f"SHOW TABLES IN glue_catalog.{database_name}")
            existing_tables = [row['tableName'] for row in tables_df.collect()]
            table_listed = table_name in existing_tables
            
            if not table_listed:
                logger.info(f"Table {table_name} does not exist in catalog")
                return False
                
            # Table is listed, now validate it's accessible and not corrupted
            logger.debug(f"Table {table_name} found in catalog, validating accessibility...")
            
            try:
                # Try to access table metadata - this will fail for corrupted tables
                self.spark.sql(f"DESCRIBE TABLE {full_table_name}").collect()
                logger.info(f"Table {table_name} exists and is accessible")
                return True
                
            except Exception as access_error:
                error_msg = str(access_error).lower()
                
                # Check for known corruption patterns
                corruption_indicators = [
                    "inputformat cannot be null",
                    "storagedescriptor",
                    "hiveexception",
                    "unable to fetch table"
                ]
                
                if any(indicator in error_msg for indicator in corruption_indicators):
                    logger.warning(f"Table {table_name} exists but has corrupted metadata: {access_error}")
                    
                    # attempt to clean up corrupted tables
                    try:
                        logger.info(f"Attempting to drop corrupted table {table_name}")
                        self.spark.sql(f"DROP TABLE IF EXISTS {full_table_name}")
                        logger.info(f"Successfully dropped corrupted table {table_name}")
                        return False
                        
                    except Exception as drop_error:
                        logger.error(f"Failed to drop corrupted table {table_name}: {drop_error}")
                        raise RuntimeError(f"Cannot proceed: table {table_name} is corrupted and cannot be dropped")
                else:
                    #log and re-raise for investigation
                    logger.error(f"Unexpected error accessing table {table_name}: {access_error}")
                    raise
                    
        except Exception as e:
            # Catch-all for any other issues
            logger.error(f"Critical error checking table existence for {table_name}: {e}")
            raise RuntimeError(f"Failed to check table existence: {e}")

    def _ensure_table_ready(self, database_name, table_name):
        """
        Additional production helper method to ensure table is ready for operations.
        Call this before any table operations.
        """
        full_table_name = f"glue_catalog.{database_name}.{table_name}"
        
        try:
            # Validate table structure and properties
            describe_result = self.spark.sql(f"DESCRIBE EXTENDED {full_table_name}").collect()
            
            # Check if it's actually an Iceberg table
            table_properties = {}
            for row in describe_result:
                if row['col_name'] and 'table.type' in str(row):
                    if 'iceberg' not in str(row).lower():
                        logger.warning(f"Table {table_name} exists but is not an Iceberg table")
                        return False
            
            logger.info(f"Table {table_name} is ready for operations")
            return True
            
        except Exception as e:
            logger.error(f"Table readiness check failed for {table_name}: {e}")
            return False

    def _create_table(self, df, table_name, warehouse_path):
        # Debug: show catalogs to confirm catalog config
        print("DEBUG CATALOGS:", self.spark.sql("SHOW CATALOGS").toPandas())
        self.spark.sql("SHOW CATALOGS").show()
        writer = df.writeTo(f"glue_catalog.{self.database_name}.{table_name}").using("iceberg") \
                   .tableProperty("location", f"{self.warehouse_path}/{table_name}") \
                   .tableProperty("format-version", "2")
        if self.partition_column:
            writer = writer.partitionedBy(self.partition_column)
        writer.create()
        logger.info(f"Iceberg table {self.database_name}.{table_name} created successfully")

    def _append_data(self, df, table_name):
        df.writeTo(f"glue_catalog.{self.database_name}.{table_name}").append()
        logger.info(f"Data appended to Iceberg table {self.database_name}.{table_name}")

    def _add_surrogate_key(self, df, cols):
        return df.withColumn(
            "surrogate_key",
            md5(concat_ws("||", *cols))
        )

    def _merge_data(self, df, table_name):
        temp_view_name = f"updates_{table_name}"
        df.createOrReplaceTempView(temp_view_name)
        full_table_name = f"glue_catalog.{self.database_name}.{table_name}"
        merge_sql = f"""
            MERGE INTO {full_table_name} T
            USING {temp_view_name} S
            ON T.surrogate_key = S.surrogate_key
            WHEN MATCHED THEN UPDATE SET *
            WHEN NOT MATCHED THEN INSERT *
        """
        logger.info(f"Running MERGE INTO for table {full_table_name}")
        self.spark.sql(merge_sql)
        logger.info(f"MERGE INTO completed for table {full_table_name}")

    def _run_optimizations(self, table_name):
        full_table_name = f"glue_catalog.{self.database_name}.{table_name}"
        try:
            logger.info(f"Running data file compaction for {full_table_name}")
            self.spark.sql(f"CALL system.rewrite_data_files(table => '{full_table_name}')")
            logger.info(f"Compaction succeeded for {full_table_name}")
            logger.info(f"Running snapshot expiration for {full_table_name}")
            self.spark.sql(f"CALL system.expire_snapshots(table => '{full_table_name}')")
            logger.info(f"Snapshot expiration succeeded for {full_table_name}")
        except Exception as ex:
            logger.warning(f"Iceberg optimization call failed for {full_table_name}, continuing job. Error: {ex}")

def main():
    try:
        args = getResolvedOptions(
            sys.argv,
            [
                'JOB_NAME',
                'database_name',
                'iceberg_table_name',
                's3_input_path',
                'warehouse_name',
                'partition_column',
                'surrogate_key_columns'
            ]
        )
       
        spark = (
            SparkSession.builder
            .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions")
            .config("spark.sql.catalog.glue_catalog", "org.apache.iceberg.spark.SparkCatalog")
            .config("spark.sql.catalog.glue_catalog.catalog-impl", "org.apache.iceberg.aws.glue.GlueCatalog")
            .config("spark.sql.catalog.glue_catalog.warehouse", args['warehouse_name'])
            .config("spark.sql.catalog.glue_catalog.io-impl", "org.apache.iceberg.aws.s3.S3FileIO")
            .config("spark.sql.defaultCatalog", "glue_catalog")
            .config("spark.sql.warehouse.dir", args['warehouse_name'])
            .getOrCreate()
        )
        
        print("warehouse.dir:", spark.conf.get("spark.sql.warehouse.dir"))
        print("default.catalog:", spark.conf.get("spark.sql.defaultCatalog"))
        print("glue warehouse:", spark.conf.get("spark.sql.catalog.glue_catalog.warehouse"))
        
        glue_context = GlueContext(spark.sparkContext)
        job = Job(glue_context)
        job.init(args['JOB_NAME'], args)
        
        etl_job = IcebergJob(spark, glue_context, args)
        etl_job.run()
        
        job.commit()
        logger.info(f"Glue job {args['JOB_NAME']} completed successfully")
        
    except Exception as e:
        logger.critical(f"Glue job failed: {e}", exc_info=True)
        raise

if __name__ == '__main__':
    main()