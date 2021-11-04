from itertools import chain
import logging
import sys

from pyspark.sql import functions as f
from pyspark.sql.session import SparkSession
from pyspark.sql.window import Window
from pyspark.sql.types import ArrayType, StringType

spark = SparkSession.builder.getOrCreate()
spark.conf.set("spark.sql.execution.arrow.enabled", "true")

logger = logging.getLogger(__name__)
log_handler = logging.StreamHandler(sys.stdout)
log_handler.setFormatter(logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
log_handler.setLevel(logging.DEBUG)
logger.addHandler(log_handler)
logger.setLevel(logging.DEBUG)


input_path = '/data/raw/'
output_path = '/data/data_science/powerBI/'


def write_output(df):
    logger.info("CREATING PCP HCC DROPPED DATASET")
    logger.info("WRITING: {}".format(output_path + "PCP_HCC_dropped.parquet"))
    df.write.mode('overwrite').parquet(output_path + 'PCP_HCC_dropped.parquet')
    df.coalesce(1).write.mode('overwrite').option("header", "true").option("delimiter", '|').csv('wasbs://rdp-uploads@coretechsnmdev.blob.core.windows.net/PCP_HCC_dropped.csv')
    return df
		

def main():
    
    NWhcc = spark.read.parquet('/data/data_science/raf/NW_diag_HCC_raf_new_V22_V24.parquet')
    df = spark.read.parquet('/data/data_science/powerBI/Final_PCP_attribution.parquet')
    df = df.withColumnRenamed('year', 'claim_year')

    hcc_join = df.join(NWhcc.select('BENE_MBI_ID', 'claim_year', 'cum_hcc_diff'), on=['BENE_MBI_ID','claim_year'], how='left')
    hcc_join = hcc_join.withColumn('cum_hcc_diff', f.col('cum_hcc_diff').cast('string'))
    hcc_join = hcc_join.withColumn('cum_hcc_diff', f.regexp_replace('cum_hcc_diff', '\\[', '')).withColumn('cum_hcc_diff', f.regexp_replace('cum_hcc_diff', '\\]', '')).withColumn('cum_hcc_diff', f.regexp_replace('cum_hcc_diff', "'", '')).withColumn("HCC_ARRAY", f.split(f.col("cum_hcc_diff"), ",\s*").cast(ArrayType(StringType())).alias("HCC_ARRAY"))
    hcc_join = hcc_join.select('BENE_MBI_ID','claim_year','FINAL_PCP_NPI',f.explode('HCC_ARRAY').alias('dropped_HCC'))

    hcc_join = hcc_join.select([f.when(f.col(c)=="",None).otherwise(f.col(c)).alias(c) for c in hcc_join.columns])  
    hcc_join = hcc_join.dropna()
    hcc_join = hcc_join.drop_duplicates()    
    d_list = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10']
    hcc_join = hcc_join.filter(~hcc_join.dropped_HCC.isin(d_list))
    hcc_join = hcc_join.orderBy('BENE_MBI_ID', 'claim_year')
    write_output(hcc_join)


	
if __name__ == "__main__":
	
    logger.info('START')
    main()
    logger.info('END') 
