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
    logger.info("CREATING ALR HCC LIST DATASET")
    logger.info("WRITING: {}".format(output_path + "ALR_HCC_list.parquet"))
    df.write.mode('overwrite').parquet(output_path + 'ALR_HCC_list.parquet')
    df = df.withColumn('HCC_ALR', f.col('HCC_ALR').cast('string'))
    df.coalesce(1).write.mode('overwrite').option("header", "true").option("delimiter", '|').csv('wasbs://rdp-uploads@coretechsnmdev.blob.core.windows.net/ALR_HCC_list.csv')
    return df
		

def main():
    
    df = spark.read.parquet('/data/data_science/raf/NW_diag_HCC_raf_new_V22_V24.parquet')
    df = df.select('BENE_MBI_ID', 'claim_year', 'hcc_map', 'hcc_map_diff', 'hcc_map_diags', 'hcc_map_diff_diags') 
    pcp18 = spark.read.parquet('/data/attribution/attribution_2018.parquet')   
    pcp19 = spark.read.parquet('/data/attribution/attribution_2019.parquet')
    pcp20 = spark.read.parquet('/data/attribution/attribution_2020.parquet')
    pcp21 = spark.read.parquet('/data/attribution/attribution_2021.parquet')
    
    pcp18 = pcp18.withColumn('year', f.lit('2018'))
    pcp19 = pcp18.withColumn('year', f.lit('2019'))
    pcp20 = pcp18.withColumn('year', f.lit('2020'))
    pcp21 = pcp18.withColumn('year', f.lit('2021'))

    pcp = pcp18.union(pcp19).union(pcp20).union(pcp21)
    pcp = pcp.select('member_id', 'provider_npi', 'year')
   
    alr = alr.withColumn('filename', f.regexp_replace('filename', 'P.A2620.ACO.QASSGN.D190213.T1208000', '2018'))
    alr = alr.withColumn('filename', f.regexp_replace('filename', 'P.A2620.ACO.QALR.D200213.T1200012_1-1', '2019'))
    alr = alr.withColumn('filename', f.regexp_replace('filename', 'Q42020ALR', '2020'))
    alr = alr.withColumn('filename', f.regexp_replace('filename', '2021Q2', '2021'))
    alr = alr.withColumn('filename', f.regexp_replace('filename', '2021 Q2 Assignment List Report', '2021'))
    alr = alr.withColumn("filename", f.trim(f.col("filename")))
    alr = alr.select('filename', 'BENE_MBI_ID')
 




    alrjoin = alr.join(df_join, [df_join.BENE_MBI_ID == alr.BENE_MBI_ID, df_join.claim_year == alr.filename], how='left')
    alrjoin.filter(alrjoin.provider_npi.isNull()).groupBy('claim_year').count().show()

    df = df.join(pcp, [df.BENE_MBI_ID == pcp.member_id, df.claim_year == pcp.year], how='left').drop(pcp.year, df.member_id)

    write_output(dfhcc)
    # convert array dtype column to string before converting to csv


	
if __name__ == "__main__":
	
    logger.info('START')
    main()
    logger.info('END') 
