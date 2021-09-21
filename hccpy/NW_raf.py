from hccV2421.hcc_2421 import HCCEngine

import logging
import sys

from pyspark.sql import functions as f
from pyspark.sql.session import SparkSession
import pandas as pd
pd.set_option('display.max_colwidth', 1000)
pd.set_option('display.max_columns', 500)


spark = SparkSession.builder.getOrCreate()
spark.conf.set("spark.sql.execution.arrow.enabled", "true")

logger = logging.getLogger(__name__)
log_handler = logging.StreamHandler(sys.stdout)
log_handler.setFormatter(logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
log_handler.setLevel(logging.DEBUG)
logger.addHandler(log_handler)
logger.setLevel(logging.DEBUG)


# Paths
input_path = '/data/data_science/master_file/'
output_path = '/data/data_science/raf/'


def load_datasets():
    master = spark.read.parquet(input_path + 'master_v2.parquet')
    return master


def write_output(df):
    logger.info("CREATING raf")
    logger.info("WRITING: {}".format(output_path
                + "NW_master_raf.parquet"))
    df.write.mode('overwrite').parquet(output_path
                                       + 'NW_master_raf.parquet')
    return df


def main():

    master = load_datasets()
    master = master.select('BENE_MBI_ID', 'BENE_AGE', 'BENE_SEX_CD', 'concat_elig', 'diagnosis_list', 'year', 'BENE_ORGNL_ENTLMT_RSN_CD')
    master = master.withColumn('BENE_AGE', f.col('BENE_AGE').cast('int'))
    master = master.withColumn('BENE_SEX_CD', f.when(f.col('BENE_SEX_CD')=='1', f.lit('M')).otherwise(f.lit('F')))
    master = master.withColumnRenamed('BENE_ORGNL_ENTLMT_RSN_CD', 'oerc')

    master_18 = master.filter(f.col('year')=='2018')
    master_19 = master.filter(f.col('year')=='2019')
    master_20 = master.filter(f.col('year')=='2020')
    master_21 = master.filter(f.col('year')=='2021')
    
    master_18 = master_18.toPandas()
    master_19 = master_19.toPandas()
    master_20 = master_20.toPandas()
    master_21 = master_21.toPandas()
    
    he = HCCEngine(version="23")
    master_18 = master_18.dropna()
    master_18['risk_profile'] = master_18.apply(lambda row: he.profile(row['diagnosis_list'], row['BENE_AGE'], row['BENE_SEX_CD'], row['concat_elig'], row['oerc']), axis=1)

    he = HCCEngine(version="24_19")
    master_19 = master_19.dropna()
    master_19['risk_profile'] = master_19.apply(lambda row: he.profile(row['diagnosis_list'], row['BENE_AGE'], row['BENE_SEX_CD'], row['concat_elig'], row['oerc']), axis=1)

    he = HCCEngine(version="24_19")
    master_20 = master_20.dropna()
    master_20['risk_profile'] = master_20.apply(lambda row: he.profile(row['diagnosis_list'], row['BENE_AGE'], row['BENE_SEX_CD'], row['concat_elig'], row['oerc']), axis=1)

    he = HCCEngine(version="24_21")
    master_21 = master_21.dropna()
    master_21['risk_profile'] = master_21.apply(lambda row: he.profile(row['diagnosis_list'], row['BENE_AGE'], row['BENE_SEX_CD'], row['concat_elig'], row['oerc']), axis=1)

    
    result_master = pd.concat([master_18, master_19, master_20, master_21 ], axis=0, ignore_index=True)   
    df_result = pd.DataFrame(result_master['risk_profile'].values.tolist())
    result_master = pd.concat([result_master, df_result['risk_score'], df_result['hcc_lst'], df_result['hcc_map']], axis=1)


    fields = result_master.columns
    for field in fields:
        result_master[field] = [str(x) for x in result_master[field]]	

	
    # logger.info('final row count:' + str(result_master.count()))
    result_master = spark.createDataFrame(result_master)
    write_output(result_master)
  
if __name__ == "__main__":
    
    logger.info('START')   
    main()
    logger.info('END')   
