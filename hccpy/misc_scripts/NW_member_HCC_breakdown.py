from itertools import chain
import logging
import sys

from pyspark.sql import functions as f
from pyspark.sql.session import SparkSession
from pyspark.sql.window import Window
from pyspark.sql.types import ArrayType, StringType

from hccV2421.hcc_2421 import HCCEngine
import pandas as pd
import numpy as np
pd.set_option('display.max_colwidth', 1000)
pd.set_option('display.max_columns', 500)


spark = SparkSession.builder.getOrCreate()
spark.conf.set("spark.driver.maxResultSize", "2g")
# spark.conf.set("spark.sql.execution.arrow.enabled", "true")

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
    logger.info("CREATING MASTER DIFF DATASET")
    logger.info("WRITING: {}".format(output_path + "NW_member_HCC_breakdown_JanFeb2020.parquet"))
    df.write.mode('overwrite').parquet(output_path + 'NW_member_HCC_breakdown_JanFeb2020.parquet')
    return df


def main():

    master = spark.read.parquet('/data/data_science/powerBI/NW_member_breakdown.parquet')
    '''
    master_16 = master.filter(f.col('claim_year') <= '2016')
    master_17 = master.filter(f.col('claim_year') == '2017')
    master_18 = master.filter(f.col('claim_year') == '2018')
    master_19 = master.filter(f.col('claim_year') == '2019')
    master_20 = master.filter(f.col('claim_year') == '2020')
    master_21 = master.filter(f.col('claim_year') == '2021')

    master_16 = master_16.toPandas()
    master_17 = master_17.toPandas()
    master_18 = master_18.toPandas()
    master_19 = master_19.toPandas()
    master_20 = master_20.toPandas()
    master_21 = master_21.toPandas()
    '''
        
    master_20 = master.filter((f.col('CLM_THRU_DT') >= '2020-01-01') & (f.col('CLM_THRU_DT') < '2020-03-01'))
    master_20 = master_20.toPandas()
    #master = pd.concat([master_16, master_17, master_18, master_19, master_20, master_21], ignore_index=True)
       # master_18['diagnosis_list'] = [ [] if x is np.NaN else x for x in master_18['diagnosis_list'] ]

    # master= master[master['diagnosis_list'].notna()]
    he = HCCEngine(version="24_19")
    # master_18 = master_18.dropna()
    master_20['risk_profile'] = master_20.apply(lambda row: he.profile(row['diagnosis_list'], row['BENE_AGE'], row['BENE_SEX_CD'], row['concat_elig'], row['oerc']), axis=1)
      
    df_result = pd.DataFrame(master_20['risk_profile'].values.tolist())
    result_master = pd.concat([master_20, df_result['hcc_lst'], df_result['hcc_map']], axis=1)
    result_master = result_master[['BENE_MBI_ID', 'CUR_CLM_UNIQ_ID', 'BENE_AGE', 'BENE_SEX_CD', 'concat_elig', 'oerc',
                                   'source_year', 'claim_year', 'CLM_THRU_DT', 'claim_source', 'claim_type', 'facility_provider_npi', 'attending_provider_npi', 'diagnosis_list', 'hcc_lst', 'hcc_map']]
    fields = result_master.columns
    for field in fields:
        result_master[field] = [str(x) for x in result_master[field]]

    # logger.info('final row count:' + str(result_master.count()))
    result_master = spark.createDataFrame(result_master)
    # drop risk profile column
    # write_output(result_master)

    # df = spark.read.parquet('/data/data_science/raf/NW_master_raf.parquet')
    # df = df.withColumn('diagnosis_list', f.array(f.col('diagnosis_list')))
    # df = df.withColumn('hcc_lst',f.array(f.col('hcc_lst')))

    # df = df.withColumn('diag_lag',f.lag(df['diagnosis_list']).over(Window.partitionBy("BENE_MBI_ID").orderBy('BENE_MBI_ID','year')))
    # df = df.withColumn('hcc_lag',f.lag(df['hcc_lst']).over(Window.partitionBy("BENE_MBI_ID").orderBy('BENE_MBI_ID','year')))

    # df = df.withColumn('diff_diag', f.array_except(f.col('diagnosis_list'), f.col('diag_lag')))
    # df = df.withColumn('diff_hcc', f.array_except(f.col('hcc_lst'), f.col('hcc_lag')))
 
    write_output(result_master)
    # write_output(master)


if __name__ == "__main__":

    logger.info('START')
    main()
    logger.info('END')
