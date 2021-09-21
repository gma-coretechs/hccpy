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
# spark.conf.set("spark.sql.execution.arrow.enabled", "true")

logger = logging.getLogger(__name__)
log_handler = logging.StreamHandler(sys.stdout)
log_handler.setFormatter(logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
log_handler.setLevel(logging.DEBUG)
logger.addHandler(log_handler)
logger.setLevel(logging.DEBUG)


# input_path = '/data/raw/'
output_path = '/data/data_science/raf/'


def write_output(df):
    logger.info("CREATING MASTER DIFF DATASET")
    logger.info("WRITING: {}".format(output_path + "NW_diag_HCC_and_diff_raf_temp.parquet"))
    df.write.mode('overwrite').parquet(output_path + 'NW_diag_HCC_and_diff_raf_temp.parquet')
    return df


def main():
    
    master = spark.read.parquet('/data/data_science/raf/NW_diag_HCC_raf.parquet')
    master = master.select('BENE_MBI_ID', 'BENE_AGE', 'BENE_SEX_CD', 'concat_elig', 'diagnosis_list', 'oerc', 'source_year', 'claim_year', 'hcc_lst', 'hcc_map', 'risk_score', 'details')
    master = master.withColumn('BENE_AGE', f.col('BENE_AGE').cast('int'))

    master = master.withColumn("hcc_nobrackets", f.regexp_replace('hcc_lst',"\\[", ""))
    master = master.withColumn('hcc_nobrackets', f.regexp_replace('hcc_nobrackets', '\\]', ''))
    master = master.withColumn('hcc_lst', f.split('hcc_nobrackets', ','))
    
    window_member = (Window.partitionBy('BENE_MBI_ID').orderBy('BENE_MBI_ID', 'source_year', 'claim_year').rangeBetween(Window.unboundedPreceding, 0))
    master = master.withColumn('cum_hcc',f.array_distinct(f.flatten(f.collect_list('hcc_lst').over(window_member))))
    master = master.withColumn('cum_hcc_diff',f.array_except('cum_hcc','hcc_lst'))

    master = master.withColumn("diag_nobrackets", f.regexp_replace('diagnosis_list',"\\[", ""))
    master = master.withColumn('diag_nobrackets', f.regexp_replace('diag_nobrackets', '\\]', ''))
    master = master.withColumn('diagnosis_list', f.split('diag_nobrackets', ','))
    
    window_member = (Window.partitionBy('BENE_MBI_ID').orderBy('BENE_MBI_ID', 'source_year', 'claim_year').rangeBetween(Window.unboundedPreceding, 0))
    master = master.withColumn('cum_diag',f.array_distinct(f.flatten(f.collect_list('diagnosis_list').over(window_member))))
    master = master.withColumn('cum_diag_diff',f.array_except('cum_diag','diagnosis_list'))

    master = master.withColumn('total_diag', f.array_distinct(f.concat('diagnosis_list','cum_diag_diff')))
    master = master.select('BENE_MBI_ID', 'BENE_AGE', 'BENE_SEX_CD', 'concat_elig', 'oerc', 'source_year', 'claim_year', 'hcc_lst', 'hcc_map', 'risk_score', 'cum_hcc_diff', 'cum_diag_diff', 'total_diag', 'details')

    master_18 = master.filter((f.col('source_year')=='2018') & (f.col('claim_year')=='2018'))
    master_19 = master.filter((f.col('source_year')=='2019') & (f.col('claim_year')=='2019'))
    master_20 = master.filter((f.col('source_year')=='2020') & (f.col('claim_year')=='2020'))
    master_21 = master.filter((f.col('source_year')=='2021') & (f.col('claim_year')=='2021'))
 
    master_18 = master_18.toPandas()
    master_19 = master_19.toPandas()
    master_20 = master_20.toPandas()
    master_21 = master_21.toPandas()


    master_18 = master_18[master_18['total_diag'].notna()]
    he = HCCEngine(version="23")
    # master_18 = master_18.dropna()
    master_18['risk_profile_diff'] = master_18.apply(lambda row: he.profile(row['total_diag'], row['BENE_AGE'], row['BENE_SEX_CD'], row['concat_elig'], row['oerc']), axis=1)
    
    he = HCCEngine(version="24_19")
    master_19 = master_19[master_19['total_diag'].notna()]
    master_19['risk_profile_diff'] = master_19.apply(lambda row: he.profile(row['total_diag'], row['BENE_AGE'], row['BENE_SEX_CD'], row['concat_elig'], row['oerc']), axis=1)

    he = HCCEngine(version="24_19")
    master_20 = master_20[master_20['total_diag'].notna()]
    master_20['risk_profile_diff'] = master_20.apply(lambda row: he.profile(row['total_diag'], row['BENE_AGE'], row['BENE_SEX_CD'], row['concat_elig'], row['oerc']), axis=1)

    he = HCCEngine(version="24_21")
    master_21 = master_21[master_21['total_diag'].notna()]
    master_21['risk_profile_diff'] = master_21.apply(lambda row: he.profile(row['total_diag'], row['BENE_AGE'], row['BENE_SEX_CD'], row['concat_elig'], row['oerc']), axis=1)


    result_master = pd.concat([master_18, master_19, master_20, master_21 ], axis=0, ignore_index=True)
    df_result_diff = pd.DataFrame(result_master['risk_profile_diff'].values.tolist())
    df_result_diff = df_result_diff.rename(columns={'risk_score': 'risk_score_diff', 'hcc_lst': 'hcc_lst_diff', 'hcc_map': 'hcc_map_diff', 'details':'details_diff'})
    result_master = pd.concat([result_master, df_result_diff['risk_score_diff'], df_result_diff['hcc_lst_diff'], df_result_diff['hcc_map_diff'], df_result_diff['details_diff']], axis=1)
    
    result_master = result_master[['BENE_MBI_ID', 'BENE_AGE', 'BENE_SEX_CD', 'concat_elig', 'oerc', 'source_year', 'claim_year', 'hcc_lst', 'hcc_map' ,'risk_score', 'details', 'cum_hcc_diff']]
    fields = result_master.columns
    for field in fields:
        result_master[field] = [str(x) for x in result_master[field]]

    # logger.info('final row count:' + str(result_master.count()))
    result_master = spark.createDataFrame(result_master)
    # drop risk profile column
    # write_output(result_master)

    write_output(result_master)


if __name__ == "__main__":

    logger.info('START')
    main()
    logger.info('END')
