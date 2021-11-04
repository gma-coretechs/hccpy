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


def load_datasets():
    cclf1 = spark.read.parquet(input_path + 'cclf1.parquet')
    cclf4 = spark.read.parquet(input_path + 'cclf4.parquet')
    cclf5 = spark.read.parquet(input_path + 'cclf5.parquet')
    cclf8 = spark.read.parquet(input_path + 'cclf8.parquet')
    return cclf1, cclf4, cclf5, cclf8


def write_output(df):
    logger.info("CREATING MASTER DIFF DATASET")
    logger.info("WRITING: {}".format(output_path + "NW_member_breakdown.parquet"))
    df.write.mode('overwrite').parquet(output_path + 'NW_member_breakdown.parquet')
    return df


def main():
    cclf1, cclf4, cclf5, cclf8 = load_datasets()

    cclf1 = cclf1.select('CUR_CLM_UNIQ_ID', 'PRVDR_OSCAR_NUM', 'FAC_PRVDR_NPI_NUM', 'ATNDG_PRVDR_NPI_NUM')
    cclf4 = cclf4.withColumn("file_year", f.split(f.col("source_file"), "/").getItem(6)).withColumn('source_year', f.substring(f.col('file_year'), -4, 4))
    cclf5 = cclf5.withColumn("file_year", f.split(f.col("source_file"), "/").getItem(6)).withColumn('source_year', f.substring(f.col('file_year'), -4, 4))
    cclf8 = cclf8.withColumn("file_year", f.split(f.col("source_file"), "/").getItem(6)).withColumn('source_year', f.substring(f.col('file_year'), -4, 4))

    cclf4 = cclf4.withColumn('claim_year', f.substring(f.col('CLM_THRU_DT'), 1, 4))
    cclf5 = cclf5.withColumn('claim_year', f.substring(f.col('CLM_THRU_DT'), 1, 4))
    
    cclf4 = cclf4.select('BENE_MBI_ID', 'CLM_DGNS_CD', 'source_year', 'claim_year', 'CUR_CLM_UNIQ_ID', 'CLM_THRU_DT', 'PRVDR_OSCAR_NUM')
    cclf4 = cclf4.withColumn('CLM_DGNS_CD', f.when(f.col('CLM_DGNS_CD') == '', None).otherwise(f.col('CLM_DGNS_CD')))
    # cclf4 = cclf4.groupBy('BENE_MBI_ID', 'source_year', 'claim_year').agg(f.array_distinct(f.collect_list('CLM_DGNS_CD')))
    #cclf4 = cclf4.dropna()
    cclf4 = cclf4.drop_duplicates()
    # cclf4 = cclf4.withColumnRenamed('array_distinct(collect_list(CLM_DGNS_CD))', 'CLM_DGNS_CD')

    cclf8 = cclf8.select('BENE_MBI_ID', 'BENE_SEX_CD', 'BENE_AGE', 'BENE_MDCR_STUS_CD', 'BENE_DUAL_STUS_CD', 'BENE_ORGNL_ENTLMT_RSN_CD', 'source_year' )
    cclf8 = cclf8.withColumn('concat_elig', f.concat(cclf8.BENE_DUAL_STUS_CD, cclf8.BENE_MDCR_STUS_CD))

    elig_comb = \
    {'0210' : 'CFA', '0410' : 'CFA', '0810' : 'CFA', '0211' : 'CFA', '0411' : 'CFA', '0811' : 'CFA',
     '0220' : 'CFD', '0420' : 'CFD', '0820': 'CFD', '0221': 'CFD', '0421': 'CFD', '0821': 'CFD',
     'NA10': 'CNA', 'NA11': 'CNA',
     'NA20': 'CND', 'NA21': 'CND',
     '0110' : 'CPA', '0310': 'CPA', '0510': 'CPA', '0610': 'CPA', '0111': 'CPA', '0311': 'CPA', '0511': 'CPA', '0611': 'CPA',
     '0120': 'CPD', '0320': 'CPD', '0520': 'CPD', '0620': 'CPD', '0121': 'CPD', '0321': 'CPD', '0521': 'CPD', '0621': 'CPD'}

    mapping_expr = f.create_map([f.lit(x) for x in chain(*elig_comb.items())])
    cclf8 = cclf8.replace(to_replace=elig_comb, subset=['concat_elig'])
    cclf8 = cclf8.select('BENE_MBI_ID', 'BENE_AGE', 'BENE_SEX_CD', 'BENE_ORGNL_ENTLMT_RSN_CD', 'concat_elig', 'source_year')
    cclf8 = cclf8.dropna()
    cclf8 = cclf8.drop_duplicates()
    w2 = Window.partitionBy("BENE_MBI_ID", 'source_year').orderBy(f.col("BENE_AGE"))
    cclf8 = cclf8.withColumn("row", f.row_number().over(w2)).filter(f.col("row") == 1).drop("row").orderBy(f.col('BENE_MBI_ID'))

    cclf5 = cclf5.select('BENE_MBI_ID', 'CLM_DGNS_1_CD', 'CLM_DGNS_2_CD', 'CLM_DGNS_3_CD', 'CLM_DGNS_4_CD', 'CLM_DGNS_5_CD', 'CLM_DGNS_6_CD',
                         'CLM_DGNS_7_CD', 'CLM_DGNS_8_CD', 'CLM_DGNS_9_CD', 'CLM_DGNS_10_CD', 'CLM_DGNS_11_CD', 'CLM_DGNS_12_CD', 'source_year', 
                         'claim_year', 'CUR_CLM_UNIQ_ID', 'RNDRG_PRVDR_NPI_NUM', 'CLM_THRU_DT')

    diag_cds = ['CLM_DGNS_1_CD', 'CLM_DGNS_2_CD', 'CLM_DGNS_3_CD', 'CLM_DGNS_4_CD', 'CLM_DGNS_5_CD', 'CLM_DGNS_6_CD', 'CLM_DGNS_7_CD',
                'CLM_DGNS_8_CD', 'CLM_DGNS_9_CD', 'CLM_DGNS_10_CD', 'CLM_DGNS_11_CD', 'CLM_DGNS_12_CD']

    cols = [f.when(~f.col(x).isin("~"), f.col(x)).alias(x)  for x in cclf5.columns]
    cclf5 = cclf5.select(*cols)
    cclf5 = cclf5.withColumn('DIAG_ARRAY', f.concat_ws(',', *diag_cds)) # concat diags and aggregate
    cclf5 = cclf5.select('BENE_MBI_ID', 'source_year', 'claim_year', 'DIAG_ARRAY', 'CUR_CLM_UNIQ_ID', 'RNDRG_PRVDR_NPI_NUM', 'CLM_THRU_DT')
    cclf5 = cclf5.dropna()
    cclf5 = cclf5.drop_duplicates()
    cclf5 = cclf5.withColumn('DIAG_ARRAY', f.split(f.col('DIAG_ARRAY'), ','))
    # cclf5 = cclf5.groupBy('BENE_MBI_ID', 'source_year', 'claim_year').agg(f.array_distinct(f.collect_list('DIAG_ARRAY')))
    # cclf5 = cclf5.withColumnRenamed('array_distinct(collect_list(DIAG_ARRAY))', 'DIAG_ARRAY')
    # cclf5 = cclf5.withColumn("DIAG_ARRAY",f.concat_ws(",",f.col("DIAG_ARRAY")))
    # cclf5 = cclf5.withColumn("DIAG_ARRAY", f.split(f.col("DIAG_ARRAY"), ",\s*").cast(ArrayType(StringType())).alias("DIAG_ARRAY"))
    cclf5 = cclf5.withColumnRenamed('DIAG_ARRAY', 'diagnosis_list')
    cclf4 = cclf4.withColumnRenamed('CLM_DGNS_CD', 'diagnosis_list')
    cclf4 = cclf4.withColumn('diagnosis_list', f.array(f.col('diagnosis_list')))

    cclf4 = cclf4.join(cclf1, on=['CUR_CLM_UNIQ_ID', 'PRVDR_OSCAR_NUM'], how='left')
    cclf4 = cclf4.select('BENE_MBI_ID', 'source_year', 'claim_year', 'CUR_CLM_UNIQ_ID', 'diagnosis_list', 'FAC_PRVDR_NPI_NUM', 'ATNDG_PRVDR_NPI_NUM', 'CLM_THRU_DT')
    cclf4 = cclf4.withColumnRenamed('FAC_PRVDR_NPI_NUM', 'facility_provider_npi')
    cclf4 = cclf4.withColumnRenamed('ATNDG_PRVDR_NPI_NUM', 'attending_provider_npi')

    cclf5 = cclf5.select('BENE_MBI_ID', 'source_year', 'claim_year', 'CUR_CLM_UNIQ_ID', 'RNDRG_PRVDR_NPI_NUM', 'diagnosis_list', 'CLM_THRU_DT')
    cclf5 = cclf5.withColumn('attending_provider_npi', f.col('RNDRG_PRVDR_NPI_NUM'))
    cclf5 = cclf5.withColumnRenamed('RNDRG_PRVDR_NPI_NUM', 'facility_provider_npi')
    
    cclf4 = cclf4.withColumn('claim_source', f.lit('CCLF4'))
    cclf5 = cclf5.withColumn('claim_source', f.lit('CCLF5'))

    cclf4 = cclf4.withColumn('claim_type', f.lit('Part A'))
    cclf5 = cclf5.withColumn('claim_type', f.lit('Part B'))

    cclf4 = cclf4.select('BENE_MBI_ID', 'source_year', 'claim_year', 'CUR_CLM_UNIQ_ID', 'diagnosis_list', 'facility_provider_npi', 'attending_provider_npi', 'CLM_THRU_DT', 'claim_source', 'claim_type' )
    cclf5 = cclf5.select('BENE_MBI_ID', 'source_year', 'claim_year', 'CUR_CLM_UNIQ_ID', 'diagnosis_list', 'facility_provider_npi', 'attending_provider_npi', 'CLM_THRU_DT', 'claim_source', 'claim_type' )

    master1 = cclf8.join(cclf4, on=['BENE_MBI_ID','source_year'], how='left')
    master2 = cclf8.join(cclf5, on =['BENE_MBI_ID','source_year'], how ='left')
    # logger.info('final columns after joins:' + str(master.columns))
    # logger.info('final row count:' + str(master.count()))
    master = master1.union(master2)
    # master = master.withColumn('diagnosis_code', f.explode(f.col('diagnosis_list')))
    # master = master.withColumn('diag_list', f.array(f.col('diagnosis_list')))
    # master = master.withColumn('diagnosis_list', f.array_distinct(f.concat(master.DIAG_ARRAY, master.PRNCPL_DGNS_CD, master.CLM_DGNS_CD)))
    # master = master.drop('DIAG_ARRAY', 'PRNCPL_DGNS_CD', 'CLM_DGNS_CD')	
    # master = master.drop('diagnosis_list')
    master = master.withColumn('BENE_AGE', f.col('BENE_AGE').cast('int'))
    master = master.withColumn('BENE_SEX_CD', f.when(f.col('BENE_SEX_CD')=='1', f.lit('M')).otherwise(f.lit('F')))
    master = master.withColumnRenamed('BENE_ORGNL_ENTLMT_RSN_CD', 'oerc')
  

    logger.info('master dtypes:' + str(master.dtypes))
    logger.info('final row count:' + str(master.count()))

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

    master = pd.concat([master_16, master_17, master_18, master_19, master_20, master_21], ignore_index=True)
       # master_18['diagnosis_list'] = [ [] if x is np.NaN else x for x in master_18['diagnosis_list'] ]

    # master= master[master['diagnosis_list'].notna()]
    he = HCCEngine(version="24_19")
    # master_18 = master_18.dropna()
    master['risk_profile'] = master.apply(lambda row: he.profile(row['diagnosis_list'], row['BENE_AGE'], row['BENE_SEX_CD'], row['concat_elig'], row['oerc']), axis=1)
      
    df_result = pd.DataFrame(master['risk_profile'].values.tolist())
    result_master = pd.concat([result_master, df_result['risk_score'], df_result['hcc_lst'], df_result['hcc_map']], axis=1)
    result_master = result_master[['BENE_MBI_ID', 'BENE_AGE', 'BENE_SEX_CD', 'concat_elig', 'diagnosis_list', 'oerc',
                                   'source_year', 'claim_year', 'CUR_CLM_UNIQ_ID', 'CLM_THRU_DT', 'claim_source', 'claim_type', 'facility_provider_npi', 'attending_provider_npi', 'hcc_lst', 'hcc_map']]
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
