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
    logger.info("CREATING FINAL PCP ATTRIBUTION LIST DATASET")
    logger.info("WRITING: {}".format(output_path + "Final_PCP_attribution.parquet"))
    df.write.mode('overwrite').parquet(output_path + 'Final_PCP_attribution.parquet')
    df.coalesce(1).write.mode('overwrite').option("header", "true").option("delimiter", '|').csv('wasbs://rdp-uploads@coretechsnmdev.blob.core.windows.net/Final_PCP_attribution.csv')
    return df
		

def main():
    
    # df = spark.read.parquet('/data/data_science/raf/NW_diag_HCC_raf_new_V22_V24.parquet')
    # df = df.select('Bl.dropna()ENE_MBI_ID', 'claim_year', 'hcc_map', 'hcc_map_diff', 'hcc_map_diags', 'hcc_map_diff_diags') 
    pcp18 = spark.read.parquet('/data/attribution/attribution_2018.parquet')   
    pcp19 = spark.read.parquet('/data/attribution/attribution_2019.parquet')
    pcp20 = spark.read.parquet('/data/attribution/attribution_2020.parquet')
    pcp21 = spark.read.parquet('/data/attribution/attribution_2021.parquet')
    
    alrpcp18 = spark.read.csv('wasbs://rdp-uploads@coretechsnmdev.blob.core.windows.net/2018Q4ALRT4.csv', header= True)
    alrpcp19 = spark.read.csv('wasbs://rdp-uploads@coretechsnmdev.blob.core.windows.net/Beneficiary Assignment List/2019/P.A2620.ACO.QALR.D200213.T1200012_1-4.csv', header= True)
    alrpcp20 = spark.read.csv('wasbs://rdp-uploads@coretechsnmdev.blob.core.windows.net/Beneficiary Assignment List/2020/P.A2620.ACO.QALR.D201112.T1200012_1-4.csv', header= True)
    alrpcp21 = spark.read.csv('wasbs://rdp-uploads@coretechsnmdev.blob.core.windows.net/2021 Q2 Assignment List Report Table 4.csv', header=True) 

    alrpcp18 = alrpcp18.withColumn('year', f.lit('2018'))
    alrpcp19 = alrpcp19.withColumn('year', f.lit('2019'))
    alrpcp20 = alrpcp20.withColumn('year', f.lit('2020'))
    alrpcp21 = alrpcp21.withColumn('year', f.lit('2021'))

    # pcp = pcp18.union(pcp19).union(pcp20).union(pcp21)
    # pcp = pcp.select('member_id', 'provider_npi', 'year')
   
    w = Window().partitionBy("BENE_MBI_ID").orderBy(f.desc("PCS_COUNT"))
    alrpcp21 =(alrpcp21.withColumn("rank", f.dense_rank().over(w)))
    alrpcp21 = alrpcp21.where(f.col("rank") == 1)
    result21 = alrpcp21.select('BENE_MBI_ID', 'NPI_used', 'year').join(pcp21.select('member_id', 'provider_npi'),pcp21.member_id == alrpcp21.BENE_MBI_ID, how='left')

    w = Window().partitionBy("BENE_MBI_ID").orderBy(f.desc("PCS_COUNT"))
    alrpcp20 =(alrpcp20.withColumn("rank", f.dense_rank().over(w)))
    alrpcp20 = alrpcp20.where(f.col("rank") == 1)
    result20 = alrpcp20.select('BENE_MBI_ID', 'NPI_used', 'year').join(pcp20.select('member_id', 'provider_npi'),pcp20.member_id == alrpcp20.BENE_MBI_ID, how='left')

    result19 = alrpcp19.select('BENE_MBI_ID', 'NPI_used', 'year').join(pcp19.select('member_id', 'provider_npi'),pcp19.member_id == alrpcp19.BENE_MBI_ID, how='left')
    
    
    result18 = alrpcp18.select('MBI', 'Individual NPI ', 'year').join(pcp18.select('member_id', 'provider_npi'),pcp18.member_id == alrpcp18.MBI, how='left')

    result21 = result21.withColumn('final_npi', f.when(f.col('provider_npi').isNull(), f.col('NPI_used')).otherwise(f.col('provider_npi')))
    result20 = result20.withColumn('final_npi', f.when(f.col('provider_npi').isNull(), f.col('NPI_used')).otherwise(f.col('provider_npi')))
    result19 = result19.withColumn('final_npi', f.when(f.col('provider_npi').isNull(), f.col('NPI_used')).otherwise(f.col('provider_npi')))
    result18 = result18.withColumn('final_npi', f.when(f.col('provider_npi').isNull(), f.col('Individual NPI ')).otherwise(f.col('provider_npi')))


    result18 = result18.withColumnRenamed('MBI', 'BENE_MBI_ID')
    result18 = result18.withColumnRenamed('Individual NPI ', 'NPI_used')    

    result = result18.union(result19).union(result20).union(result21)
    
    result = result.withColumn('join_key', f.concat(f.col('BENE_MBI_ID'), f.col('NPI_used'), f.col('year')))

    cclf5 = spark.read.parquet('/data/new_raw/cclf5.parquet')
    cclf5 = cclf5.select('BENE_MBI_ID', 'RNDRG_PRVDR_NPI_NUM', 'CLM_THRU_DT', 'CLM_FROM_DT')
    cclf5 = cclf5.withColumn('claim_year', f.substring(f.col('CLM_THRU_DT'), 0, 4))
    cclf5 = cclf5.withColumn('join_key', f.concat(f.col('BENE_MBI_ID'), f.col('RNDRG_PRVDR_NPI_NUM'), f.col('claim_year')))
 
    df_join = result.join(cclf5.select('RNDRG_PRVDR_NPI_NUM', 'CLM_THRU_DT', 'CLM_FROM_DT', 'claim_year', 'join_key'),on = 'join_key', how='inner')
    df_join = df_join.drop_duplicates()

    w = Window().partitionBy('BENE_MBI_ID', 'year').orderBy(f.desc("CLM_THRU_DT"))
    df_join =(df_join.withColumn("rank", f.dense_rank().over(w)))
    df_join_final = df_join.where(f.col("rank") == 1)
    df_join_final = df_join_final.drop_duplicates()

    df_join_final = df_join_final.select('BENE_MBI_ID', 'RNDRG_PRVDR_NPI_NUM', 'year')
    df_join_final = df_join_final.drop_duplicates()
    
    df_impute = result.join(df_join_final, on=['BENE_MBI_ID', 'year'], how='left')
    df_impute = df_impute.withColumn('FINAL_PCP_NPI', f.when(f.col('provider_npi').isNull(), f.col('RNDRG_PRVDR_NPI_NUM')).otherwise(f.col('final_npi'))) 
    df_impute = df_impute.withColumn('FINAL_PCP_NPI', f.when(f.col('provider_npi').isNull() & f.col('RNDRG_PRVDR_NPI_NUM').isNull(), f.col('NPI_used')).otherwise(f.col('FINAL_PCP_NPI')))

    df_final = df_impute.select('BENE_MBI_ID', 'year', 'FINAL_PCP_NPI')
    df_final = df_final.dropna()
    df_final = df_final.drop_duplicates()    
    write_output(df_final)


	
if __name__ == "__main__":
	
    logger.info('START')
    main()
    logger.info('END') 
