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
output_path = '/data/data_science/master_file/'

'''
elig : str
               The eligibility segment of the patient.
               Allowed values are as follows:
               - "CFA": Community Full Benefit Dual Aged
               - "CFD": Community Full Benefit Dual Disabled
               - "CNA": Community NonDual Aged
               - "CND": Community NonDual Disabled
               - "CPA": Community Partial Benefit Dual Aged
               - "CPD": Community Partial Benefit Dual Disabled
               - "INS": Long Term Institutional
               - "NE": New Enrollee
               - "SNPNE": SNP NE
'''

def load_datasets():
    cclf1 = spark.read.parquet(input_path + 'cclf1.parquet')
    cclf4 = spark.read.parquet(input_path + 'cclf4.parquet')
    cclf5 = spark.read.parquet(input_path + 'cclf5.parquet')
    cclf8 = spark.read.parquet(input_path + 'cclf8.parquet')
    return cclf1, cclf4, cclf5, cclf8


def write_output(df):
    logger.info("CREATING MASTER DATASET")
    logger.info("WRITING: {}".format(output_path + "master_v2.parquet"))
    df.write.mode('overwrite').parquet(output_path + 'master_v2.parquet')
    return df
		

def main():
    cclf1, cclf4, cclf5, cclf8 = load_datasets()
	# source_year_expr = f.substring(f.col('source_file'), 72, 4)
	
    cclf1 = cclf1.withColumn("file_year", f.split(f.col("source_file"), "/").getItem(6)).withColumn('year', f.substring(f.col('file_year'), -4, 4))
    cclf4 = cclf4.withColumn("file_year", f.split(f.col("source_file"), "/").getItem(6)).withColumn('year', f.substring(f.col('file_year'), -4, 4))
    cclf5 = cclf5.withColumn("file_year", f.split(f.col("source_file"), "/").getItem(6)).withColumn('year', f.substring(f.col('file_year'), -4, 4))
    cclf8 = cclf8.withColumn("file_year", f.split(f.col("source_file"), "/").getItem(6)).withColumn('year', f.substring(f.col('file_year'), -4, 4))
   
    cclf1 = cclf1.select('BENE_MBI_ID', 'PRNCPL_DGNS_CD', 'year')	
    cclf1 = cclf1.withColumn('PRNCPL_DGNS_CD', f.when(f.col('PRNCPL_DGNS_CD') == '', None).otherwise(f.col('PRNCPL_DGNS_CD')))
    cclf1 = cclf1.groupBy('BENE_MBI_ID', 'year').agg(f.array_distinct(f.collect_list('PRNCPL_DGNS_CD')))
    cclf1 = cclf1.dropna()
    cclf1 = cclf1.drop_duplicates()
    cclf1 = cclf1.withColumnRenamed('array_distinct(collect_list(PRNCPL_DGNS_CD))', 'PRNCPL_DGNS_CD')
    
    cclf4 = cclf4.select('BENE_MBI_ID', 'CLM_DGNS_CD', 'year')	
    cclf4 = cclf4.withColumn('CLM_DGNS_CD', f.when(f.col('CLM_DGNS_CD') == '', None).otherwise(f.col('CLM_DGNS_CD')))
    cclf4 = cclf4.groupBy('BENE_MBI_ID', 'year').agg(f.array_distinct(f.collect_list('CLM_DGNS_CD')))
    cclf4 = cclf4.dropna()
    cclf4 = cclf4.drop_duplicates()
    cclf4 = cclf4.withColumnRenamed('array_distinct(collect_list(CLM_DGNS_CD))', 'CLM_DGNS_CD')
    
    cclf8 = cclf8.select('BENE_MBI_ID', 'BENE_SEX_CD', 'BENE_AGE', 'BENE_MDCR_STUS_CD', 'BENE_DUAL_STUS_CD', 'BENE_ORGNL_ENTLMT_RSN_CD', 'year' )
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
    cclf8 = cclf8.select('BENE_MBI_ID', 'BENE_AGE', 'BENE_SEX_CD', 'BENE_ORGNL_ENTLMT_RSN_CD', 'concat_elig', 'year')
    cclf8 = cclf8.dropna()
    cclf8 = cclf8.drop_duplicates()
    w2 = Window.partitionBy("BENE_MBI_ID", 'year').orderBy(f.col("BENE_AGE"))
    cclf8 = cclf8.withColumn("row", f.row_number().over(w2)).filter(f.col("row") == 1).drop("row").orderBy(f.col('BENE_MBI_ID'))


    cclf5 = cclf5.select('BENE_MBI_ID', 'CLM_DGNS_1_CD', 'CLM_DGNS_2_CD', 'CLM_DGNS_3_CD', 'CLM_DGNS_4_CD', 'CLM_DGNS_5_CD', 'CLM_DGNS_6_CD',
    		         'CLM_DGNS_7_CD', 'CLM_DGNS_8_CD', 'CLM_DGNS_9_CD', 'CLM_DGNS_10_CD', 'CLM_DGNS_11_CD', 'CLM_DGNS_12_CD', 'year')

    diag_cds = ['CLM_DGNS_1_CD', 'CLM_DGNS_2_CD', 'CLM_DGNS_3_CD', 'CLM_DGNS_4_CD', 'CLM_DGNS_5_CD', 'CLM_DGNS_6_CD', 'CLM_DGNS_7_CD',
	        'CLM_DGNS_8_CD', 'CLM_DGNS_9_CD', 'CLM_DGNS_10_CD', 'CLM_DGNS_11_CD', 'CLM_DGNS_12_CD'] 

    cols = [f.when(~f.col(x).isin("~"), f.col(x)).alias(x)  for x in cclf5.columns]	
    cclf5 = cclf5.select(*cols)
    cclf5 = cclf5.withColumn('DIAG_ARRAY', f.concat_ws(',', *diag_cds)) # concat diags and aggregate 
    cclf5 = cclf5.select('BENE_MBI_ID', 'year', 'DIAG_ARRAY')	
    cclf5 = cclf5.dropna()
    cclf5 = cclf5.drop_duplicates()
    cclf5 = cclf5.groupBy('BENE_MBI_ID', 'year').agg(f.array_distinct(f.collect_list('DIAG_ARRAY')))
    cclf5 = cclf5.withColumnRenamed('array_distinct(collect_list(DIAG_ARRAY))', 'DIAG_ARRAY')
    cclf5 = cclf5.withColumn("DIAG_ARRAY",f.concat_ws(",",f.col("DIAG_ARRAY")))
    cclf5 = cclf5.withColumn("DIAG_ARRAY", f.split(f.col("DIAG_ARRAY"), ",\s*").cast(ArrayType(StringType())).alias("DIAG_ARRAY"))
    
    master = cclf8.join(cclf1, on=['BENE_MBI_ID', 'year'], how='left')
    master = master.join(cclf4, on=['BENE_MBI_ID','year'], how='left')
    master = master.join(cclf5, on =['BENE_MBI_ID','year'], how ='left')
    logger.info('final columns after joins:' + str(master.columns))
    logger.info('final row count:' + str(master.count()))


    # master  = master.withColumn('c2',f.coalesce(df.c2,f.array())) 
    master = master.withColumn('diagnosis_list', f.array_distinct(f.concat(master.DIAG_ARRAY, master.PRNCPL_DGNS_CD, master.CLM_DGNS_CD)))	
    master = master.drop('DIAG_ARRAY', 'PRNCPL_DGNS_CD', 'CLM_DGNS_CD')

    # print(master.show(3, truncate=False))
    write_output(master)
	
	
if __name__ == "__main__":
	
    logger.info('START')
    main()
    logger.info('END') 
