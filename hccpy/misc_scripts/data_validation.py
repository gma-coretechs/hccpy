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
    logger.info("CREATING MASTER DATASET")
    logger.info("WRITING: {}".format(output_path + "data_validation_with_diag.parquet"))
    df.write.mode('overwrite').parquet(output_path + 'data_validation_with_diag.parquet')
    return df
		

def main():
    dfRAF = spark.read.csv('wasbs://rdp-uploads@coretechsnmdev.blob.core.windows.net/NW_diag_HCC_raf_new_V22.csv',header=True, sep='|')
    dfT1 = spark.read.csv('wasbs://rdp-uploads@coretechsnmdev.blob.core.windows.net/GMA_T1_Template_2017_2021.csv',header=True)
    diab = spark.read.csv('wasbs://rdp-uploads@coretechsnmdev.blob.core.windows.net/NW_diab_cmd_memb_level.csv', header=True)

    dfRAF = dfRAF.drop('BENE_SEX_CD')
    
    dfT1_2020 = dfT1.filter(f.col('file_source')== 'P.A2620.ACO.QALR.D200210.T1200012_1-1')
    dfRAF_2020= dfRAF.filter(f.col('claim_year')=='2018')
    
    df20 = dfT1_2020.join(dfRAF_2020, on='BENE_MBI_ID', how='left')
    

    #  df20 = spark.read.csv('wasbs://rdp-uploads@coretechsnmdev.blob.core.windows.net/PY2019.csv', header=True)
    cols_list = [['HCC_COL_1', 'HCC1'], ['HCC_COL_2', 'HCC2'], ['HCC_COL_3', 'HCC6'], ['HCC_COL_4', 'HCC8'], ['HCC_COL_5', 'HCC9'], ['HCC_COL_6', 'HCC10'],
                 ['HCC_COL_7', 'HCC11'],['HCC_COL_8', 'HCC12'],['HCC_COL_9', 'HCC17'],['HCC_COL_10', 'HCC18'],['HCC_COL_11', 'HCC19'],['HCC_COL_12', 'HCC21'],
                 ['HCC_COL_13', 'HCC22'],['HCC_COL_14', 'HCC23'],['HCC_COL_15', 'HCC27'],['HCC_COL_16', 'HCC28'],['HCC_COL_17', 'HCC29'], ['HCC_COL_18', 'HCC33'],
                 ['HCC_COL_19', 'HCC34'],['HCC_COL_20', 'HCC35'],['HCC_COL_21', 'HCC39'],['HCC_COL_22', 'HCC40'],['HCC_COL_23', 'HCC46'],['HCC_COL_24', 'HCC47'],['HCC_COL_25', 'HCC48'],['HCC_COL_26', 'HCC54'],
                 ['HCC_COL_27', 'HCC55'],['HCC_COL_28', 'HCC57'],['HCC_COL_29', 'HCC58'],['HCC_COL_30', 'HCC70'],['HCC_COL_31', 'HCC71'],['HCC_COL_32', 'HCC72'],['HCC_COL_33', 'HCC73'],
                 ['HCC_COL_34', 'HCC74'],['HCC_COL_35', 'HCC75'],['HCC_COL_36', 'HCC76'],['HCC_COL_37', 'HCC77'],['HCC_COL_38', 'HCC78'],['HCC_COL_39', 'HCC79'],['HCC_COL_40', 'HCC80'],['HCC_COL_41', 'HCC82'],
                 ['HCC_COL_42', 'HCC83'],['HCC_COL_43', 'HCC84'],['HCC_COL_44', 'HCC85'],['HCC_COL_45', 'HCC86'],['HCC_COL_46', 'HCC87'],['HCC_COL_47', 'HCC88'],
                 ['HCC_COL_48', 'HCC96'],['HCC_COL_49', 'HCC99'],['HCC_COL_50', 'HCC100'],['HCC_COL_51', 'HCC103'],['HCC_COL_52', 'HCC104'],['HCC_COL_53', 'HCC106'],['HCC_COL_54', 'HCC107'],['HCC_COL_55', 'HCC108'],
                 ['HCC_COL_56', 'HCC110'],['HCC_COL_57', 'HCC111'],['HCC_COL_58', 'HCC112'],['HCC_COL_59', 'HCC114'],['HCC_COL_60', 'HCC115'],['HCC_COL_61', 'HCC122'],['HCC_COL_62', 'HCC124'],['HCC_COL_63', 'HCC134'],
                 ['HCC_COL_64', 'HCC135'],['HCC_COL_65', 'HCC136'],['HCC_COL_66', 'HCC137'],['HCC_COL_67', 'HCC157'],['HCC_COL_68', 'HCC158'],['HCC_COL_69', 'HCC161'],['HCC_COL_70', 'HCC162'],['HCC_COL_71', 'HCC166'],
     		 ['HCC_COL_72', 'HCC167'],['HCC_COL_73', 'HCC169'],['HCC_COL_74', 'HCC170'],['HCC_COL_75', 'HCC173'],['HCC_COL_76', 'HCC176'],['HCC_COL_77', 'HCC186'],['HCC_COL_78', 'HCC188'],['HCC_COL_79', 'HCC189']]    


    old_name = [str(cols_list[i][0]) for i in range(len(cols_list))] + ['BENE_MBI_ID']
    new_name = [str(cols_list[i][1]) for i in range(len(cols_list))] + ['BENE_MBI_ID']
    df20hcc = df20.select(old_name).toDF(*new_name)

    new_name.remove('BENE_MBI_ID')
    df20hcc = df20hcc.select(*[f.when(f.col(x) == 1, f.lit(x)).otherwise(f.lit('')).alias(x) for x in new_name], 'BENE_MBI_ID')     
    df20hcc = df20hcc.withColumn('HCC_ALR', f.concat_ws(',', *new_name))
    df20hcc = df20hcc.withColumn("HCC_ALR", f.split(f.col("HCC_ALR"), ",\s*").cast(ArrayType(StringType())).alias("HCC_ALR"))
    df20hcc = df20hcc.withColumn("HCC_ALR", f.expr("filter(HCC_ALR, elem -> elem != '')"))
    df20hcc = df20hcc.select('BENE_MBI_ID', 'HCC_ALR')
    
    df20 = df20.join(df20hcc, on=['BENE_MBI_ID'], how='left')
   
    df20 = df20.withColumn("hcc_nobrackets", f.regexp_replace('hcc_lst',"\\[", ""))
    df20 = df20.withColumn('hcc_nobrackets', f.regexp_replace('hcc_nobrackets', '\\]', ''))
    df20 = df20.withColumn('hcc_nobrackets', f.regexp_replace('hcc_nobrackets', "\'", ''))
    df20 = df20.withColumn('hcc_nobrackets', f.regexp_replace('hcc_nobrackets', " ", ''))
    df20 = df20.withColumn('hcc_lst', f.split('hcc_nobrackets', ',')) 

    df20 = df20.withColumn('HCC_GMA', f.expr("filter(hcc_lst, x -> x not rlike '[_]')"))
    df20 = df20.withColumn('HCC_DIFF', f.array_except('HCC_ALR','HCC_GMA'))
    
    df20 = df20.filter(f.col('HCC_GMA').isNotNull())

    df20 = df20.select('BENE_MBI_ID', 'BENE_1ST_NAME', 'BENE_LAST_NAME', 'BENE_SEX_CD', 'ESRD_SCORE', 'DIS_SCORE', 'AGDU_SCORE', 'AGND_SCORE', 'DEM_ESRD_SCORE', 'DEM_DIS_SCORE', 'DEM_AGDU_SCORE', 'DEM_AGND_SCORE', 'BENE_AGE', 'concat_elig', 'oerc', 'source_year', 'claim_year',  'hcc_map', 'risk_score', 'risk_score_diff', 'details', 'hcc_lst_diff', 'hcc_map_diff', 'details_diff', 'cum_hcc_diff', 'HCC_ALR', 'HCC_GMA', 'HCC_DIFF')

    diab = diab.select('BENE_MBI_ID', 'diagnosis_list', 'source_year', 'claim_year')
    diab = diab.filter(f.col('claim_year')=='2018').filter(f.col('source_year')=='2018')
    diab = diab.drop('claim_year','source_year')
    df20 = df20.join(diab, on='BENE_MBI_ID', how='left')
    
    df20 = df20.withColumn('HCC_ALR', f.col('HCC_ALR').cast('string'))
    df20 = df20.withColumn('HCC_GMA', f.col('HCC_GMA').cast('string'))
    df20 = df20.withColumn('HCC_DIFF', f.col('HCC_DIFF').cast('string'))
    df20 = df20.withColumn('hcc_lst_diff', f.col('hcc_lst_diff').cast('string'))

    df20 = df20.withColumn("HCC_ALR", f.regexp_replace('HCC_ALR',"\\,", "|"))
    df20 = df20.withColumn("HCC_GMA", f.regexp_replace('HCC_GMA',"\\,", "|"))
    df20 = df20.withColumn("HCC_DIFF", f.regexp_replace('HCC_DIFF',"\\,", "|"))
    df20 = df20.withColumn("hcc_lst_diff", f.regexp_replace('hcc_lst_diff',"\\,", "|"))
    # print(master.show(3, truncate=False))
   
    df20 = df20.withColumnRenamed('risk_score', 'PROXY_RAF_SCORE') 
    df20 = df20.withColumnRenamed('risk_score_diff', 'OPPORTUNITY_RAF_SCORE') 
    df20 = df20.withColumnRenamed('hcc_map', 'HCCs_MAPPED_FROM_CCLFs') 
    df20 = df20.withColumnRenamed('details', 'HCC_RAF_DETAILS_COEFFICIENTS') 
    df20 = df20.withColumnRenamed('hcc_lst_diff', 'OPPORTUNITYvsPROXY_HCC_DELTA') 
    df20 = df20.withColumnRenamed('hcc_map_diff', 'OPPORTUNITYvsPROXY_HCC_DETAILS_DELTA') 
    df20 = df20.withColumnRenamed('cum_hcc_diff', 'CUMULATIVE_OPPORTUNITY_HCC_DELTA') 
    df20 = df20.withColumnRenamed('HCC_ALR', 'HCCs_MAPPED_FROM_ALRs') 
    df20 = df20.withColumnRenamed('HCC_DIFF', 'CCLFvsALR_HCC_DELTA') 
    df20 = df20.withColumnRenamed('diagnosis_list', 'CCLF_DIAGNOSIS_LIST') 

    write_output(df20)
    df20.coalesce(1).write.mode('overwrite').option("header", "true").csv('wasbs://rdp-uploads@coretechsnmdev.blob.core.windows.net/data_validation_with_diag.csv')	
	
if __name__ == "__main__":
	
    logger.info('START')
    main()
    logger.info('END') 
