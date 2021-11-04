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
    # dfRAF = spark.read.csv('wasbs://rdp-uploads@coretechsnmdev.blob.core.windows.net/NW_diag_HCC_raf_new_V22.csv',header=True, sep='|')
    dfT1 = spark.read.csv('wasbs://rdp-uploads@coretechsnmdev.blob.core.windows.net/ALRT1_2018_2021_csv.csv',header=True)
    dfT2 = spark.read.csv('wasbs://rdp-uploads@coretechsnmdev.blob.core.windows.net/2021 Q2 Assignment List Report Table 1.csv',header=True)
    # diab = spark.read.csv('wasbs://rdp-uploads@coretechsnmdev.blob.core.windows.net/NW_diab_cmd_memb_level.csv', header=True)
    
    dfT2 = dfT2.withColumn('filename', f.lit('2021Q2'))
    
    dfT2 = dfT2.select('filename', 'BENE_MBI_ID', 'BENE_HIC_NUM', 'BENE_1ST_NAME', 'BENE_LAST_NAME', 'BENE_SEX_CD', 'BENE_BRTH_DT', 'BENE_DEATH_DT', 'GEO_SSA_CNTY_CD_NAME', 'GEO_SSA_STATE_NAME', 'STATE_COUNTY_CD', 'IN_VA_MAX', 'VA_TIN', 'VA_NPI', 'CBA_FLAG', 'ASSIGNMENT_TYPE', 'ASSIGNED_BEFORE', 'ASG_STATUS', 'PARTD_MONTHS', 'EnrollFlag1', 'EnrollFlag2', 'EnrollFlag3', 'EnrollFlag4', 'EnrollFlag5', 'EnrollFlag6', 'EnrollFlag7', 'EnrollFlag8', 'EnrollFlag9', 'EnrollFlag10', 'EnrollFlag11', 'EnrollFlag12', 'HCC_version', 'HCC_COL_1', 'HCC_COL_2', 'HCC_COL_3', 'HCC_COL_4', 'HCC_COL_5', 'HCC_COL_6', 'HCC_COL_7', 'HCC_COL_8', 'HCC_COL_9', 'HCC_COL_10', 'HCC_COL_11', 'HCC_COL_12', 'HCC_COL_13', 'HCC_COL_14', 'HCC_COL_15', 'HCC_COL_16', 'HCC_COL_17', 'HCC_COL_18', 'HCC_COL_19', 'HCC_COL_20', 'HCC_COL_21', 'HCC_COL_22', 'HCC_COL_23', 'HCC_COL_24', 'HCC_COL_25', 'HCC_COL_26', 'HCC_COL_27', 'HCC_COL_28', 'HCC_COL_29', 'HCC_COL_30', 'HCC_COL_31', 'HCC_COL_32', 'HCC_COL_33', 'HCC_COL_34', 'HCC_COL_35', 'HCC_COL_36', 'HCC_COL_37', 'HCC_COL_38', 'HCC_COL_39', 'HCC_COL_40', 'HCC_COL_41', 'HCC_COL_42', 'HCC_COL_43', 'HCC_COL_44', 'HCC_COL_45', 'HCC_COL_46', 'HCC_COL_47', 'HCC_COL_48', 'HCC_COL_49', 'HCC_COL_50', 'HCC_COL_51', 'HCC_COL_52', 'HCC_COL_53', 'HCC_COL_54', 'HCC_COL_55', 'HCC_COL_56', 'HCC_COL_57', 'HCC_COL_58', 'HCC_COL_59', 'HCC_COL_60', 'HCC_COL_61', 'HCC_COL_62', 'HCC_COL_63', 'HCC_COL_64', 'HCC_COL_65', 'HCC_COL_66', 'HCC_COL_67', 'HCC_COL_68', 'HCC_COL_69', 'HCC_COL_70', 'HCC_COL_71', 'HCC_COL_72', 'HCC_COL_73', 'HCC_COL_74', 'HCC_COL_75', 'HCC_COL_76', 'HCC_COL_77', 'HCC_COL_78', 'HCC_COL_79', 'HCC_COL_80', 'HCC_COL_81', 'HCC_COL_82', 'HCC_COL_83', 'HCC_COL_84', 'HCC_COL_85', 'HCC_COL_86', 'HCC_COL_87', 'HCC_COL_88', 'HCC_COL_89', 'HCC_COL_90', 'BENE_RSK_R_SCRE_01', 'BENE_RSK_R_SCRE_02', 'BENE_RSK_R_SCRE_03', 'BENE_RSK_R_SCRE_04', 'BENE_RSK_R_SCRE_05', 'BENE_RSK_R_SCRE_06', 'BENE_RSK_R_SCRE_07', 'BENE_RSK_R_SCRE_08', 'BENE_RSK_R_SCRE_09', 'BENE_RSK_R_SCRE_10', 'BENE_RSK_R_SCRE_11', 'BENE_RSK_R_SCRE_12', 'ESRD_SCORE', 'DIS_SCORE', 'AGDU_SCORE', 'AGND_SCORE', 'DEM_ESRD_SCORE', 'DEM_DIS_SCORE', 'DEM_AGDU_SCORE', 'DEM_AGND_SCORE', 'NEW_ENROLLEE') 
    # dfT1_2020 = dfT1.filter(f.col('file_source')== 'P.A2620.ACO.QALR.D200210.T1200012_1-1')
    # dfRAF_2020= dfRAF.filter(f.col('claim_year')=='2018')
    
    # df20 = dfT1_2020.join(dfRAF_2020, on='BENE_MBI_ID', how='left')
    

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

    df_union = dfT1.union(dfT2)
    other_cols = ['filename', 'BENE_MBI_ID' , 'BENE_HIC_NUM', 'BENE_1ST_NAME', 'BENE_LAST_NAME', 'BENE_SEX_CD', 'BENE_BRTH_DT', 'BENE_DEATH_DT', 'GEO_SSA_CNTY_CD_NAME', 'GEO_SSA_STATE_NAME', 'STATE_COUNTY_CD', 'IN_VA_MAX', 'VA_TIN', 'VA_NPI', 'CBA_FLAG', 'ASSIGNMENT_TYPE', 'ASSIGNED_BEFORE', 'ASG_STATUS', 'PARTD_MONTHS', 'EnrollFlag1', 'EnrollFlag2', 'EnrollFlag3', 'EnrollFlag4', 'EnrollFlag5', 'EnrollFlag6', 'EnrollFlag7', 'EnrollFlag8', 'EnrollFlag9', 'EnrollFlag10', 'EnrollFlag11', 'EnrollFlag12', 'HCC_version' , 'BENE_RSK_R_SCRE_01', 'BENE_RSK_R_SCRE_02', 'BENE_RSK_R_SCRE_03', 'BENE_RSK_R_SCRE_04', 'BENE_RSK_R_SCRE_05', 'BENE_RSK_R_SCRE_06', 'BENE_RSK_R_SCRE_07', 'BENE_RSK_R_SCRE_08', 'BENE_RSK_R_SCRE_09', 'BENE_RSK_R_SCRE_10', 'BENE_RSK_R_SCRE_11', 'BENE_RSK_R_SCRE_12', 'ESRD_SCORE', 'DIS_SCORE', 'AGDU_SCORE', 'AGND_SCORE', 'DEM_ESRD_SCORE', 'DEM_DIS_SCORE', 'DEM_AGDU_SCORE', 'DEM_AGND_SCORE', 'NEW_ENROLLEE']

    old_name = [str(cols_list[i][0]) for i in range(len(cols_list))] + other_cols 
    new_name = [str(cols_list[i][1]) for i in range(len(cols_list))] + other_cols 
    dfhcc = df_union.select(old_name).toDF(*new_name)

    new_name = list(set(new_name) - set(other_cols))

    dfhcc = dfhcc.select(*[f.when(f.col(x) == 1, f.lit(x)).otherwise(f.lit('')).alias(x) for x in new_name], *other_cols)
    dfhcc = dfhcc.withColumn('HCC_ALR', f.concat_ws(',', *new_name))
    dfhcc = dfhcc.withColumn("HCC_ALR", f.split(f.col("HCC_ALR"), ",\s*").cast(ArrayType(StringType())).alias("HCC_ALR"))
    dfhcc = dfhcc.withColumn("HCC_ALR", f.expr("filter(HCC_ALR, elem -> elem != '')"))
    dfhcc = dfhcc.select(*other_cols, 'HCC_ALR')
    
    write_output(dfhcc)
    # convert array dtype column to string before converting to csv


	
if __name__ == "__main__":
	
    logger.info('START')
    main()
    logger.info('END') 
