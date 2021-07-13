from hccV2421.hcc_2421 import HCCEngine

import logging
import sys

from pyspark.sql import functions as f
from pyspark.sql.session import SparkSession
import pandas as pd
pd.set_option('display.max_colwidth', None)
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
input_path = 's3://horizon-prod-pipeline/views/'
member_path = 's3://horizon-prod-pipeline/member_sds/latest/'
output_path = 's3://horizon-prod-pipeline/data_science/hccV2421/'


def load_datasets():
    diagnoses = spark.read.parquet(input_path + 'claim_header_w_dos.parquet')
    member = spark.read.parquet(member_path + 'member.parquet')
    return diagnoses, member


def logs(member, diagnoses):
    logger.info('claim unique member_ids: '
                + str(member.select('member_id').distinct().count()))
    #  logger.info('claim unique claim_ids: '
    #              + str(member.select('claim_id').distinct().count()))
    logger.info('claim row count: '
                + str(member.count()))
    logger.info('diagnoses unique member_ids: '
                + str(diagnoses.select('member_id').distinct().count()))
    logger.info('diagnoses row count: '
                + str(diagnoses.count()))


def write_output(df):
    logger.info("CREATING output")
    logger.info("WRITING: {}".format(output_path
                + "member_hcc_raf.parquet"))
    df.write.mode('overwrite').parquet(output_path
                                       + 'member_hcc_raf.parquet')
    return df


def main():

    diagnoses, member = load_datasets()
    member = member.select('member_id', 'date_of_birth', 'gender', 'eligibility')
    diagnoses = diagnoses.select('diagnoses', 'first_dos', 'coretechs_member_id')

    diagnoses = diagnoses.withColumnRenamed('diagnoses', 'diagnosis_codes')
    diagnoses = diagnoses.withColumnRenamed('first_dos', 'first_dos_dx')

    diagnoses = diagnoses.drop_duplicates()

    member = member.withColumn('age', f.ceil(f.datediff(f.current_date(), f.to_date(f.col('date_of_birth'))) / 365.25))
    member = member.drop_duplicates()
    # member = member.select('member_id', 'age', 'gender', 'eligibility.product_category') \
    #                .withColumn('pc_elig', f.col('product_category').getItem(0))
    member = member.select('member_id', 'age', 'gender')
    member = member.withColumn('pc_elig', f.lit('CNA'))
    data = diagnoses.join(member, member.member_id == diagnoses.coretechs_member_id, how='left')
    data = data.limit(10000)
    data = data.toPandas()

    he = HCCEngine(version="24_19")
    # rp = he.profile(data['diagnosis_list'], data['age'], data['gender'], data['pc_elig'])
    data = data.dropna()
    data['risk_score'] = data.apply(lambda row: he.profile(row['diagnosis_codes'], row['age'], row['gender'], row['pc_elig']), axis=1)

    # rp = he.profile(["E1169", "I5030", "I509", "I211", "I209", "R05"],
    #                 age=70, sex="M", elig="CNA")
    # print(rp["risk_score"])


if __name__ == "__main__":
    main()
