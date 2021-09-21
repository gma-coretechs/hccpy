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


input_path = '/data/data_science/raf/'
output_path = '/data/data_science/raf/'


def main():
    dfRAF = spark.read.parquet(input_path + 'NW_diag_HCC_raf_new_V22.parquet') 
    dfRAF = dfRAF.withColumn('hcc_lst', f.col('hcc_lst').cast('string'))
    dfRAF = dfRAF.withColumn('cum_hcc_diff', f.col('cum_hcc_diff').cast('string'))
    logger.info("CREATING csv DATASET")
    logger.info("WRITING: {}".format(output_path + "NW_diag_HCC_raf_new_V22.csv"))

    dfRAF.coalesce(1).write.mode('overwrite').option("header", "true").option("delimiter", '|').csv('/data/data_science/raf/NW_diag_HCC_raf_new_V22.csv')


if __name__ == "__main__":

    logger.info('START')
    main()
    logger.info('END')
