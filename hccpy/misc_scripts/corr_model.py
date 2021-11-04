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


input_path = '/data/measures/'
output_path = '/data/data_science/master_file/'


def load_datasets():
    hba1c = spark.read.parquet(input_path + 'hba1c/hba1c_results_*.parquet')
    unplanned = spark.read.parquet(input_path + 'unplanned_admission/unplanned_admission_results_*.parquet')
    raf_hcc = spark.read.parquet('/data/data_science/raf/NW_diag_HCC_raf_new_V22.parquet')
    pcw = spark.read.parquet(input_path + 'pcw/results.parquet')
    pcp20 = spark.read.parquet('/data/attribution/attribution_2020.parquet')    
    return hba1c, unplanned, raf_hcc, pcw, pcp20


def write_output(df):
    logger.info("CREATING MASTER DATASET")
    logger.info("WRITING: {}".format(output_path + "master_v2.parquet"))
    df.write.mode('overwrite').parquet(output_path + 'master_v2.parquet')
    return df
		

def main():
    hba1c, unplanned, raf_hcc, pcw, pcp20 = load_datasets()
	# source_year_expr = f.substring(f.col('source_file'), 72, 4)
	
        # print(master.show(3, truncate=False))
    
    hba1c = hba1c.withColumnRenamed('numerator', 'numerator_hba1c')
    unplanned = unplanned.withColumnRenamed('numerator','numerator_unp')
    pcw = pcw.withColumnRenamed('numerator','numerator_pcw')
    
    pcw = pcw.select('member_id', 'numerator_pcw')
    hba1c = hba1c.select('member_id', 'numerator_hba1c')
    unplanned = unplanned.select('member_id', 'numerator_unp')
    chbp20 = chbp20.select('member_id', 'numerator_chbp')
    pcp20 = pcp20.select('member_id', 'provider_npi', 'provider_specialty', 'count_of_visits', 'latest_visit_date')
    raf_hcc = raf_hcc.select('BENE_MBI_ID', 'BENE_AGE', 'BENE_SEX_CD', 'concat_elig', 'oerc', 'source_year', 'claim_year', 'hcc_lst', 'risk_score')
    




    df = df.fillna(0)
    df = df.withColumn('medicaid_flag', df['medicaid_flag'].cast('integer'))
    df = df.withColumn('outcome', df['outcome'].cast(DoubleType()))

    feature_cols = [col for col in df.columns if col not in ['member_id', 'outcome']]
    remove_feature_cols = []
    feature_cols = list(set(feature_cols) - set(remove_feature_cols))

    '''
    ##########################################################################
    Before SMOTE model
    ##########################################################################
    '''
    print('\n \n')
    print('=============================================================== \n')
    print('Before SMOTE Model results ')
    print('=============================================================== \n')
    #  train, test = df.randomSplit([0.8, 0.2], seed=12345)
    dataset_size = float(df.select("outcome").count())
    numPositives = df.select("outcome").where('outcome == 1').count()
    per_ones = (float(numPositives) / float(dataset_size)) * 100
    print('The number of ones are {}'.format(numPositives))
    print('Percentage of ones are {}'.format(per_ones))

    bucketizer = Bucketizer(splits=[15, 30, 38, 55], inputCol='age', outputCol='age_groups')
    df_buck = bucketizer.setHandleInvalid('keep').transform(df)
    df_buck = df_buck.withColumn('age_group_31-38', f.when(f.col('age_groups') == 1.0, f.lit(1)).otherwise(f.lit(0)))
    df_buck = df_buck.withColumn('age_group_38-55', f.when(f.col('age_groups') == 2.0, f.lit(1)).otherwise(f.lit(0)))

    binarizer = Binarizer(threshold=0.5, inputCol='outcome', outputCol='label')
    binarizedDF = binarizer.transform(df_buck)
    binarizedDF = binarizedDF.drop('outcome', 'age', 'age_groups')

    feature_cols1 = [col for col in binarizedDF.columns if col not in ['member_id', 'label']]
    assembler = VectorAssembler(inputCols=feature_cols1, outputCol='features')

    assembled = assembler.transform(binarizedDF)
    print(assembled.describe().show(vertical=True))

    (trainData, testData) = assembled.randomSplit([0.75, 0.25], seed=42)
    print('Distribution of Ones and Zeros in trainData is: ', trainData.groupBy('label').count().take(3))

    lr = LogisticRegression(labelCol='label', featuresCol='features', maxIter=100)
    lrModel = lr.fit(trainData)

    print("Intercept: " + str(lrModel.intercept))
    modelcoefficients = np.array(lrModel.coefficients)
    names = [x['name']
             for x in sorted(trainData.schema["features"]
                             .metadata["ml_attr"]["attrs"]['numeric'], key=lambda x: x['idx'])]
    matchcoefs = np.column_stack((modelcoefficients, np.array(names)))
    matchcoefsdf = pd.DataFrame(matchcoefs)
    matchcoefsdf.columns = ['Coefvalue', 'Feature']
    print(matchcoefsdf)

    predictions = lrModel.transform(testData)
    results = predictions.select('probability', 'prediction', 'label')
    print(results.show(10, truncate=False))

    evaluator = BinaryClassificationEvaluator()
    print('Test Data Area under ROC score is : ',
          evaluator.evaluate(predictions))

    accuracy = predictions.filter(
        predictions.label == predictions.prediction).count() / float(predictions.count())
    print('Accuracy : ', accuracy)

    # compute TN, TP, FN, and FP
    print(predictions.groupBy('label', 'prediction').count().show())
    # Calculate the elements of the confusion matrix
    TN = predictions.filter('prediction = 0 AND label = prediction').count()
    TP = predictions.filter('prediction = 1 AND label = prediction').count()
    FN = predictions.filter('prediction = 0 AND label <> prediction').count()
    FP = predictions.filter('prediction = 1 AND label <> prediction').count()

    # calculate accuracy, precision, recall, and F1-score
    accuracy = (TN + TP) / (TN + TP + FN + FP + 1)
    precision = TP / (TP + FP + 1)
    recall = TP / (TP + FN + 1)
    F = 2 * (precision * recall) / (precision + recall + 1)
    print('n precision: %0.3f' % precision)
    print('n recall: %0.3f' % recall)
    print('n accuracy: %0.3f' % accuracy)
    print('n F1 score: %0.3f' % F)
    print('\n \n')


    write_output(master)
	
	
if __name__ == "__main__":
	
    logger.info('START')
    main()
    logger.info('END') 
