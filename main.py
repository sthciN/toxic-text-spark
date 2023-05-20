import pandas as pd

# import os
# os.environ['PYSPARK_SUBMIT_ARGS'] = "--master mymaster --total-executor 2 --conf spark.driver.extraJavaOptions=-Dhttp.proxyHost=proxy.mycorp.com-Dhttp.proxyPort=1234 -Dhttp.nonProxyHosts=localhost|.mycorp.com|127.0.0.1 -Dhttps.proxyHost=proxy.mycorp.com -Dhttps.proxyPort=1234 -Dhttps.nonProxyHosts=localhost|.mycorp.com|127.0.0.1 pyspark-shell"

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.ml.feature import Tokenizer, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression


sc = (SparkSession.builder
                  .appName('Toxic Comment Classification')
                  .enableHiveSupport()
                  .config("spark.executor.memory", "2G")
                  .config("spark.driver.memory","5G")
                  .config("spark.executor.cores","3")
                  .config("spark.python.worker.memory","2G")
                  .config("spark.driver.maxResultSize","0")
                  .config("spark.sql.crossJoin.enabled", "true")
                  .config("spark.serializer","org.apache.spark.serializer.KryoSerializer")
                  .getOrCreate())

sc.sparkContext.setLogLevel('INFO')

print('VERSION', sc.version)

# tt = pd.read_csv('./input/train.csv')
# print(tt.head())
train_fields = ["id", "comment_text"]
train_tt = pd.read_csv('./input/train.csv')
test_tt = pd.read_csv('./input/test.csv')

train_tt.fillna('', inplace=True)
test_tt.fillna('', inplace=True)
print('*'*20)
print(train_tt.head(2))
print()
print(test_tt.head(2))
print('='*20)

# Create Spark DataFrmae
train_sdf = sc.createDataFrame(train_tt)
test_sdf = sc.createDataFrame(test_tt)
train_sdf.show(5)

train_sdf.filter(F.col('toxic') == 1).show(5)

# Enriching the dataframe with tokenized words
tokenizer = Tokenizer(inputCol="comment_text", outputCol="tokens")
tokens = tokenizer.transform(train_sdf)

# Term frequency
hashing_term_frequency = HashingTF(inputCol="tokens", outputCol="token_tf")
tf = hashing_term_frequency.transform(tokens)

tf.select('token_tf').take(2)

# Enriching the dataframe with TF-IDF
idf = IDF(inputCol="token_tf", outputCol="tfidf")
tfidf_model = idf.fit(tf)
tfidf = tfidf_model.transform(tf)
print(':tfidf.limit(5000):', tfidf.limit(5000))
print(':tfidf.select("tfidf").first():', tfidf.select("tfidf").first())

# Fit a lr on tfidf column
lr = LogisticRegression(featuresCol="tfidf", labelCol='toxic', regParam=0.01)
lr_model = lr.fit(tfidf.limit(5000))
trained_lr = lr_model.transform(tfidf)
print('trained_lr', trained_lr.select("id", "toxic", "probability", "prediction").show(10))
print('trained_lr.show(5)', trained_lr.show(5))

probability_convert = F.udf(lambda x: float(x[1]), T.FloatType())


print('proba>>>', trained_lr.withColumn("proba", probability_convert("probability")).select("proba", "prediction").show(5))

# Test dataset
test_tokens = tokenizer.transform(test_sdf)
test_tf = hashing_term_frequency.transform(test_tokens)
test_tfidf = tfidf_model.transform(test_tf)

test_res = test_sdf.select('id')
test_res.head()


test_probs = []
for col in [i for i in train_tt.columns if i not in train_fields]:
    print(col)
    lr = LogisticRegression(featuresCol="token_tf", labelCol=col, regParam=0.01)
    lr_model = lr.fit(tfidf)
    transformed = lr_model.transform(test_tfidf)
    test_res = test_res.join(transformed.select('id', 'probability'), on="id")
    test_res = test_res.withColumn(col, probability_convert('probability')).drop("probability")
    test_res.show(3)
    
test_res.coalesce(1).write.csv('./output/toxic_text_spark_lr.csv', mode='overwrite', header=True)
