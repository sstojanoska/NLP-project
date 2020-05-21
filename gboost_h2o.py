import h2o
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.random_forest import H2ORandomForestEstimator

h2o.init(nthreads = -1, max_mem_size = 8)


#call: from gboost_h2o import Boosting/Forest(path to generated model(CSV))

def prepareData(path):
	data = h2o.import_file(path)
	data['polarity'] = data['polarity'].asfactor()
	splits = data.split_frame(ratios=[0.7, 0.15], seed=1)
	train = splits[0]
	valid = splits[1]
	test = splits[2]
	y = 'polarity'
	x = list(data.columns)
	x.remove(y)
	return train, test, x, y

def Boosting(path):
	train, test, x, y  = prepareData(path)
	gbm_fit1 = H2OGradientBoostingEstimator(model_id='gbm_fit1', seed=1)
	gbm_fit1.train(x=x, y=y, training_frame=train)
	gbm_perf1 = gbm_fit1.model_performance(test)
	print(gbm_perf1.confusion_matrix())
	print(gbm_fit1.F1())

def Forest(train, test, x, y):
	train, test, x, y  = prepareData(path)
	rf_fit1 = H2ORandomForestEstimator(model_id='rf_fit1', seed=1)
	rf_fit1.train(x=x, y=y, training_frame=train)
	rf_perf1 = rf_fit1.model_performance(test)
	print(rf_fit1.F1())


Boosting('MODELS/5m.csv')


