import numpy as np
import pandas as pd
import csv

from sklearn.feature_extraction import DictVectorizer as DV
from sklearn.decomposition import PCA
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC


from datetime import datetime


month = {}
month['Jan'] = 1
month['Feb'] = 2
month['Mar'] = 3
month['Apr'] = 4
month['May'] = 5
month['Jun'] = 6
month['Jul'] = 7
month['Aug'] = 8
month['Sep'] = 9
month['Oct'] = 10
month['Nov'] = 11
month['Dec'] = 12

def extractFromFile(train):
	for i in range(len(train)):
		if pd.isnull(train.SP_rating[i])==False: 
			train.SP_rating[i] = int(train.SP_rating[i].replace('sp_rating','').strip())
		if pd.isnull(train.Moody_rating[i])==False:
			train.Moody_rating[i] = int(train.Moody_rating[i].replace('moody_rating','').strip())
		if pd.isnull(train.Coupon_Frequency[i])==False:
			train.Coupon_Frequency[i] = int(train.Coupon_Frequency[i].replace('Coupon Frequency','').strip())
		# if pd.isnull(train.Seniority[i])==False:
		# 	train.Seniority[i] = int(train.Seniority[i].replace('Seniority','').strip())


		if pd.isnull(train.Issue_Date[i])==False and pd.isnull(train.Maturity_Date[i])==False: 
			component = train.Issue_Date[i].strip().split('-')
			issue_Date = datetime(2000+int(component[2]),month[component[1]],int(component[0])).date()
			component = train.Maturity_Date[i].strip().split('-')
			maturity_Date = datetime(2000+int(component[2]),month[component[1]],int(component[0])).date()
			if issue_Date > maturity_Date:
				train.Issue_Date[i] = np.nan
			else:	
				timedeltaDate = maturity_Date-issue_Date
				train.Issue_Date[i] = timedeltaDate.days
		else:
			train.Issue_Date[i] = np.nan

	return train



def normalizeData(data):
	data.Days_to_Maturity = (data.Days_to_Maturity-data.Days_to_Maturity.mean()) / float(data.Days_to_Maturity.max()-data.Days_to_Maturity.min()) * 10
	data.SP_rating = (data.SP_rating-data.SP_rating.mean()) / float(data.SP_rating.max()-data.SP_rating.min()) * 10
	data.Moody_rating = (data.Moody_rating-data.Moody_rating.mean()) / float(data.Moody_rating.max()-data.Moody_rating.min()) * 10
	data.Seniority = (data.Seniority-data.Seniority.mean()) / float(data.Seniority.max()-data.Seniority.min()) * 10
	data.Days_to_Settle = (data.Days_to_Settle-data.Days_to_Settle.mean()) / float(data.Days_to_Settle.max()-data.Days_to_Settle.min()) * 10
	data.Coupon_Frequency = (data.Coupon_Frequency-data.Coupon_Frequency.mean()) / float(data.Coupon_Frequency.max()-data.Coupon_Frequency.min()) * 10
	return data




data_dir = '/home/aayush/Desktop/codes/gsquantify15/bondclustering/'	# needs trailing slash

# validation split, both files with headers and the Happy column
train_file = data_dir + 'Final_Training_Data.csv'
test_file = data_dir + 'Final_Test_Data.csv'



train = pd.read_csv( train_file )
test = pd.read_csv( test_file )

train = extractFromFile(train)
test = extractFromFile(test)


t_train = train.Risk_Stripe
x_train = train.ISIN

x_test = test.ISIN

#labelling target
le = LabelEncoder()
t_train = le.fit_transform(t_train)


train.rename(columns={'Issue_Date':'Days_to_Maturity'}, inplace=True)
train.drop('Maturity_Date', axis=1, inplace=True)
train.drop('ISIN', axis=1, inplace=True)
train.drop('Risk_Stripe', axis=1, inplace=True)
train.drop('Is_Emerging_Market', axis=1, inplace=True)



test.rename(columns={'Issue_Date':'Days_to_Maturity'}, inplace=True)
test.drop('Maturity_Date', axis=1, inplace=True)
test.drop('ISIN', axis=1, inplace=True)
test.drop('Is_Emerging_Market', axis=1, inplace=True)


numeric_cols = ['SP_rating','Moody_rating','Days_to_Settle','Days_to_Maturity','Coupon_Frequency']


#normalize Days_to_Maturity
train.Days_to_Maturity = train.Days_to_Maturity / float(train.Days_to_Maturity.max()) * 12
test.Days_to_Maturity = test.Days_to_Maturity / float(test.Days_to_Maturity.max()) * 12







# targetIndex = {}
# for i in range(len(t_train)):
# 	if targetIndex.has_key(t_train[i]):
# 		targetIndex[t_train[i]].append(i)
# 	else:
# 		targetIndex[t_train[i]] = [i]


# keys = train.columns.tolist()

# targetFeature = {}
# for key,value in targetIndex.items():
# 	targetFeature[key] = {}
# 	for j in range(len(value)):
# 		if j==0:
# 			targetFeature[key] = dict.fromkeys(keys,[])
# 		for k in keys:
# 			if pd.isnull(train[k][value[j]])==False:
# 				targetFeature[key][k].append(train[k][value[j]])


# print targetFeature['Stripe 0']['SP_rating']

# for i in range(len(train)):
# 	for j in keys:
# 		if pd.isnull(train[j][i]):
# 			if j in numeric_cols:
# 				train[j][i] = sum(targetFeature[t_train[i]][j])/float(len(targetFeature[t_train[i]][j]))
















# # numerical
f_num_train = train[ numeric_cols ].as_matrix()
f_num_test = test[ numeric_cols ].as_matrix()




#Fill missing values 
imp = Imputer(missing_values='NaN',strategy='mean',axis=0)
f_num_train = imp.fit_transform(f_num_train)
f_num_test = imp.transform(f_num_test)




cat_train = train.drop(numeric_cols, axis=1)
cat_test = test.drop(numeric_cols, axis=1)


f_cat_train = cat_train.T.to_dict().values()
f_cat_test = cat_test.T.to_dict().values()

	

# vectorize
vectorizer = DV( sparse = False )
vec_f_cat_train = vectorizer.fit_transform( f_cat_train )
vec_f_cat_test = vectorizer.transform( f_cat_test )


#Fill missing values 
imp = Imputer(missing_values='NaN',strategy='most_frequent',axis=0)
vec_f_cat_train = imp.fit_transform(vec_f_cat_train)
vec_f_cat_test = imp.transform(vec_f_cat_test)


# complete x
f_train = np.hstack((f_num_train, vec_f_cat_train ))
f_test = np.hstack((f_num_test, vec_f_cat_test ))



clf = OneVsOneClassifier(LinearSVC(random_state=0))
clf.fit(f_train,t_train)
y_test = clf.predict(f_test)

y_train = clf.predict(f_train)


y_test = le.inverse_transform(y_test)



ofile = open(data_dir+'output.csv','w')
writer = csv.writer(ofile)
writer.writerow(['ISIN','Risk_Stripe'])
for i in range(len(x_test)):
	writer.writerow([x_test[i],y_test[i]])
ofile.close()
