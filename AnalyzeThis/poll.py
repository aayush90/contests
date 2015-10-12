import numpy as np
import pandas as pd
import csv

from sklearn.feature_extraction import DictVectorizer as DV
from sklearn.decomposition import PCA
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC,SVC
from sklearn.ensemble import GradientBoostingClassifier


from datetime import datetime


def extractFromFile(data):

	data['mvar_2'] = data['mvar_2'].apply(lambda x: x.replace('$', ''))
	data['mvar_3'] = data['mvar_3'].apply(lambda x: x.replace('$', ''))
	data['mvar_4'] = data['mvar_4'].apply(lambda x: x.replace('$', ''))
	data['mvar_5'] = data['mvar_5'].apply(lambda x: x.replace('$', ''))
	data['mvar_6'] = data['mvar_6'].apply(lambda x: x.replace('$', ''))

	data['mvar_2'] = data['mvar_2'].apply(lambda x: float(x.replace(',', '')))
	data['mvar_3'] = data['mvar_3'].apply(lambda x: float(x.replace(',', '')))
	data['mvar_4'] = data['mvar_4'].apply(lambda x: float(x.replace(',', '')))
	data['mvar_5'] = data['mvar_5'].apply(lambda x: float(x.replace(',', '')))
	data['mvar_6'] = data['mvar_6'].apply(lambda x: float(x.replace(',', '')))


	data.mvar_23.fillna(value=0,inplace=True)
	data.mvar_24.fillna(value=0,inplace=True)
	data.mvar_25.fillna(value=0,inplace=True)
	data.mvar_26.fillna(value=0,inplace=True)
	data.mvar_28.fillna(value=0,inplace=True)

	data['mvar_7'] = data['mvar_7'].apply(lambda x: float(x))
	data['mvar_8'] = data['mvar_8'].apply(lambda x: float(x))
	data['mvar_9'] = data['mvar_9'].apply(lambda x: float(x))
	data['mvar_10'] = data['mvar_10'].apply(lambda x: float(x))
	data['mvar_11'] = data['mvar_11'].apply(lambda x: float(x))
	data['mvar_23'] = data['mvar_23'].apply(lambda x: float(x))
	data['mvar_24'] = data['mvar_24'].apply(lambda x: float(x))
	data['mvar_25'] = data['mvar_25'].apply(lambda x: float(x))
	data['mvar_26'] = data['mvar_26'].apply(lambda x: float(x))
	data['mvar_28'] = data['mvar_28'].apply(lambda x: float(x))
	data['mvar_30'] = data['mvar_30'].apply(lambda x: float(x))

	return data



def normalizeData(data):
	data.mvar_2 = data.mvar_2 / float(data.mvar_2.max()-data.mvar_2.min()) * 20
	data.mvar_3 = data.mvar_3 / float(data.mvar_3.max()-data.mvar_3.min()) * 20
	data.mvar_4 = data.mvar_4 / float(data.mvar_4.max()-data.mvar_4.min()) * 20
	data.mvar_5 = data.mvar_5 / float(data.mvar_5.max()-data.mvar_5.min()) * 20
	data.mvar_6 = data.mvar_6 / float(data.mvar_6.max()-data.mvar_6.min()) * 20

	data.mvar_7 = data.mvar_7 / float(data.mvar_7.max()-data.mvar_7.min()) * 20
	data.mvar_8 = data.mvar_8 / float(data.mvar_8.max()-data.mvar_8.min()) * 20
	data.mvar_9 = data.mvar_9 / float(data.mvar_9.max()-data.mvar_9.min()) * 20
	data.mvar_10 = data.mvar_10 / float(data.mvar_10.max()-data.mvar_10.min()) * 20
	data.mvar_11 = data.mvar_11 / float(data.mvar_11.max()-data.mvar_11.min()) * 20


	data.mvar_19 = data.mvar_19 / float(data.mvar_19.max()-data.mvar_19.min()) * 10
	data.mvar_20 = data.mvar_20 * 10

	data.mvar_23 = data.mvar_23 / float(data.mvar_23.max()-data.mvar_23.min()) * 20
	data.mvar_24 = data.mvar_24 / float(data.mvar_24.max()-data.mvar_24.min()) * 20
	data.mvar_25 = data.mvar_25 / float(data.mvar_25.max()-data.mvar_25.min()) * 20
	data.mvar_26 = data.mvar_26 / float(data.mvar_26.max()-data.mvar_26.min()) * 20
	data.mvar_28 = data.mvar_28 / float(data.mvar_28.max()-data.mvar_28.min()) * 20

	data.mvar_30 = data.mvar_30 / float(data.mvar_30.max()-data.mvar_30.min()) * 20

	return data




def createFeature(df):

	donation_CENTAUR_index = df['mvar_2'==df[['mvar_2','mvar_3','mvar_4','mvar_5','mvar_6']].idxmax(axis=1)].index
	donation_EBONY_index = df['mvar_3'==df[['mvar_2','mvar_3','mvar_4','mvar_5','mvar_6']].idxmax(axis=1)].index
	donation_TOKUGAWA_index = df['mvar_4'==df[['mvar_2','mvar_3','mvar_4','mvar_5','mvar_6']].idxmax(axis=1)].index
	donation_ODYSSEY_index = df['mvar_5'==df[['mvar_2','mvar_3','mvar_4','mvar_5','mvar_6']].idxmax(axis=1)].index
	donation_COSMOS_index = df['mvar_6'==df[['mvar_2','mvar_3','mvar_4','mvar_5','mvar_6']].idxmax(axis=1)].index


	social_CENTAUR_index = df['mvar_7'==df[['mvar_7','mvar_8','mvar_9','mvar_10','mvar_11']].idxmax(axis=1)].index
	social_EBONY_index = df['mvar_8'==df[['mvar_7','mvar_8','mvar_9','mvar_10','mvar_11']].idxmax(axis=1)].index
	social_TOKUGAWA_index = df['mvar_9'==df[['mvar_7','mvar_8','mvar_9','mvar_10','mvar_11']].idxmax(axis=1)].index
	social_ODYSSEY_index = df['mvar_11'==df[['mvar_7','mvar_8','mvar_9','mvar_10','mvar_11']].idxmax(axis=1)].index
	social_COSMOS_index = df['mvar_10'==df[['mvar_7','mvar_8','mvar_9','mvar_10','mvar_11']].idxmax(axis=1)].index

	rally_CENTAUR_index = df['mvar_23'==df[['mvar_23','mvar_24','mvar_25','mvar_26','mvar_28']].idxmax(axis=1)].index
	rally_EBONY_index = df['mvar_24'==df[['mvar_23','mvar_24','mvar_25','mvar_26','mvar_28']].idxmax(axis=1)].index
	rally_TOKUGAWA_index = df['mvar_25'==df[['mvar_23','mvar_24','mvar_25','mvar_26','mvar_28']].idxmax(axis=1)].index
	rally_ODYSSEY_index = df['mvar_26'==df[['mvar_23','mvar_24','mvar_25','mvar_26','mvar_28']].idxmax(axis=1)].index
	rally_COSMOS_index = df['mvar_28'==df[['mvar_23','mvar_24','mvar_25','mvar_26','mvar_28']].idxmax(axis=1)].index


	df['mvar_29'] = 0
	df['mvar_32'] = 0
	df['mvar_31'] = 0

	df['mvar_29'].loc[donation_CENTAUR_index] = 1
	df['mvar_32'].loc[social_CENTAUR_index] = 1
	df['mvar_31'].loc[rally_CENTAUR_index] = 1

	df['mvar_29'].loc[donation_EBONY_index] = 2
	df['mvar_32'].loc[social_EBONY_index] = 2
	df['mvar_31'].loc[rally_EBONY_index] = 2

	df['mvar_29'].loc[donation_TOKUGAWA_index] = 3
	df['mvar_32'].loc[social_TOKUGAWA_index] = 3
	df['mvar_31'].loc[rally_TOKUGAWA_index] = 3

	df['mvar_29'].loc[donation_ODYSSEY_index] = 4
	df['mvar_32'].loc[social_ODYSSEY_index] = 4
	df['mvar_31'].loc[rally_ODYSSEY_index] = 4

	df['mvar_29'].loc[donation_COSMOS_index] = 5
	df['mvar_32'].loc[social_COSMOS_index] = 5
	df['mvar_31'].loc[rally_COSMOS_index] = 5

	return df


def fillSalaries(df):

	salary = dict.fromkeys(df['mvar_12'].values,(0,0))

	for key in salary:
		salary[key] = (df[key==df['mvar_12']].index,df['mvar_30'].loc[df[key==df['mvar_12']].index].mean())

	for value in salary.values(): 
		s= df['mvar_30'].loc[value[0]]
		s.fillna(value=value[1],inplace=True)
		df['mvar_30'].loc[value[0]] = s

	return df












train_file = 'Training_Dataset.csv'
test_file = 'Leaderboard_Dataset.csv'

numeric_cols = ['mvar_2','mvar_3','mvar_4','mvar_5','mvar_6','mvar_7','mvar_8','mvar_9','mvar_10','mvar_11','mvar_19','mvar_20','mvar_23','mvar_24','mvar_25','mvar_26','mvar_28','mvar_29','mvar_31','mvar_32']


print 'Reading ...'
train = pd.read_csv( train_file )
test = pd.read_csv( test_file )


print 'Extracting ...'
train = extractFromFile(train)
test = extractFromFile(test)

train = fillSalaries(train)
test = fillSalaries(test)


print 'Normalizing ...'
train = normalizeData(train)
test = normalizeData(test)



t_train = train.actual_vote
x_train = train['Citizen ID']

x_test = test['Citizen ID']

#labelling target
le = LabelEncoder()
t_train = le.fit_transform(t_train)


train.drop('Citizen ID', axis=1, inplace=True)
train.drop('actual_vote', axis=1, inplace=True)
train.drop('mvar_27', axis=1, inplace=True)
train.drop('mvar_29', axis=1, inplace=True)
# train.drop('mvar_30', axis=1, inplace=True)
train.drop('mvar_13', axis=1, inplace=True)


test.drop('Citizen ID', axis=1, inplace=True)
test.drop('mvar_27', axis=1, inplace=True)
test.drop('mvar_29', axis=1, inplace=True)
# test.drop('mvar_30', axis=1, inplace=True)
test.drop('mvar_13', axis=1, inplace=True)


print 'Creating new features'
train = createFeature(train)
test = createFeature(test)


# numerical
f_num_train = train[ numeric_cols ].as_matrix()
f_num_test = test[ numeric_cols ].as_matrix()




#Fill missing values
print 'Imputation of numerical features by mean...' 
imp = Imputer(missing_values='NaN',strategy='mean',axis=0)
f_num_train = imp.fit_transform(f_num_train)
f_num_test = imp.transform(f_num_test)



cat_train = train.drop(numeric_cols, axis=1)
cat_test = test.drop(numeric_cols, axis=1)


f_cat_train = cat_train.T.to_dict().values()
f_cat_test = cat_test.T.to_dict().values()

	

# vectorize
print "Applying Vectorizer..."
vectorizer = DV( sparse = False )
vec_f_cat_train = vectorizer.fit_transform( f_cat_train )
vec_f_cat_test = vectorizer.transform( f_cat_test )


#Fill missing values 
print 'Imputation of categorical features by mode'
imp = Imputer(missing_values='NaN',strategy='most_frequent',axis=0)
vec_f_cat_train = imp.fit_transform(vec_f_cat_train)
vec_f_cat_test = imp.transform(vec_f_cat_test)


# complete x
f_train = np.hstack((f_num_train, vec_f_cat_train ))
f_test = np.hstack((f_num_test, vec_f_cat_test ))



print 'Learning classifier...' 
# clf = OneVsOneClassifier(LinearSVC(random_state=0))
clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0)
clf.fit(f_train,t_train)
y_test = clf.predict(f_test)



y_test = le.inverse_transform(y_test)



ofile = open('Aayush_IITKharagpur_31.csv','w')
writer = csv.writer(ofile)
for i in range(len(x_test)):
	writer.writerow([x_test[i],y_test[i]])
ofile.close()
