import pandas as pd

def fre_max(li):
	fre  = dict()
	for cla in li:
		if cla not in fre:
			fre[cla] = 1
		else:
			fre[cla] += 1

	result = []
	for key, val in list(fre.items()):
		result.append((val, key))
		result.sort(reverse=True)
	_, key = result[0]

	return key

# load data
data = pd.read_csv('20230319_6_train_good/news_out_shot_class.csv')

# set threshold to norm predict ver.1
predict_norm = []
filiter = 12
for i in range(filiter):
	predict_norm.append(data.iloc[i,1])
for i in range(filiter,data.shape[0]-filiter):
	predict_norm.append(fre_max(data.iloc[i-filiter:i+filiter,1]))
for i in range(filiter):
	predict_norm.append(data.iloc[i+(data.shape[0]-filiter),1])

# class to shot change frame
change_real = []
change_predict = []
change_predict_norm = []
temp_real = -1
temp_predict = -1
temp_predict_norm = -1

for i in range(data.shape[0]):
	# real
	if(temp_real==-1): change_real.append(0)
	elif(temp_real!=data.iloc[i,0]): change_real.append(3)
	else: change_real.append(0)
	# predict
	if(temp_predict==-1): change_predict.append(0)
	elif(temp_predict!=data.iloc[i,1]): change_predict.append(1)
	else: change_predict.append(0)
	# predict_norm
	if(temp_predict_norm==-1): change_predict_norm.append(0)
	elif(temp_predict_norm!=predict_norm[i]): change_predict_norm.append(2)
	else: change_predict_norm.append(0)

	temp_real = data.iloc[i,0]
	temp_predict = data.iloc[i,1]
	temp_predict_norm = predict_norm[i]

# save to csv
data['predict_norm'] = predict_norm
data['change_real'] = change_real
data['change_predict'] = change_predict
data['change_predict_norm'] = change_predict_norm
order = ['real', 'predict', 'predict_norm', 'change_real', 'change_predict', 'change_predict_norm']
data = data[order]
data.to_csv('news_out_shot_class.csv', index=False)