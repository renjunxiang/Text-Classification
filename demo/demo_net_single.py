from TextClassification import TextClassification, DataPreprocess
from sklearn.model_selection import train_test_split
from TextClassification import load_data
import numpy as np

# load data
data = load_data(name='single')
x = data['evaluation']
y = [[i] for i in data['label']]

# split train and test
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

model = TextClassification()

# train
model.fit(x=X_train, y=y_train, method='CNN', model=None,
          x_need_preprocess=True, y_need_preprocess=True,
          epochs=1, batchsize=128, output_type='single')

# predict
label_set = model.label_set
y_predict = model.predict(x=X_test, x_need_preprocess=True)
y_predict_label = model.label2toptag(predictions=y_predict, labelset=label_set)
print(sum([y_predict_label[i] == y_test[i] for i in range(len(y_predict))]) / len(y_predict))
