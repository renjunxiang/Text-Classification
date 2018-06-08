from TextClassification import TextClassification, DataPreprocess
from sklearn.model_selection import train_test_split
from TextClassification import load_data
import numpy as np

########################################################################################
# data process

# load data
# -----------------------------------
data = load_data(name='single')
x = data['evaluation']
y = [[i] for i in data['label']]

process = DataPreprocess()
# cut texts
x_cut = process.cut_texts(texts=x, need_cut=True, word_len=2, savepath=None)
# texts to sequence
x_seq = process.text2seq(texts_cut=x_cut, tokenizer=None, tokenizer_savapah=None,
                         num_words=500, maxlen=20, batchsize=10000)
# list to array
x_seq = np.array(x_seq)

# get tokenizer
tokenizer = process.tokenizer
########################################################################################

# single target

# load data
# -----------------------------------
data = load_data(name='single')
x = data['evaluation']
y = [[i] for i in data['label']]

# train and predict
# ------------------------------------
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

model = TextClassification()
model.fit(x=X_train, y=y_train, method='CNN', model=None,
          x_need_preprocess=True, y_need_preprocess=True,
          epochs=10, batchsize=128, output_type='single')
label_set = model.label_set
y_predict = model.predict(x=X_test, x_need_preprocess=True)
y_predict_label = model.label2toptag(predictions=y_predict, labelset=label_set)
print(sum([y_predict_label[i] == y_test[i] for i in range(len(y_predict))]) / len(y_predict))
########################################################################################

# multiple target

# load data
# -----------------------------------
data = load_data(name='multiple')
x = [i['fact'] for i in data]
y = [i['accusation'] for i in data]

# train and predict
# ------------------------------------
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

model = TextClassification()
model.fit(x=X_train, y=y_train, method='CNN', model=None,
          x_need_preprocess=True, y_need_preprocess=True,
          epochs=10, batchsize=128, output_type='multiple')
label_set = model.label_set
y_predict = model.predict(x=X_test, x_need_preprocess=True)
y_predict_label = model.label2tag(predictions=y_predict, labelset=label_set)
print(sum([y_predict_label[i] == y_test[i] for i in range(len(y_predict))]) / len(y_predict))
