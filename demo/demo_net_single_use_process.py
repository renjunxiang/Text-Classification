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

# deal train
# ----------------------------------------
process = DataPreprocess()

# cut texts
X_train_cut = process.cut_texts(texts=X_train, need_cut=True, word_len=2, savepath=None)

# texts to sequence
X_train_seq = process.text2seq(texts_cut=X_train_cut, tokenizer=None, tokenizer_savapah=None,
                               num_words=500, maxlen=20, batchsize=10000)
# list to array
X_train_seq = np.array(X_train_seq)

# get tokenizer
tokenizer = process.tokenizer

# label to one-hot
label_set = process.creat_label_set(y_train)
train_labels = process.creat_labels(labels=y_train, label_set=label_set)
train_labels = np.array(train_labels)

# deal test
# ----------------------------------------
process = DataPreprocess()

# cut texts
X_test_cut = process.cut_texts(texts=X_test, need_cut=True, word_len=2, savepath=None)

# texts to sequence
X_test_seq = process.text2seq(texts_cut=X_test_cut, tokenizer=tokenizer, tokenizer_savapah=None,
                              num_words=500, maxlen=20, batchsize=10000)
# list to array
X_test_seq = np.array(X_test_seq)

# label to one-hot
test_labels = process.creat_labels(labels=y_test, label_set=label_set)
test_labels = np.array(test_labels)

# creat model
# ----------------------------------------
model = TextClassification()

# train
model.fit(x=X_train_seq, y=train_labels, method='CNN', model=None,
          x_need_preprocess=False, y_need_preprocess=False, maxlen=20,
          epochs=1, batchsize=128,
          output_type='single', output_shape=train_labels.shape[1])

# predict
y_predict = model.predict(x=X_test_seq, x_need_preprocess=False)
y_predict_label = model.label2tag(predictions=y_predict, labelset=label_set)
print(sum([y_predict_label[i] == y_test[i] for i in range(len(y_predict))]) / len(y_predict))