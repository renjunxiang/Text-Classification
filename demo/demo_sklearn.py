from TextClassification import TextClassification, DataPreprocess
from sklearn.model_selection import train_test_split
from TextClassification import load_data
import numpy as np

# load data
# -----------------------------------
data = load_data(name='single')
x = data['evaluation']
y = data['label']

# data process
# -----------------------------------
process = DataPreprocess()
# cut texts
x_cut = process.cut_texts(texts=x, need_cut=True, word_len=2, savepath=None)

# texts to word vector
x_word_vec = process.text2vec(texts_cut=x, sg=1, size=20, window=5, min_count=1)
# texts vector
x_vec = np.array([sum(i) / len(i) for i in x_word_vec])

# train model
# ------------------------------------
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state=1)

model = TextClassification()

# # without process
# model.fit(x=x_vec,
#           y=y,
#           x_need_preprocess=False,
#           y_need_preprocess=False,
#           method='SVM', output_type='single')

# use process
model.fit(x=X_train,
          y=y_train,
          x_need_preprocess=True,
          y_need_preprocess=False,
          method='SVM', output_type='single')

y_predict = model.predict(x=X_test, x_need_preprocess=True)
# score 0.8331
print(sum(y_predict == np.array(y_test)) / len(y_predict))
