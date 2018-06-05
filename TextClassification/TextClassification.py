from .DataPreprocess import DataPreprocess
from .models import CNN, RNN
import numpy as np


class TextClassification():
    def __init__(self):
        pass

    def fit(self, x=None, y=None, model=None,
            method='CNN', epochs=10, batchsize=256,
            x_need_preprocess=True, y_need_preprocess=True,
            tokenizer=None, num_words=2000, maxlen=30,
            vec_size=128, output_shape=None, output_type='multiple'):
        self.tokenizer = tokenizer
        self.num_words = num_words
        self.maxlen = maxlen
        self.vec_size = vec_size

        # need process
        if x_need_preprocess:
            process = DataPreprocess()
            # cut texts
            x_cut = process.cut_texts(texts=x, need_cut=True, word_len=2, savepath=None)
            # use average length
            if maxlen is None:
                maxlen = int(np.array([len(x) for i in x_cut]).mean())
            # texts to sequence
            x_seq = process.text2seq(texts_cut=x_cut, tokenizer=tokenizer, tokenizer_savapah=None,
                                     num_words=num_words, maxlen=maxlen, batchsize=10000)
            x_seq = np.array(x_seq)
            x = x_seq
            self.num_words = num_words
            self.maxlen = maxlen
            self.tokenizer = process.tokenizer

        if y_need_preprocess:
            process = DataPreprocess()
            label_set = process.creat_label_set(y)
            labels = process.creat_labels(labels=y, label_set=label_set)
            labels = np.array(labels)
            output_shape = labels.shape[1]
            y = labels
            self.output_shape = output_shape
            self.label_set=label_set
        
        if model is None:
            if method == 'CNN':
                model = CNN(input_dim=num_words, input_length=maxlen,
                            vec_size=vec_size, output_shape=output_shape,
                            output_type=output_type)
            elif method == 'RNN':
                model = RNN(input_dim=num_words, input_length=maxlen,
                            vec_size=vec_size, output_shape=output_shape,
                            output_type=output_type)
        else:
            # maybe sklearn
            pass

        model.fit(x=x, y=y, epochs=epochs, batch_size=batchsize)
        self.model = model

    def predict(self, x=None, x_need_preprocess=True,
                tokenizer=None, num_words=None, maxlen=None):
        if x_need_preprocess:
            if tokenizer is not None:
                tokenizer = self.tokenizer
            if num_words is None:
                num_words = self.num_words
            if maxlen is None:
                maxlen = self.maxlen
            process = DataPreprocess()
            x_cut = process.cut_texts(texts=x, need_cut=True, word_len=2, savepath=None)
            x_seq = process.text2seq(texts_cut=x_cut, tokenizer=tokenizer,
                                     num_words=num_words, maxlen=maxlen, batchsize=10000)
            x = np.array(x_seq)

        model = self.model
        y = model.predict(x=x)
        return y

    def label2toptag(self, predictions, labelset):
        labels = []
        for prediction in predictions:
            label = labelset[prediction == prediction.max()]
            labels.append(label.tolist())
        return labels

    def label2half(self, predictions, labelset):
        labels = []
        for prediction in predictions:
            label = labelset[prediction > 0.5]
            labels.append(label.tolist())
        return labels
    
    def label2tag(self, predictions, labelset):
        labels1=self.label2toptag(predictions, labelset)
        labels2 = self.label2half(predictions, labelset)
        labels = []
        for i in range(len(predictions)):
            if len(labels2[i])==0:
                labels.append(labels1[i])
            else:
                labels.append(labels2[i])
        return labels
