from .DataPreprocess import DataPreprocess
from .models import CNN, RNN, SklearnClf
import numpy as np


class TextClassification():
    def __init__(self):
        pass

    def fit(self, x=None, y=None, model=None,
            method='CNN', epochs=10, batchsize=256,
            x_need_preprocess=False, y_need_preprocess=False,
            tokenizer=None, num_words=2000, maxlen=30,
            vec_size=128, output_shape=None, output_type='multiple',
            **sklearn_param):
        '''
        Process texts and labels, creat model to fit
        :param x: Data feature
        :param y: Data label
        :param model: Model to fit
        :param method: Model type
        :param epochs: Number of epochs to train the model
        :param batchsize: Size of minibatch
        :param x_need_preprocess: TRUE will load DataPreprocess to process x to sequence
        :param y_need_preprocess: TRUE will load DataPreprocess to process y to one-hot
        :param tokenizer: Keras tokenizer model
        :param num_words: The maximum number of words to keep in tokenizer
        :param maxlen: The number of words to keep in sentence
        :param vec_size: Word vector size
        :param output_shape: Num of labels
        :param output_type: Single or multiple, sklearn only support single
        :param sklearn_param: Param for sklearn model
        :return: None
        '''
        self.tokenizer = tokenizer
        self.num_words = num_words
        self.maxlen = maxlen
        self.vec_size = vec_size
        self.method = method

        # need process
        if method in ['CNN', 'RNN']:
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
                # list to array
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
                self.label_set = label_set

            if model is None:
                if method == 'CNN':
                    model = CNN(input_dim=num_words, input_length=maxlen,
                                vec_size=vec_size, output_shape=output_shape,
                                output_type=output_type)
                elif method == 'RNN':
                    model = RNN(input_dim=num_words, input_length=maxlen,
                                vec_size=vec_size, output_shape=output_shape,
                                output_type=output_type)
            model.fit(x=x, y=y, epochs=epochs, batch_size=batchsize)

        elif method in ['SVM', 'Logistic']:
            if output_type != 'single':
                raise ValueError('sklearn output_type should be single')
            else:
                if x_need_preprocess:
                    process = DataPreprocess()
                    # cut texts
                    x_cut = process.cut_texts(texts=x, need_cut=True, word_len=2, savepath=None)
                    x_seq_vec = process.text2vec(texts_cut=x, sg=1, size=128, window=5, min_count=1)
                    x_vec = np.array([sum(i) for i in x_seq_vec])
                    x = x_vec
                    self.model_word2vec = process.model_word2vec

                model = SklearnClf(method=method, **sklearn_param)
                model.fit(X=x, y=y)

        self.model = model

    def predict(self, x=None, x_need_preprocess=True, model=None,
                tokenizer=None, num_words=None, maxlen=None,
                model_word2vec=None):
        '''

        :param x:
        :param x_need_preprocess:
        :param model:
        :param tokenizer:
        :param num_words:
        :param maxlen:
        :param model_word2vec:
        :return:
        '''
        method = self.method
        if method in ['CNN', 'RNN']:
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
        elif method in ['SVM', 'Logistic']:
            if x_need_preprocess:
                if model_word2vec is None:
                    model_word2vec = self.model_word2vec
                process = DataPreprocess()
                # cut texts
                x_cut = process.cut_texts(texts=x, need_cut=True, word_len=2, savepath=None)
                x_seq_vec = process.text2vec(texts_cut=x, model_word2vec=model_word2vec)
                x_vec = np.array([sum(i) for i in x_seq_vec])
                x = x_vec

        if model is None:
            model = self.model

        y = model.predict(x)
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
        labels1 = self.label2toptag(predictions, labelset)
        labels2 = self.label2half(predictions, labelset)
        labels = []
        for i in range(len(predictions)):
            if len(labels2[i]) == 0:
                labels.append(labels1[i])
            else:
                labels.append(labels2[i])
        return labels
