import numpy as np
import pandas as pd
import jieba
import pandas as pd
import numpy as np
from sentence_2_sparse import sentence_transform

class LDA():
    def __init__(self,
                 dataset=None,
                 topic_num=5,
                 alpha=0.0002,
                 beta=0.02,
                 steps=500,
                 error=0.1):
        dataset[1]=sentence_transform(train_data=dataset[1], hash=False)
        self.dataset = dataset
        self.topic_num = topic_num
        self.alpha = alpha
        self.beta = beta
        self.steps = steps
        self.error = error

    def LDA(self):  # calulate similar matrix
        dataset = self.dataset
        topic_num = self.topic_num
        alpha = self.alpha
        beta = self.beta
        steps = self.steps
        error = self.error

        document_id = dataset[0]
        document_word = dataset[1]  # 取出稀疏矩阵
        word = document_word.columns
        # train_data=P*Q.T,step迭代次数,alpha学习率,beta正则系数
        # 1.计算梯度(R里面的每个元素逐个计算)
        # loss=sum(e[i,j]**2)=sum((R[i][j] - np.dot(P[i,:], Q[:,j]))**2)原始误差
        # loss_full=loss+0.5*beta*(P**2+Q**2)带上正则项
        # P[i][k]的偏导=2*e[i,j]*-Q[k,j]+beta*P[i][k]
        # 反向更新P:[i][k]=P[i][k]-alpha*P[i][k]的偏导
        # 计算整体误差,sum(loss_full)

        document_num, word_num = document_word.shape[0], document_word.shape[1]
        document_topic = np.random.rand(document_num, topic_num)  # 先生成随机数矩阵
        word_topic = np.random.rand(word_num, topic_num)
        topic_word = word_topic.T
        for step in range(steps):  # 迭代5000次
            n = 100 * ((step + 1) / steps)
            if n % 10 == 0:
                print('迭代次数=%d' % (step + 1), " 完成%d" % (n), '%, ', sep='')
            e = 0  # 先计算误差,小于阈值就停止
            not_zero = 0
            for i in range(document_num):
                for j in range(word_num):
                    if document_word.iloc[i, j] > 0:
                        not_zero += 1
                        e = e + pow(document_word.iloc[i, j] - np.dot(document_topic[i, :], topic_word[:, j]),
                                    2)  # 当R[i][j]位置不等于0,计算误差
                        for k in range(topic_num):
                            e = e + (beta / 2) * (pow(document_topic[i][k], 2) + pow(topic_word[k][j], 2))  # 加上正则项惩罚
            e = e / max(not_zero, 1)
            if e < error:  # loss function < 0.001
                break

            for i in range(document_num):  # row
                # print('i=',i)
                for j in range(word_num):  # columns
                    # print('j=',j)
                    eij = document_word.iloc[i, j] - np.dot(document_topic[i, :], topic_word[:, j])  # 计算梯度并反向更新
                    for k in range(topic_num):
                        # print('k=',k)
                        document_topic[i][k] = document_topic[i][k] + alpha * (
                            2 * eij * topic_word[k][j] - beta * document_topic[i][k])  # 带正则项的偏导*alpha
                        topic_word[k][j] = topic_word[k][j] + alpha * (
                            2 * eij * document_topic[i][k] - beta * topic_word[k][j])

        document_topic = pd.DataFrame(document_topic)
        topic_word = pd.DataFrame(topic_word)
        document_topic.index = document_id
        topic_word.columns = word
        return document_topic, topic_word, e

    def document_recommend_topic(self, document_id=None, num_topic=2, num_word=5):
        document_topic, topic_word, e = self.LDA()
        num_word=min(num_word,topic_word.shape[1])
        document_topic.columns = ['topic ' + str(i) for i in range(1, self.topic_num + 1)]
        document_topic = document_topic.agg(lambda x: x.sort_values(ascending=False).index, axis=1).iloc[:, 0:num_topic]
        topic_word = topic_word.agg(lambda x: x.sort_values(ascending=False).index, axis=1).iloc[:, 0:num_word]
        if document_id != None:
            document_topic = document_topic.loc[[document_id]]
        document_topic.columns = ['top ' + str(i) for i in range(1, num_topic + 1)]
        topic_word.columns = ['top ' + str(i) for i in range(1, num_word + 1)]
        topic_word.index = ['topic ' + str(i) for i in range(1, self.topic_num + 1)]
        return document_topic, topic_word


if __name__ == '__main__':
    dataset = [['document' + str(i) for i in range(1, 11)],
               ['全面从严治党，是十九大报告的重要内容之一。十九大闭幕不久，习近平总书记在十九届中央纪委二次全会上发表重要讲话',
                '根据国际公约和国际法，对沉船进行打捞也要听取船东的意见。打捞工作也面临着很大的风险和困难，如残留凝析油可能再次燃爆',
                '下午召开的北京市第十四届人大常委会第四十四次会议决定任命殷勇为北京市副市长',
                '由中国航天科技集团有限公司所属中国运载火箭技术研究院抓总研制的长征十一号固体运载火箭“一箭六星”发射任务圆满成功',
                '直到2016年7月份，谢某以性格不合为由，向卢女士提出分手，并要求喝分手酒，可谁知，这醉翁之意不在酒哪',
                '湖北男子吴锐在其居住的湖南长沙犯下了一桩大案：跟踪一名开玛莎拉蒂女子',
                '甚而至于得罪了名人或名教授',
                '判决书显示，现年不到30岁的吴锐出生于湖北省天门市，住湖南省长沙县',
                '张某报警后，公安机关在侯某家门前将李某抢劫来的车辆前后别住。李某见状开始倒车',
                '被打女童来自哪里？打人者是谁？1月17日晚，澎湃新闻联系上女童曾某的母亲']]
    model = LDA(dataset=dataset, steps=200)
    document_topic, topic_word = model.document_recommend_topic(num_topic=2, num_word=8)
    print('document_recommend_topic\n', document_topic)
    print('topic_recommend_word\n', topic_word)
