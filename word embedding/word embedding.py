import numpy
import keras
from keras.datasets import imdb
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense
from keras.optimizers import RMSprop
from keras import losses
from keras import metrics
from keras.preprocessing.sequence import pad_sequences
# import matplotlib.pyplot as plt
import os
'''
加载数据
num_words=1000:表示只保留训练数据中最常出现的10000个单词
train_data和test_data是评论列表，数据里的单词序列已被转换为整数序列，每个整数代表字典中的特定单词
train_labels和test_labels是0和1的列表，其中0表示‘负面评论’,1表示‘正面评论’
'''
class imdb():
    def __init__(self):
        self.training_samples=3000
        self.validation_samples=12000
        self.max_words=10000
        self.max_len = 100
        self.embedding_dim = 50

    def read_data(self):
        dataset = []
        labels = []
        def eachFile(filepath,lab):
            pathDir = os.listdir(filepath)
            for allDir in pathDir:
                with open(filepath + '/' + allDir) as f:
                    try:
                        dataset.append(f.read())
                        labels.append(lab)
                    except UnicodeDecodeError:
                        pass
        filepath = ['C:/Users/伊雅/PycharmProjects/untitled/venv/share/nlp/aclImdb/aclImdb/test/neg','C:/Users/伊雅/PycharmProjects/untitled/venv/share/nlp/aclImdb/aclImdb/test/pos']
        eachFile(filepath[0],0)
        eachFile(filepath[1], 1)
        return dataset,labels

    def tokennize_data(self):
        # 导入数据集
        dataset,labels=self.read_data()
        # 构造一个分词器，num_words默认是None处理所有字词，但是如果设置成一个整数，那么最后返回的是最常见的、出现频率最高的num_words个字词。
        tokenizer = Tokenizer(num_words=self.max_words)
        # 类方法之一，texts为将要训练的文本列表
        tokenizer.fit_on_texts(texts=dataset)
        # 将texts所有句子所有单词变为word_index对应的数字
        sequences = tokenizer.texts_to_sequences(texts=dataset)
        # 将所有的单词从大到小排列list，构建一个key为单词，value为单词对应在list的位置
        word_index = tokenizer.word_index
        print(word_index)
        print('Found %s unique tokens.' % len(word_index))
        # 序列填充，如果向量长度超过maxlen则保留前maxlen，如果没有maxlen这么长，就用0填充
        data = pad_sequences(sequences, maxlen=self.max_len)
        # asarray和array不同点是当labels发生变化时，asarray(labels)跟着发生变化，但array(labels)不变
        labels = np.asarray(labels)
        print('Shape of data tensor:', data.shape)
        print('Shape of label tensor:', labels.shape)
        indices = np.arange(data.shape[0])
        # 将indices的顺序打乱
        np.random.shuffle(indices)
        data = data[indices]
        labels = labels[indices]
        # 生成训练样本和测试样本
        x_train = data[:self.training_samples]
        y_train = labels[:self.training_samples]
        x_val = data[self.training_samples: self.training_samples + self.validation_samples]
        y_val = labels[self.training_samples: self.training_samples + self.validation_samples]
        return x_train, y_train, x_val, y_val, word_index

    def parse_word_embedding(self,word_index):
        glove_dir = 'D:/BaiduYunDownload/glove.6B.50d.txt'
        embeddings_index = {}
        f = open(glove_dir, encoding='UTF-8')
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()
        print('Found %s word vectors.' % len(embeddings_index))
        embedding_matrix = np.zeros((self.max_words, self.embedding_dim))
        for word, i in word_index.items():
            if i < self.max_words:
                embedding_vector = embeddings_index.get(word)
                if embedding_vector is not None:
                    embedding_matrix[i] = embedding_vector
        return embedding_matrix



    def model(self):
        # 创建模型，序贯模型
        model = Sequential()
        model.add(Embedding(self.max_words, self.embedding_dim, input_length=self.max_len))
        #output= shape=(?,100,50).input=shape=(?,100)
        print(model.layers[0].get_weights())
        model.add(Flatten())
        # input=shape=(?,100,50),output=(?,5000)
        #  (32, 10, 64) #32句话，每句话10个单词，每个单词有64维的词向量
        print(model.layers[0].get_weights())
        # Dense表示连接层，32是units，是输出纬度，activation表示激活函数，relu表示平均函数，sigmoid是正交矩阵
        model.add(Dense(32, activation='relu'))
        # input=shape=(?,5000),output=(?,32)
        # model.add(Dense(32, input_shape=(16,)))
        # 模型输入shape为 (*, 16)，输出shape (*, 32)
        model.add(Dense(1, activation='sigmoid'))
        # input=shape=(?,32),output=(?,1)
        model.summary()
        # 将GLOVE加载到模型中
        x_train, y_train, x_val, y_val, word_index = self.tokennize_data()
        # 嵌入层：
        # input_dim： 字典长度，即输入数据最大下标+1
        # output_dim：全连接嵌入的维度
        # input_length：当输入序列的长度固定时，该值为其长度。如果要在该层后接Flatten层，然后接Dense层，则必须指定该参数，否则Dense层的输出维度无法自动推断。
        embedding_matrix = self.parse_word_embedding(word_index)
        print(embedding_matrix)
        model.layers[0].set_weights([embedding_matrix])
        # 权重矩阵固定，不在需要自我训练
        model.layers[0].trainable = False
#模型的编译，优化器，可以是现成的优化器如rmsprop，binary_crossentropy是交叉熵损失函数，一般用于二分类，metrics=['acc']评估模型在训练和测试时的网络性能的指标，
        model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])
        # epochs表示循环次数，batch_size表示每一次循环使用多少数据量
        history = model.fit(x_train, y_train,epochs=10,batch_size=512,validation_data=(x_val, y_val))
        result=model.evaluate(x_val, y_val, verbose=1)
        print('test score is',result[0])
        print('test accuracy is ', result[1])
        history_dict = history.history
        # acc = history_dict['acc']
        # val_acc = history_dict['val_acc']
        # loss = history_dict['loss']
        # val_loss = history_dict['val_loss']
        print(history.history)
        return history_dict

    def plot(self):
        history_dict=self.model()
        # acc = history_dict['acc']
        # val_acc = history_dict['val_acc']
        # loss = history_dict['loss']
        # val_loss = history_dict['val_loss']
        # epochs = range(1, len(acc) + 1)
        # plt.plot(epochs, loss, 'bo', label='Training loss')
        # plt.plot(epochs, acc, 'b', label='Training acc')
        # plt.plot(epochs, val_loss, 'ro', label='Validation loss')
        # plt.plot(epochs, val_acc, 'r', label='Validation acc')
        # plt.title('Loss and Acc')
        # plt.xlabel('Epochs')
        # plt.ylabel('Loss and Acc')
        # plt.legend()
        # plt.show()

if __name__=="__main__":
    imdbmodel=imdb()
    imdbmodel.plot()

# 绘制loss-acc图

# epochs = range(1, len(acc) + 1)
# "bo" is for "blue dot"
# plt.plot(epochs, loss, 'bo', label='Training loss')
# plt.plot(epochs, acc, 'ro', label='Training acc')
# # b is for "solid blue line"
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
# plt.plot(epochs, val_acc, 'r', label='Validation acc')
# plt.title('Loss and Acc')
# plt.xlabel('Epochs')
# plt.ylabel('Loss and Acc')
# plt.legend()
#
# plt.show()
