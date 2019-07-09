from  numpy import *
import requests
import  re
import random
def read_txt():
    file = 'bayes.txt'
    with open(file, encoding='utf-8') as f:
        con = f.readlines()
    return con


def create_word_list(con):
    reg = re.compile(r'\W*')
    dataset = []
    wordlist = set([])
    txt_class = []
    for i in con[1:]:
        dataset.append(re.split(reg, i)[1:])
        wordlist = wordlist | set(re.split(reg, i)[1:])
        if re.split(reg, i)[0]=='ham':#有用的邮件
            txt_class.append(1)
        else:#垃圾邮件
            txt_class.append(0)
    print(dataset[:3])
    wordlist=[i for i in wordlist if len(i)>2]
    return dataset,wordlist,txt_class
def words_vec(txt,wordlist):
    returnvec=[0]*len(wordlist)
    for word in txt:
        if word in wordlist:
            returnvec[list(wordlist).index(word)] = 1#词集模型
            returnvec[list(wordlist).index(word)]=1+returnvec[list(wordlist).index(word)]#词袋模型
    return returnvec

def classify(vec,p0vec,p1vec,pclass1):
    p1=sum(vec*p1vec)+log(pclass1)
    p0=sum(vec*p0vec)+log(1.0-pclass1)
    if p1>p0:
        return 1
    else:
        return 0
def train(trainmatrix,traincategory):
    numword=len(trainmatrix[0])#列/10860
    numtrain=len(trainmatrix)#行/4574
    pa=sum(traincategory)/float(len(trainmatrix))#‘1’的占比
    p0num,p1num=ones(numword),ones(numword)
    p0dem,p1dem=2.0,2.0
    for i in range(numtrain):
        if traincategory[i]==1:
            p1num+=trainmatrix[i]
            p1dem+=sum(trainmatrix[i])
        else:
            p0num+=trainmatrix[i]
            p0dem+=sum(trainmatrix[i])
    p1vect= log(p1num / p1dem)
    p0vect = log(p0num / p0dem)
    return p1vect,p0vect,pa
def mul(num):
    totalerror = 0
    for times in range(num):
        trainset = [i for i in range(len(txt_class))]
        testset = []
        testclass = []
        for i in range(1000):
            number = int(random.choice(trainset))
            testset.append(words_vec(dataset[number], wordlist))
            testclass.append(txt_class[number])
            trainset.remove(number)
        trainMat = [];
        trainclass = []
        for i in trainset:
            trainMat.append(words_vec(dataset[i], wordlist))
            trainclass.append(txt_class[i])
        p1vect, p0vect, pa = train(trainMat, trainclass)
        error=0
        for i in range(len(testset)):
            if classify(array(testset[i]), p0vect, p1vect, pa) != testclass[i]:
                error += 1
        print('the accurate is', 1 - error / float(len(testset)))
        totalerror+=error / float(len(testset))
    print('after %d times the accurate of bayes is %f'%(num,1-float(totalerror)/num))


if __name__ == '__main__':
    con = read_txt()
    dataset, wordlist, txt_class = create_word_list(con)
    num=15
    mul(num)

