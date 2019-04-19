import sys,re,os,getopt
import glob,random,time,string
import numpy as np
from collections import Counter

class CommandLine:

    def __init__(self):
        opts,args=getopt.getopt(sys.argv[1:],'')
        self.trainFile=args[0] #get train file name
        self.testFile=args[1] #get test file name

    def getFile(self):
        return self.trainFile,self.testFile

class model:

    def __init__(self,trainFile,testFile):
        self.trainFile=trainFile  #train file path
        self.testFile=testFile #test file path

        self.context=[] #the context of train data after preprocess(tokenization)

        self.unigram=[] #store the count of context by unigram
        self.bigram=[] #store the count of context by bigram

        self.wordCount=0 # initialise total wordCount=0

        self.uniProbability={} #store the probability of unigram
        self.biProbability={} #store the probability of bigram
        self.biSmoothProb={} #store the probability of bigram using add1 smoothing


    # add <s> , <s> to each sentence, and tokenization
    def preprocess(self):
        p=re.compile(r"<\\s>|[\w'-]+|<s>")
        pForTest=re.compile(r'[a-z]+')
        context=''
        f=open(self.trainFile)
        for line in f:
            line=line.lower()
            line=' <s> '+line #add start and end symbol
            line=line+' <\s>'
            context=context+line
        self.context=p.findall(context)
        print('tokenization done')

    #count by unigram, and store the probability in dictionary
    def uniToken(self):
        self.unigram=Counter(self.context)

        self.wordCount=len(self.context) #get the lenth of text

        #get the whole probability of unigram set
        for word in self.unigram:
            self.uniProbability[word]=self.unigram[word]/float(self.wordCount)

    #count by bigram, and store the probability of bigram and add1 smoothing in dictionary
    def biToken(self):
        V=len(self.unigram)
        bigram=list(zip(*[self.context[i:] for i in range(2)]))
        self.bigram=Counter(bigram)

        #get the whole probability of bigram set
        for item in self.bigram:
            self.biProbability[item]=self.bigram[item]/float(self.unigram[item[0]]) #probability of bigram
            self.biSmoothProb[item]=(self.bigram[item]+1)/float(self.unigram[item[0]]+V) #probability of add1 smoothing

    #get the probability of answer using unigram
    def testUni(self,answer):
        firstWord=self.uniProbability[answer[0]]
        secondWord=self.uniProbability[answer[1]]
        if(firstWord>secondWord):
            return answer[0]
        elif(firstWord<secondWord):
            return answer[1]
        else:
            return 'incorrect' # incorrect

    #get the probability of answer using bigram
    def testBi(self,sentence,answer):
        result=1 #initialise probability as 1

        mySentence=sentence.copy()
        location=mySentence.index('____')
        wordBefore=mySentence[location-1]
        wordAfter=mySentence[location+1]

        bigram=[(wordBefore,answer),(answer,wordAfter)]
        for item in bigram:
            if(item not in self.bigram):
                result=0
                break
            else:
                result=result*self.biProbability[item]
        return result

    #get the smooth probability of answer using bigram
    def testSmooth(self,sentence,answer):
        result=1 #initialise probability as 1total

        mySentence=sentence.copy()
        location=mySentence.index('____')
        wordBefore=mySentence[location-1]
        wordAfter=mySentence[location+1]

        bigram=[(wordBefore,answer),(answer,wordAfter)]
        V=len(self.unigram)
        for item in bigram:
            if(item not in self.bigram):
                probability=1/(self.unigram[item[0]]+V)
                result*=probability
            else:
                result*=self.biSmoothProb[item]
        return result

    #choose the answer
    def calculate(self):
        resultUni=[] #get the answer by unigram
        resultBi=[] #get the answer by bigram
        resultSmooth=[] #get the answer by add1 smoothing

        p=re.compile(r"<\\s>|[\w'-]+|<s>") #tokenization
        pForTest=re.compile(r'[a-z]+') #tokenization for answer
        with open(self.testFile) as f:
            for line in f:
                line=line.lower()
                cutQuestion=line.split(':') #split the sentence,
                                            #first part is sentence with blank,second part is candidate word

                sentence='<s> '+cutQuestion[0] #add the start end symbol
                sentence+=' <\s>'

                answer=cutQuestion[1] #get the candidate word

                sentenceList=p.findall(sentence) #tokenization for sentence
                answerList=pForTest.findall(answer) #tokenization for candidate word

                #================get the answer using unigram language model==================================
                result=self.testUni(answerList)
                resultUni.append(result)

                #================get the answer using bigram language model====================================
                prob1=self.testBi(sentenceList,answerList[0]) #try first candidate word
                prob2=self.testBi(sentenceList,answerList[1]) #try second candidate word
                if(prob1>prob2):
                    resultBi.append(answerList[0])
                elif(prob1<prob2):
                    resultBi.append(answerList[1])
                else:
                    if(prob1==0):
                        resultBi.append('0') #incorrect
                    else:
                        resultBi.append('half correct')

                #================get the answer using add1 smoothing language model===============================
                prob1=self.testSmooth(sentenceList,answerList[0]) #try with first candidate word
                prob2=self.testSmooth(sentenceList,answerList[1]) #try with second candidate word
                if(prob1>prob2):
                    resultSmooth.append(answerList[0])
                elif(prob1<prob2):
                    resultSmooth.append(answerList[1])
                else:
                    if(prob1==0):
                        resultSmooth.append('0') #incorrect
                    else:
                        resultSmooth.append('half correct')

        #print the answer
        print('result of using unigram language model:')
        print(resultUni,'\n')
        print('result of using bigram language model:')
        print(resultBi,'\n')
        print('result of using add1 smoothing language model:')
        print(resultSmooth)

    # start set language ,model and test
    def implement(self):
        self.preprocess()
        self.uniToken()
        self.biToken()

        self.calculate()

    def getToken(self):
        return self.unigram,self.bigram

#===========main===================================================================
if __name__=='__main__':
    timeStart=time.time()

    config=CommandLine()
    trainFile,testFile=config.getFile()

    p=model(trainFile,testFile)
    p.implement()

    timeEnd=time.time()
    print('total duration: ',timeEnd-timeStart,'s')
