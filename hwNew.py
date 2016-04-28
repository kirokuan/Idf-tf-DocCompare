from __future__ import division
from VectorSpace import VectorSpace
from pprint import pprint
from Parser import Parser
import util
import sys
from os import listdir
from os.path import isfile, join
from collections import OrderedDict,Counter
import math
import numpy as np
import nltk

class VectorSpaceNew(VectorSpace):
    idf=[]
    def __init__(self,distFn,weighFn,buildIdf, documents=[]):
        self.weightingFunc=weighFn
        self.distanceFunc=distFn
        super(VectorSpaceNew,self).__init__(documents)
        self.idf=buildIdf(self.documentVectors)
        self.documentVectors= [[a*b for a,b in zip(vector,self.idf)] for vector in self.documentVectors]

    def tokenizeAndSentize(self,wordString):
        vocabularyList = self.parser.tokenise(wordString)
        vocabularyList = self.parser.removeStopWords(vocabularyList)
        return vocabularyList

    def makeVector(self, wordString):
        """ do filter and weighting"""
        return self.weightingFunc(self.vectorKeywordIndex,Counter(self.tokenizeAndSentize(wordString)))

    def _search(self,searchList,queryVector):
        return [self.distanceFunc(queryVector, documentVector) for documentVector in self.documentVectors]

    def search(self,searchList):
        return self._search(searchList, self.buildQueryVector(searchList))

    def buildQueryVector(self,searchList):
        return [a*b for a,b in zip(super(VectorSpaceNew,self).buildQueryVector(searchList),self.idf)]

    def related(self,documentId):
        return [self.distanceFunc(self.documentVectors[documentId], documentVector) for documentVector in self.documentVectors]

    def FeedbackSearch(self,searchList,feedbackFn):
        result=self.search(searchList)
        queryVector = self.buildQueryVector(searchList)
        queryResult=OrderedDict(sorted( dict(zip(result,self.documentVectors)).items(), key=lambda t: -t[0]))
        newQueryVector=feedbackFn(queryVector,queryResult,self.vectorKeywordIndex)
        return self._search(searchList,newQueryVector)

def jaccard(vector1, vector2):
    x = zip(map(convert,vector1),map(convert,vector2))
    return [a and b for a,b in x].count(True) / [a or b for a,b in x].count(True)

def convert(a):
    return a>0

def tf(vectorKeywordIndex,totalFreq):
    total=sum(totalFreq.values())
    vector = [0 for i in range(len(vectorKeywordIndex))]
    for pos,freq in [ (vectorKeywordIndex[w],totalFreq[w]/total) for w in totalFreq.keys() if w in vectorKeywordIndex]:
        vector[pos] = freq;
    return vector

def BuildIdfDummy(documentsVectors):
    return [1]*len(documentsVectors[0])

def BuildIdf(documentsVectors):
    occ=reduce(lambda x,y: [a+b for a,b in zip(x,y) ], [[ 1 if i>0 else 0 for i in tf ] for tf in documentsVectors])
    return [math.log(len(documentsVectors)/o,10) for o in occ]

def feedback(qVector,docVectors,vectorKeywordIndex):
    topResult=docVectors.values()[0]
    keys=nltk.pos_tag([word[0] for word in vectorKeywordIndex.items() if topResult[word[1]]>0 ])
    convertToPos=[vectorKeywordIndex[word[0]] for word in keys if word[1] in ['NN','VB']]
    newVector=[1 if i in convertToPos else 0 for i in range(len(vectorKeywordIndex))]
    return [a+0.5*b*c for a,b,c in zip(qVector,topResult,newVector)]

def Q1(docuemnts,searchWords):
    vectorSpace= VectorSpaceNew(util.cosine,tf,BuildIdfDummy,documents)
    searchList=vectorSpace.search(searchWords)
    printResult(keyVector,searchList)

def Q2(docuemnts,searchWords):
    vectorSpace= VectorSpaceNew(jaccard,tf,BuildIdfDummy,documents)
    searchList=vectorSpace.search(searchWords)
    printResult(keyVector,searchList)

def Q3(docuemnts,searchWords):
    vectorSpace= VectorSpaceNew(util.cosine,tf,BuildIdf,documents)
    searchList=vectorSpace.search(searchWords)
    printResult(keyVector,searchList)

def Q4(docuemnts,searchWords):
    vectorSpace= VectorSpaceNew(jaccard,tf,BuildIdf,documents)
    searchList=vectorSpace.search(searchWords)
    printResult(keyVector,searchList)

def SecondQuestion(docuemnts,searchWords):
    vectorSpace= VectorSpaceNew(jaccard,tf,BuildIdf,documents)
    searchList=vectorSpace.FeedbackSearch(searchWords,feedback)
    printResult(keyVector,searchList)

def printResult(keyVector,result):
    searchDict  =OrderedDict(sorted( dict(zip(keyVector,result)).items(), key=lambda t: -t[1]))
    pprint(list(searchDict.items())[:10])
if __name__ == '__main__':
    mypath=sys.argv[1] #"/media/sf_shared/codes/documents/documents"
    if (len(sys.argv)>2):
        nltk.data.path.append(sys.argv[2]) #/media/sf_shared/nltk
    keyVector=[f for f in listdir(mypath) if isfile(join(mypath, f))]
    documents =[open(join(mypath, ff), "r").read() for ff in keyVector]
    query=["drill","wood","sharp"]
    print("Q1: Term Frequency (TF) Weighting + Cosine Similarity")
    Q1(documents,query)
    print("Q2: Term Frequency (TF) Weighting + Jaccard Similarity")
    Q2(documents,query)

    print("Q3: TF-IDF Weighting + Cosine Similarity")
    Q3(documents,query)
    print("Q4: TF-IDF Weighting + Jaccard Similarity")
    Q4(documents,query)

    print("feedback Query +TF-IDF Weighting + Jaccard Similarity")
    SecondQuestion(documents,query)

    #vectorSpace= VectorSpaceNew(jaccard,tf,BuildIdf,[dict(zip(keyVector,documents))['112716.product']])
    #pprint(vectorSpace.vectorKeywordIndex)
