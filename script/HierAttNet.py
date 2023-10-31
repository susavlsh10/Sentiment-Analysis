import sys
import getopt
import os
import math
import operator
import numpy as np
import json

class HierAttNet:
  class TrainSplit:
    """Represents a set of training/testing data. self.train is a list of Examples, as is self.test. 
    """
    def __init__(self):
      self.train = []
      self.test = []
      
  class Example:
    """Represents a document with a label. klass is 'pos' or 'neg' by convention.
       words is a list of strings.
    """
    def __init__(self):
      self.klass = ''
      self.words = []


  def __init__(self):
    """network initialization"""
    self.numFolds = 10
    self.vocab = {}
    self.batch=1


  def dataConvert(self, words):
    """
      convert one document into numpy array of word index [1, #sent, #word]
    """
    max_word=50
    max_sent=20
    sent=np.zeros((1, max_sent, max_word), dtype='int32')
    sid=0
    wid=0
    slen=[]

    for word in words:        
      if word =='<eos>':
        sid+=1
        slen.append(wid)
        wid=0
      elif wid<max_word:
        sent[0,sid,wid]=self.vocab[word.lower()]
        wid+=1
      if sid>=max_sent:
        return sent,slen    
    
    if words[-1]!='<eos>':
      slen.append(wid)
           
    return sent,slen

  #############################################################################
  # TODO TODO TODO TODO TODO 
  # Implement the HierAttNet classifier 


  def classify(self, words):
    """ TODO
      implement the prediction function of HierAttNet  
      'words' is a list of words to classify. Return 'pos' or 'neg' classification.
    """
    label={'pos':0,'neg':1}
    x_data,x_lens=self.dataConvert(words)

    # Write code here

    return 'pos'

  
  def train(self, split, iterations):
    """
     * TODO
     * Train your model with dataset in the format of numpy array [x_data,y_data]
     * x_data: int numpy array, dimension=[#document, #sent, #word].
     * y_data: int numpy array, dimension: [#document].
     * x_lens: list, stores the list of sentence length in each document.
     * before training, you need to define HierAttNet sub-modules
     * in the HierAttNet class with a deep learning framework.
     * Returns nothing
    """
    label={'pos':0,'neg':1}
    eid=0
    y_data=[]
    x_lens=[]
    for example in split.train:      
      words = example.words  
      x,x_len=self.dataConvert(words)
      x_lens.append(x_len)
      y_data.append(label[example.klass])
      if eid==0:
        x_data=x.copy()
        eid+=1
      else:          
        x_data=np.concatenate((x_data,x), axis=0)
    y_data=np.array(y_data)  
    
    # Write code here
    pass

  # END TODO (Modify code beyond here with caution)
  #############################################################################
  
  
  def readFile(self, fileName):
    """
     * Code for reading a file.  you probably don't want to modify anything here, 
     * unless you don't like the way we segment files.
    """
    contents = []
    f = open(fileName)
    for line in f:
      contents.append(line)
    f.close()
    result = self.segmentWords(' <eos> '.join(contents))
    return result

  
  def segmentWords(self, s):
    """
     * Splits lines on whitespace for file reading
    """
    return s.split()

  
  def trainSplit(self, trainDir):
    """Takes in a trainDir, returns one TrainSplit with train set."""
    split = self.TrainSplit()
    posTrainFileNames = os.listdir('%s/pos/' % trainDir)
    negTrainFileNames = os.listdir('%s/neg/' % trainDir)
    for fileName in posTrainFileNames:
      example = self.Example()
      example.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
      example.klass = 'pos'
      split.train.append(example)
    for fileName in negTrainFileNames:
      example = self.Example()
      example.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
      example.klass = 'neg'
      split.train.append(example)
    return split


  def crossValidationSplits(self, trainDir):
    """Returns a lsit of TrainSplits corresponding to the cross validation splits."""
    splits = [] 
    posTrainFileNames = os.listdir('%s/pos/' % trainDir)
    negTrainFileNames = os.listdir('%s/neg/' % trainDir)
    #for fileName in trainFileNames:
    for fold in range(0, self.numFolds):
      split = self.TrainSplit()
      for fileName in posTrainFileNames:
        example = self.Example()
        example.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
        example.klass = 'pos'
        if fileName[2] == str(fold):
          split.test.append(example)
        else:
          split.train.append(example)
      for fileName in negTrainFileNames:
        example = self.Example()
        example.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
        example.klass = 'neg'
        if fileName[2] == str(fold):
          split.test.append(example)
        else:
          split.train.append(example)
      splits.append(split)
    return splits
  
  def buildVocab(self, trainDir):
    posTrainFileNames = os.listdir('%s/pos/' % trainDir)
    negTrainFileNames = os.listdir('%s/neg/' % trainDir)
    #for fileName in trainFileNames:
    for fileName in posTrainFileNames:
      example = self.Example()
      example.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
      
      for word in example.words:
        if word.lower() not in self.vocab:
          self.vocab[word.lower()]=len(self.vocab)

    for fileName in negTrainFileNames:
      example = self.Example()
      example.words = self.readFile('%s/neg/%s' % (trainDir, fileName))

      for word in example.words:
        if word.lower() not in self.vocab:
          self.vocab[word.lower()]=len(self.vocab)  
    return
   
def test10Fold(args):
  pt = HierAttNet()
  pt.batch=int(args[2])
  iterations = int(args[1])
  pt.buildVocab(args[0])
  splits = pt.crossValidationSplits(args[0])
  avgAccuracy = 0.0
  fold = 0
  for split in splits:
    classifier = HierAttNet()
    classifier.vocab=pt.vocab
    classifier.batch=pt.batch
    accuracy = 0.0
    classifier.train(split,iterations)
  
    for example in split.test:
      words = example.words
      guess = classifier.classify(words)
      if example.klass == guess:
        accuracy += 1.0

    accuracy = accuracy / len(split.test)
    avgAccuracy += accuracy
    print('[INFO]\tFold %d Accuracy: %f' % (fold, accuracy)) 
    fold += 1
  avgAccuracy = avgAccuracy / fold
  print('[INFO]\tAccuracy: %f' % avgAccuracy)
      
def main():
  
  (options, args) = getopt.getopt(sys.argv[1:], '')
  test10Fold(args)

if __name__ == "__main__":
    main()
