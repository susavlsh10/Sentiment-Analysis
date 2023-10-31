import sys
import getopt
import os
import math
import operator
import numpy as np

class Perceptron:
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
    """Perceptron initialization"""
    self.numFolds = 10
    
    self.weights = {}
    self.weight_avg = {}
    
    self.bias = 0
    self.bias_avg=0
    
    self.sign = {"pos": 1, "neg":-1}
    self.count = 0
    
    self.train_complete = False
  #############################################################################
  # TODO TODO TODO TODO TODO 
  # Implement the Perceptron classifier 

  def classify(self, words):
    """ TODO
      'words' is a list of words to classify. Return 'pos' or 'neg' classification.
    """

    # Write code here
    
    if not self.train_complete:
        for i in self.weight_avg.keys():
            self.weight_avg[i] = self.weights[i] - (self.weight_avg[i])/self.count
        self.train_complete = True
    
    test_x = {}
    # generate test features
    for w in words:
        if w not in test_x:
            test_x[w] = 1
        else:
            test_x[w] += 1
        if w not in self.weights:
            self.weight_avg[w] = 0
            
    #forward pass
    score = 0
    for x in test_x.keys():
        score = score + self.weight_avg[x]*test_x[x]
    score = score #+ int(self.bias)    
    
    if score >= 0:
        #print('pos')#, score)
        return 'pos'
        
    elif score < 0:
        #print('neg')
        return 'neg'
    del score
    
  

  def addExample(self, klass, words):
    """
     * TODO
     * Train your model on an example document with label klass ('pos' or 'neg') and
     * words, a list of strings.
     * You should store whatever data structures you use for your classifier 
     * in the Perceptron class.
     * Returns nothing
    """

    # Write code here
    training_features = {}
    y = self.sign[klass]
    self.count = self.count + 1
    
    
    #count the number of words
    for w in words: 
        if w not in training_features:
            training_features[w] = 1
        else:
            training_features[w] = training_features[w] + 1
        if w not in self.weights:
            self.weights[w] = 0
            self.weight_avg[w] = 0
    
    #forward pass
    score = 0    
    for x in training_features.keys():
        score = score + self.weights[x]*training_features[x]

    score = score + self.bias

    if score >= 0:
        sign = 1
    else:
        sign = -1
    #print('Score = ', score, 'guess = ', sign, 'true = ', klass)
    
    update = 0
    update_avg = 0
    bias_update = 0
    bias_update_avg = 0
    if sign != y:
        for w in training_features.keys():
            update = y*training_features[w]
            update_avg = self.count*y*training_features[w]

            self.weights[w] += update
            self.weight_avg[w] += update_avg

        bias_update = y
        bias_update_avg = self.count*y 

        self.bias += bias_update
        self.bias_avg = bias_update_avg
        
    del training_features
    
  
  def train(self, split, iterations):
      """
      * TODO 
      * iterates through data examples
      * TODO 
      * use weight averages instead of final iteration weights
      """
      #weight_averages = {}
      for i in range (0, iterations):
          for example in split.train:
              words = example.words
              self.addExample(example.klass, words)
              
          print('iteration :', i)
  

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
    result = self.segmentWords('\n'.join(contents)) 
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
  

def test10Fold(args):
  pt = Perceptron()
  
  iterations = int(args[1])
  splits = pt.crossValidationSplits(args[0])
  avgAccuracy = 0.0
  fold = 0
  for split in splits:
    classifier = Perceptron()
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
    
    
def classifyDir(trainDir, testDir,iter):
  classifier = Perceptron()
  trainSplit = classifier.trainSplit(trainDir)
  iterations = int(iter)
  classifier.train(trainSplit,iterations)
  testSplit = classifier.trainSplit(testDir)
  #testFile = classifier.readFile(testFilePath)
  accuracy = 0.0
  for example in testSplit.train:
    words = example.words
    guess = classifier.classify(words)
    if example.klass == guess:
      accuracy += 1.0
  accuracy = accuracy / len(testSplit.train)
  print('[INFO]\tAccuracy: %f' % accuracy)
    
def main():
  (options, args) = getopt.getopt(sys.argv[1:], '')
  
  if len(args) == 3:
    classifyDir(args[0], args[1], args[2])
  elif len(args) == 2:
    test10Fold(args)

if __name__ == "__main__":
    #main()
  (options, args) = getopt.getopt(sys.argv[1:], '')
  
  if len(args) == 3:
    classifyDir(args[0], args[1], args[2])
  elif len(args) == 2:
    test10Fold(args)