import sys
import getopt
import os
import math
import operator

class NaiveBayes:
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
    """NaiveBayes initialization"""
    self.FILTER_STOP_WORDS = False
    self.BOOLEAN_NB = False
    self.stopList = set(self.readFile('../data/english.stop'))
    self.numFolds = 10
    
    """ Define dictionaries/hash tables of the vocabulary"""
    self.vocabulary = {}
    self.pos_vocab = {} 
    self.neg_vocab = {}
    
    self.dirty = True # if true new examples added or probabilites need to be calculated. If false, conditional probabilites calculated.
    self.pos_prob = {}
    self.neg_prob = {}
    self.pos_len = 0
    self.neg_len = 0
    self.vocab_len = 0
    
    self.count_pos = 0
    self.count_neg = 0
    self.p_pos = 0
    self.p_neg = 0
    
  #############################################################################
  # TODO TODO TODO TODO TODO 
  # Implement the Multinomial Naive Bayes classifier and the Naive Bayes Classifier with
  # Boolean (Binarized) features.
  # If the BOOLEAN_NB flag is true, your methods must implement Boolean (Binarized)
  # Naive Bayes (that relies on feature presence/absence) instead of the usual algorithm
  # that relies on feature counts.
  #
  #
  # If any one of the FILTER_STOP_WORDS and BOOLEAN_NB flags is on, the
  # other one is meant to be off.

  def compute_conditional_probabilities(self):
    self.vocab_len = len(self.vocabulary.keys())
    if self.BOOLEAN_NB:
        self.pos_len = len(self.pos_vocab.keys())
        self.neg_len = len(self.neg_vocab.keys())
    self.p_pos= self.count_pos/(self.count_pos + self.count_neg)
    self.p_neg = self.count_neg/(self.count_pos + self.count_neg)
    
    for i in self.vocabulary.keys():
        if i not in self.pos_vocab:
            count_w_pos = 0
        else:
            count_w_pos =self.pos_vocab[i] 
        if i not in self.neg_vocab:
            count_w_neg = 0
        else:
            count_w_neg = self.neg_vocab[i]
        self.pos_prob[i] = (count_w_pos + 1)/(self.pos_len + self.vocab_len + 1)
        self.neg_prob[i] = (count_w_neg + 1)/(self.neg_len + self.vocab_len + 1)      

  def classify(self, words):
    """ TODO
      'words' is a list of words to classify. Return 'pos' or 'neg' classification.
    """
    if self.FILTER_STOP_WORDS:
      words =  self.filterStopWords(words)

    # Write code here
   
    """ Calculate conditional probabilities if not already calculated """
    if self.dirty:
        self.compute_conditional_probabilities()
        self.dirty = False
    
    """ Create test document hash table """
    test_doc = {}
    if not self.BOOLEAN_NB:    
        for w in words:
            if w in test_doc:
                test_doc[w] = test_doc[w] + 1
            else:
                test_doc[w] = 1
    elif self.BOOLEAN_NB:
        for w in words:
            test_doc[w] = 1
        
    
    """ Caclulate probabilites and choose a class"""
    p_pos_test = 0
    p_neg_test = 0
    
    for t in test_doc.keys():
        if t not in self.vocabulary:
            pos_prob_test = 1/(self.pos_len + self.vocab_len + 1)
            neg_prob_test = 1/(self.neg_len + self.vocab_len + 1)
        else:
            pos_prob_test = self.pos_prob[t]
            neg_prob_test = self.neg_prob[t]
        p_pos_test = p_pos_test + (test_doc[t]*math.log(pos_prob_test))
        p_neg_test = p_neg_test + (test_doc[t]*math.log(neg_prob_test))
    
    p_pos_test = ((math.log(self.p_pos)) + p_pos_test)
    p_neg_test = ((math.log(self.p_neg)) + p_neg_test)
    
    if p_pos_test > p_neg_test:
        return 'pos'
    else:
        return 'neg'
    

  def addExample(self, klass, words):
    """
     * TODO
     * Train your model on an example document with label klass ('pos' or 'neg') and
     * words, a list of strings.
     * You should store whatever data structures you use for your classifier 
     * in the NaiveBayes class.
     * Returns nothing
    """
    
    """ Check for Binarized Naive Bayes """
    if self.BOOLEAN_NB:
        self.BinaryNB_train(klass, words)
        #return
    else:    
        """ Update vocabulary"""
        if klass =='pos':
            self.pos_len = self.pos_len + len(words)
            self.count_pos =  self.count_pos + 1
        elif klass =='neg':
            self.neg_len = self.neg_len + len(words)
            self.count_neg = self.count_neg + 1
        
        for i in words:
            if i in self.vocabulary:
                self.vocabulary[i] = self.vocabulary[i] + 1
            else:
                self.vocabulary[i] = 1
                
            if klass =='pos': 
                if i in self.pos_vocab:
                    self.pos_vocab[i]=self.pos_vocab[i] + 1
                else:
                    self.pos_vocab[i] = 1
                    
            elif klass =='neg':
                if i in self.neg_vocab:
                    self.neg_vocab[i]=self.neg_vocab[i] + 1 
                else:
                    self.neg_vocab[i] = 1
    """ Update dirty bit. If 1 calculate conditional probabilities during inference --> P(w|c) """
    self.dirty = True
    
  def BinaryNB_train(self, klass, words):
    """ Update vocabulary"""
    #find all the unique words
    temp = {}
    for i in words:
        temp[i] = 1        
        if i not in self.vocabulary:
            self.vocabulary[i] = 1    
    for t in temp.keys():
        if klass =='pos':
            if t in self.pos_vocab:
                self.pos_vocab[t] = self.pos_vocab[t] + 1
            else:
                self.pos_vocab[t] = 1
        if klass=='neg':
            if t in self.neg_vocab:
                self.neg_vocab[t] = self.neg_vocab[t] + 1
            else:
                self.neg_vocab[t] = 1
    
    # increment count for pos or neg class
    if klass =='pos':
        self.count_pos =  self.count_pos + 1
    elif klass =='neg':
        self.count_neg = self.count_neg + 1
    
    self.dirty = True
      
        

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

  def train(self, split):
    for example in split.train:
      words = example.words
      if self.FILTER_STOP_WORDS:
        words =  self.filterStopWords(words)
      self.addExample(example.klass, words)


  def crossValidationSplits(self, trainDir):
    """Returns a list of TrainSplits corresponding to the cross validation splits."""
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
  
  def filterStopWords(self, words):
    """Filters stop words."""
    filtered = []
    for word in words:
      if not word in self.stopList and word.strip() != '':
        filtered.append(word)
    return filtered

def test10Fold(args, FILTER_STOP_WORDS, BOOLEAN_NB):
  nb = NaiveBayes()
  splits = nb.crossValidationSplits(args[0])
  avgAccuracy = 0.0
  fold = 0
  for split in splits:
    classifier = NaiveBayes()
    classifier.FILTER_STOP_WORDS = FILTER_STOP_WORDS
    classifier.BOOLEAN_NB = BOOLEAN_NB
    accuracy = 0.0
    for example in split.train:
      words = example.words
      classifier.addExample(example.klass, words)
  
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
    
    
def classifyDir(FILTER_STOP_WORDS, BOOLEAN_NB, trainDir, testDir):
  classifier = NaiveBayes()
  classifier.FILTER_STOP_WORDS = FILTER_STOP_WORDS
  classifier.BOOLEAN_NB = BOOLEAN_NB
  trainSplit = classifier.trainSplit(trainDir)
  classifier.train(trainSplit)
  testSplit = classifier.trainSplit(testDir)
  accuracy = 0.0
  for example in testSplit.train:
    words = example.words
    guess = classifier.classify(words)
    if example.klass == guess:
      accuracy += 1.0
  accuracy = accuracy / len(testSplit.train)
  print('[INFO]\tAccuracy: %f' % accuracy)


def main():
  FILTER_STOP_WORDS = False
  BOOLEAN_NB = False
  (options, args) = getopt.getopt(sys.argv[1:], 'fbm')
  if ('-f','') in options:
    FILTER_STOP_WORDS = True
  elif ('-b','') in options:
    BOOLEAN_NB = True

  print('len(args): ',len(args))
  
  if len(args) == 2:
    classifyDir(FILTER_STOP_WORDS, BOOLEAN_NB,  args[0], args[1])
  elif len(args) == 1:
    test10Fold(args, FILTER_STOP_WORDS, BOOLEAN_NB)

if __name__ == "__main__":
    #print('Here')
    #main()
    FILTER_STOP_WORDS = False
    BOOLEAN_NB = False
    (options, args) = getopt.getopt(sys.argv[1:], 'fbm')
    if ('-f','') in options:
      FILTER_STOP_WORDS = True
    elif ('-b','') in options:
      BOOLEAN_NB = True

    print('len(args): ',len(args))
    
    if len(args) == 2:
      classifyDir(FILTER_STOP_WORDS, BOOLEAN_NB,  args[0], args[1])
    elif len(args) == 1:
      test10Fold(args, FILTER_STOP_WORDS, BOOLEAN_NB)
