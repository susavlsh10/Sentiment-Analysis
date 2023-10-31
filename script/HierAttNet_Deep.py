import sys
import getopt
import os
import math
import operator
import numpy as np
import json
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import BertTokenizer, BertModel

class BERT_Sentiment(nn.Module):
    def __init__(self, device, output_dim=2):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased').to(device)
        self.device = device
        self.tokenizer= BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)   
        
        embedding_dim = self.bert.config.to_dict()['hidden_size']
        self.out = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim//2, bias=True),
            nn.ReLU(),
            nn.Linear(embedding_dim//2, output_dim, bias=True),
            )
        
    def forward(self, input_ids, token_type_ids, attention_mask):
        embedded = self.bert(input_ids, token_type_ids, attention_mask)
        output = self.out(embedded[1])
        return output
    
    def tokenize(self, text):
        encode_dict = self.tokenizer.encode_plus(text,
                                     add_special_tokens=True,
                                     return_attention_mask=True,
                                     max_length=512,
                                     truncation=True,
                                     padding = 'max_length',
                                     return_tensors= 'pt')
        return encode_dict
    
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


  def __init__(self, device):
    """network initialization"""
    self.numFolds = 10
    self.batch=1
    self.device = device
    
    
    #self.scheduler= get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=0, num_training_steps= total_steps)
    
    self.label = {'pos': 0, 
                  'neg': 1}

    
  #############################################################################
  # TODO TODO TODO TODO TODO 
  # Implement the HierAttNet classifier 


  def classify(self, words):
    """ TODO
      implement the prediction function of HierAttNet  
      'words' is a list of words to classify. Return 'pos' or 'neg' classification.
    """

    # Write code here
    device = self.device
    encoded_dict = self.model.tokenize(words)
    
    input_ids = encoded_dict['input_ids'].to(device)
    attention_masks = encoded_dict['attention_mask'].to(device)
    
    #forward pass
    with torch.no_grad():
        output = self.model(input_ids=input_ids, token_type_ids = None, attention_mask=attention_masks)
    output = output.to('cpu').detach().numpy()
    
    if output[0][0]>output[0][1]:
        return 'pos'
    else:
        return 'neg'

  
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

    # Write code here
    
    ''' Convert text to BERT tokens'''
    device = self.device
    self.model = BERT_Sentiment(device=device, output_dim=2)
    
    input_ids = []
    attention_masks = []
    labels = []
    
    for example in split.train:
        words = example.words
        klass = example.klass
        label = self.label[klass]
        encoded_dict = self.model.tokenize(words)
        
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
        labels.append(label)
    
    #convert data structures to tensor
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks= torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)
    
    #creating datasets for the training
    dataset = TensorDataset(input_ids, attention_masks, labels)
    
    train_dataloader = DataLoader(dataset,
                                  sampler=RandomSampler(dataset),
                                  batch_size=self.batch)
    
    self.model = self.model.to(device)
    total_steps = input_ids.shape[0] * iterations
    t_step = input_ids.shape[0] // self.batch
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5, eps=1e-8)

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    
    #enable training mode
    self.model.train()
    
    ''' Main training loop '''
    for i in range(0, iterations):
        print("")
        print('======== Epoch {:} / {:} ========'.format(i, iterations))
        print('Training...')
        for step, batch in enumerate(train_dataloader):
            b_input_ids = batch[0].to(device)
            b_masks = batch[1].to(device)
            b_labels = batch[2].to(device)
            
            optimizer.zero_grad()
            
            result = self.model(input_ids=b_input_ids, token_type_ids = None, attention_mask=b_masks)
        
            loss = criterion(result, b_labels)
            print('Step {:} / {:} === Loss: {:} ====='.format(step, t_step, loss))
            loss.backward()
            
            optimizer.step()
            scheduler.step()


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
    result = ' <eos> '.join(contents)
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
  if torch.cuda.is_available():
    #use GPU
    device= torch.device("cuda")
    print('GPU available: ', torch.cuda.get_device_name(0))

  else:
    device=torch.device("cpu")
    
  pt = HierAttNet(device)
  pt.batch=int(args[2])
  iterations = int(args[1])
  splits = pt.crossValidationSplits(args[0])
  avgAccuracy = 0.0
  fold = 0

    
  for split in splits:
    classifier = HierAttNet(device)
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
    #main()
    (options, args) = getopt.getopt(sys.argv[1:], '')
    test10Fold(args)