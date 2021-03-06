import sys
import os.path, os
import itertools
import math
import marshal
from glob import iglob

inter_lambda = 0.2

def serialize_data(data, fname):
  """
  Writes `data` to a file named `fname`
  """
  with open(fname, 'wb') as f:
    marshal.dump(data, f)

def add_table(dic, key):
  if key not in dic:
    dic[key] = 1
  else:
    dic[key] += 1

def load_corpus(training_corpus_loc):
  """
  Scans through the training corpus
    1. counts how many bigrams and unigrams there are
    2. builds bigram indices
  """
  count, bigram_idx, word_dict = {}, {}, {}
  tot_cnt, word_id = 0, 1
  for block_fname in iglob(os.path.join( training_corpus_loc, '*.txt')):
    print >> sys.stderr, 'processing dir: ' + block_fname
    doc = []
    with open(block_fname, 'r') as f:
      for line in f.readlines():
        line = line.strip().split()
        doc += line
    tot_cnt += len(doc)
    for unigram in doc:
      # Bigram index
      if unigram not in count:
        word_dict[word_id] = unigram
        temp = '$' + unigram + '$'
        i = range(len(temp))
        for tup in itertools.izip(i[:-1],i[1:]):
          k = temp[tup[0]] + temp[tup[1]]
          if k not in bigram_idx:
            bigram_idx[k] = [word_id]
          else:
            bigram_idx[k].append(word_id)
        word_id += 1
      add_table(count, unigram)
      
    idx = range(len(doc))
    for tup in itertools.izip(idx[:-1],idx[1:]):
      bigram = (doc[tup[0]], doc[tup[1]])
      add_table(count, bigram)
      
  # Sorting bigram index
  for k in bigram_idx:
    bigram_idx[k] = sorted(set(bigram_idx[k]))
  return count, tot_cnt, bigram_idx, word_dict

def calc_prob(count, tot_cnt, lam):
  """
  Calculating probabilities in log-space using unigram/bigram count
  """
  global inter_lambda
  prob = {}
  for item in count:
    if isinstance(item, tuple):
      prob[item] = float(count[item]) / float(count[item[0]])
    elif isinstance(item, str):
      prob[item] = float(count[item]) / float(tot_cnt)
  for item in prob:
    if isinstance(item, tuple):
      prob[item] = (1-inter_lambda) * prob[item] + inter_lambda * prob[item[1]]
  return prob

def edit_type(wrong, correct):
  """
  Deciding which type of edits occured in this pair of string
  """
  if len(correct) == len(wrong):
    for i in range(0, len(correct)):
      if correct[i] != wrong[i]:
        # Transposition
        if i != len(correct) - 1 and correct[i] == wrong[i+1]:
          return ("t", correct[i], correct[i+1])
        # Substitution
        else:
          return ("s", correct[i], wrong[i])
  elif len(correct) > len(wrong):
    # Deletion
    for i in range(0, len(correct)):
      if i >= len(wrong) or wrong[i] != correct[i]:
        if i == 0:
          return ("d",' ',correct[i])  
        else:
          return ("d",correct[i-1],correct[i])
  else:
    assert(len(correct) < len(wrong))
    # Insertion
    for i in range(0, len(wrong)):
      if i >= len(correct) or correct[i] != wrong[i]:
        if i == 0:
          return ("i", ' ', wrong[i])
        else:
          return ("i", correct[i-1], wrong[i])
  assert(1 == 0)
  
def process_noisy_model(edit_file):
  """
  Returns the counts
  """
  noisy_model = {}
  with open(edit_file, 'r') as f:
    for line in f.readlines():
      line = line.rstrip().split('\t')
      # Type error
      if line[0] != line[1]:
        tup = edit_type(line[0], line[1])
        add_table(noisy_model, tup)
        
      # Singleton
      for t in line[1]:
        add_table(noisy_model, t)

      # Bigram
      idx = range(len(line[1]))
      for tup in itertools.izip(idx[:-1],idx[1:]):
        add_table(noisy_model, (line[1][tup[0]], line[1][tup[1]]))
      if len(line[1]) > 0: #beginning of the line
        add_table(noisy_model, (' ', line[1][0]))
  return noisy_model


if __name__ == '__main__':
  if len(sys.argv) != 3 and len(sys.argv) != 4:
    print >> sys.stderr, 'usage: python models.py [extra] <training corpus dir> <training edit1s file>'
    os._exit(-1)
  elif len(sys.argv) == 4:
    assert(sys.argv[1] == "extra")
    train_lang_dir = sys.argv[2]
    train_edit_dir = sys.argv[3]
  else:
    train_lang_dir = sys.argv[1]
    train_edit_dir = sys.argv[2]

  count, tot, index, word_dict = load_corpus(train_lang_dir)
  
  prob = calc_prob(count, tot, inter_lambda)

  model_dir = './model'
  if not os.path.exists(model_dir):
    os.makedirs(model_dir)

  serialize_data(prob, model_dir + os.sep + "language_model")
  serialize_data(index, model_dir + os.sep + "index")
  serialize_data(word_dict, model_dir + os.sep + "word")

  edit = process_noisy_model(train_edit_dir)
  serialize_data(edit, model_dir + os.sep + "edit_model")

  
  

  
  
    
  
