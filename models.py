import sys
import os.path
import itertools
import math
import marshal
from glob import iglob

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
  prob = {}
  for item in count:
    if isinstance(item, tuple):
      prob[item] = float(count[item]) / float(count[item[0]])
    elif isinstance(item, str):
      prob[item] = float(count[item]) / float(tot_cnt)
  for item in prob:
    if isinstance(item, tuple):
      prob[item] = (1-lam) * prob[item] + lam * prob[item[1]]
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
          return (correct[i]+correct[i+1], wrong[i]+wrong[i+1])
        # Substitution
        else:
          return (correct[i], wrong[i])
  elif len(correct) > len(wrong):
    # Deletion
    for i in range(0, len(wrong)):
      if wrong[i] != correct[i]:
        return (correct[i]+correct[i+1], wrong[i])
    return (correct[-2:], wrong[-1])
  else:
    # Insertion
    for i in range(0, len(correct)):
      if correct[i] != wrong[i]:
        return (correct[i], wrong[i]+wrong[i+1])
    return (correct[-1], wrong[-2:])
  assert(1 == 0)
  
def noisy_model(edit_file):
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
        add_table(noisy_model, line[1][tup[0]]+line[1][tup[1]])
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

  inter_lambda = 0.2
  count, tot, index, word_dict = load_corpus(train_lang_dir)
  
  prob = calc_prob(count, tot, inter_lambda)
  serialize_data(prob, "language_model")
  serialize_data(index, "index")
  serialize_data(word_dict, "word")

  edit = noisy_model(train_edit_dir)
  serialize_data(edit, "edit_model")

  
  

  
  
    
  
