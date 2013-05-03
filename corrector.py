import sys
import marshal
import itertools
import operator
import math
import os, os.path
from collections import deque

queries_loc = 'data/queries.txt'
gold_loc = 'data/gold.txt'
google_loc = 'data/google.txt'

alphabet = "abcdefghijklmnopqrstuvwxyz0123546789&$+_' "

def unserialize_data(fname):
  """
  Reads a pickled data structure from a file named `fname` and returns it
  IMPORTANT: Only call marshal.load( .. ) on a file that was written to using marshal.dump( .. )
  marshal has a whole bunch of brittle caveats you can take a look at in teh documentation
  It is faster than everything else by several orders of magnitude though
  """
  with open(fname, 'rb') as f:
    return marshal.load(f)

def candidate_gen(word, index, word_dict, thresh):
  """
  Generate candidate using k-gram index
  """
  if len(word) <= 2:
    return [word]
  word_aug = '$' + word + '$'
  idx = range(len(word_aug))
  query = set()
  for tup in itertools.izip(idx[:-1],idx[1:]):
    query.add(word_aug[tup[0]]+word_aug[tup[1]])
  candidate = {}
  query = [u for u in query if u in index]
  for item in query:
    lst = [word_dict[u] for u in index[item] if abs(len(word_dict[u])-len(word)) <= 2]
    for w in lst:
      if w in candidate:
        candidate[w] += 1
      else:
        candidate[w] = 1
  cand = []
  for key in candidate:
    score = float(candidate[key]) / float(len(key) + 1 + len(query) - candidate[key])
    if score >= thresh:
      cand.append(key)
  return cand

def edit_distance(word1, word2):
  """
  Compute exact edit distance
  """
  w1 = ' ' + word1
  w2 = ' ' + word2
  d = {}
  for i in range(len(w1)):
    d[i,0] = i
  for j in range(len(w2)):
    d[0,j] = j
  for i in range(1, len(w1)):
    for j in range(1, len(w2)):
      if w1[i] == w2[j]:
        d[i,j] = d[i-1,j-1]
      else:
        d[i,j] = min(d[i-1,j] + 1, d[i,j-1] + 1, d[i-1,j-1] + 1)
      if i > 1 and j > 1 and w1[i] == w2[j-1] and w1[i-1] == w2[j]:
        d[i,j] = min(d[i,j], d[i-2,j-2] + 1)
  return d[len(w1)-1, len(w2)-1]
  
def language_model(query, prob):
  """
  Compute the score of language model
  """
  score = 0.0
  score += math.log10(prob[query[0]])
  idx = range(len(query))
  for tup in itertools.izip(idx[:-1],idx[1:]):
    bigram = (query[tup[0]], query[tup[1]])
    score += math.log10(prob[bigram])
  return score


def noisy_model(correct, wrong, model):
  """
  Noisy edit model
  """
  sc = float(len(set(correct).intersection(set(wrong)))) * math.log10(0.90)
  err_list = edit_distance(' '.join(correct), ' '.join(wrong))
  sc += err_list * math.log10(0.05)
  return sc
  

def gen_candidate_query(query, index, word_dict, prob, thresh):
  """
  Generate possible query
  """
  queue = deque([])
  lst = candidate_gen(query[0], index, word_dict, thresh)
  lst = [u for u in lst if edit_distance(u, query[0]) <= 2]
  for item in lst:
    queue.append([item])
  for i in range(1, len(query)):
    lst = candidate_gen(query[i], index, word_dict, thresh)
    lst = [u for u in lst if edit_distance(u, query[i]) <= 2]
    while len(queue) != 0:
      seq = queue.popleft()
      if len(seq) == i+1:
        queue.append(seq)
        break
      for item in lst:
        if (seq[-1], item) in prob:
          queue.append(seq + [item])
  return list(queue)

def gen_candidate_query_v2(query, index, word_dict, prob, thresh):
  """
  Generate possible query
  """
  queue = deque([])
  if query[0] in prob:
    lst = [query[0]]
  else:
    lst = candidate_gen(query[0], index, word_dict, thresh)
    lst = [u for u in lst if edit_distance(u, query[0]) <= 2]
  for item in lst:
    queue.append([item])

  for i in range(1, len(query)):
    if query[i] in prob:
      lst = [query[i]]
    else:
      lst = candidate_gen(query[i], index, word_dict, thresh)
      lst = [u for u in lst if edit_distance(u, query[i]) <= 2]
      
    while len(queue) != 0:
      seq = queue.popleft()
      if len(seq) == i+1:
        queue.append(seq)
        break
      for item in lst:
        if (seq[-1], item) in prob:
          queue.append(seq + [item])
  return list(queue)

def score(original, current, lang_model, edit_model):
  """
  Output score
  """
  sc = language_model(current, lang_model)
  sc += noisy_model(current, original, edit_model)
  return sc

def read_query_data():
  """
  all three files match with corresponding queries on each line
  """
  queries = []
  gold = []
  google = []
  with open(queries_loc) as f:
    for line in f:
      queries.append(line.rstrip())
  with open(gold_loc) as f:
    for line in f:
      gold.append(line.rstrip())
  with open(google_loc) as f:
    for line in f:
      google.append(line.rstrip())
  assert( len(queries) == len(gold) and len(gold) == len(google) )
  return (queries, gold, google)

def split_combine(qry, prob):
  lst = []
  # Split
  for idx in range(0, len(qry)):
    item = qry[idx]
    for i in range(1, len(item)):
      if item[0:i] in prob and item[i:] in prob:
        temp = []
        temp.extend(qry)
        temp[idx] = item[0:i]
        temp.insert(idx+1, item[i:])
        lst.append(temp)
  # Combine
  ii = range(len(qry))
  for tup in itertools.izip(ii[:-1],ii[1:]):
    if qry[tup[0]] + qry[tup[1]] in prob:
      temp = []
      temp.extend(qry)
      temp[tup[0]] = qry[tup[0]] + qry[tup[1]]
      temp.pop(tup[1])
      lst.append(temp)
  return lst
  

if __name__ == '__main__':
  model_dir = './model'
  lang = unserialize_data(model_dir + os.sep + "language_model")
  noisy = unserialize_data(model_dir + os.sep + "edit_model")
  index = unserialize_data(model_dir + os.sep + "index")
  word_dict = unserialize_data(model_dir + os.sep + "word")
  data = read_query_data()

  # Parameters
  thresh = 0.5

  question = data[0]
  answer = data[1]
  google = data[2]

  acc = len(answer)
  print acc
  for i in range(len(question)):
    if i % 10 == 0:
      print "Progess %d" % i
    qry = question[i].split()
    cand = gen_candidate_query(qry, index, word_dict, lang, thresh)
    SplitCombine = split_combine(qry, lang)
    for item in SplitCombine:
      cand = cand + gen_candidate_query_v2(item, index, word_dict, lang, thresh)

    
    ans = ''
    if len(cand) == 0:
      ans = ' '.join(qry)
    else:
      rank = [score(qry, u, lang, noisy) for u in cand]
      idx = rank.index(max(rank))
      ans = ' '.join(cand[idx])
    if ans != answer[i]:
      acc -= 1
      print question[i]
      print ans
      print answer[i]
      print '----------------------------'
  print 'Accuracy: %d' % acc
    
    
