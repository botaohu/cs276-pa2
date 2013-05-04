import sys
import marshal
import itertools
import operator
import math
import os, os.path
import automata
from collections import deque

alphabet = "abcdefghijklmnopqrstuvwxyz0123546789&$+_' "
bindex_thresh = 0.5
spell_correct_rate = 0.9
uniform_error_rate = 0.05
cand_topk_prev = 20
cand_topk_cur = 20

def unserialize_data(fname):
  """
  Reads a pickled data structure from a file named `fname` and returns it
  IMPORTANT: Only call marshal.load( .. ) on a file that was written to using marshal.dump( .. )
  marshal has a whole bunch of brittle caveats you can take a look at in teh documentation
  It is faster than everything else by several orders of magnitude though
  """
  with open(fname, 'rb') as f:
    return marshal.load(f)

  
def edit_distance_plus(w1, w2):
  """
  Compute exact edit distance with noisy score
  """

  def update(d, p, s, i, j, pi, pj, c, err):
    err_cost = noisy_model(err)
    if d[i, j] > d[pi, pj] + c or (d[i, j] == d[pi, pj] + c and s[i, j] > s[pi, pj] + err_cost):
      d[i, j] = d[pi, pj] + c
      p[i, j] = err
      s[i, j] = s[pi, pj] + err_cost

  d = {}
  p = {}
  s = {}
  w1 = " " + w1
  w2 = " " + w2 
  for i in range(len(w1)):
    for j in range(len(w2)):
      if i == 0 and j == 0:
        d[i,j] = 0
        s[i,j] = 0
      else:
        d[i,j] = 1000000
        s[i,j] = float('inf')
      p[i,j] = ("","")
      if i > 0 and j > 0:
        if w1[i] == w2[j]:
          update(d, p, s, i, j, i - 1, j - 1, 0, ("c", "", ""))
        else:
          update(d, p, s, i, j, i - 1, j - 1, 1, ("s", w1[i], w2[j]))
      if i > 0:  #delete
        update(d, p, s, i, j, i - 1, j, 1, ('d', ' ', w1[i]) if i == 1 else ('d', w1[i - 1], w1[i]))
      if j > 0: #insert
        update(d, p, s, i, j, i, j - 1, 1, ('i', ' ', w2[j]) if i == 0 else ('i', w1[i], w2[j]))
      if i > 1 and j > 1 and w1[i] == w2[j-1] and w1[i-1] == w2[j]:
        update(d, p, s, i, j, i - 2, j - 2, 1, ('t', w1[i - 1], w1[i]))

  l1 = len(w1) - 1
  l2 = len(w2) - 1 
  i = l1
  j = l2
  ans = []
  while i > 0 or j > 0:
    err, from_ch, to_ch = p[i,j]
    if err != 'c':
      ans.append(p[i,j])
    if err == 'c' or err == 's':
      i -= 1
      j -= 1
    elif err == 't':
      i -= 2
      j -= 2
    elif err == 'd':
      i -= 1
    elif err == 'i':
      j -= 1
  ans.reverse()
  return d[l1, l2], s[l1, l2], ans

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
  
def gen_candidate(word):
  """
  Generate candidate using k-gram index or automata
  """
  global matcher, use_automata, bindex, word_dict, bindex_thresh, lang, cand_topk_cur
  if use_automata:
    cands = list(automata.find_all_matches(word, 2, matcher))
  else:
    word_aug = '$' + word + '$'
    idx = range(len(word_aug))
    query = set()
    for tup in itertools.izip(idx[:-1],idx[1:]):
      query.add(word_aug[tup[0]]+word_aug[tup[1]])
    candidate = {}
    query = [u for u in query if u in bindex]
    for item in query:
      lst = [word_dict[u] for u in bindex[item] if abs(len(word_dict[u])-len(word)) <= 2]
      for w in lst:
        if w in candidate:
          candidate[w] += 1
        else:
          candidate[w] = 1
    cands = []
    for key, cnt in candidate.iteritems():
      score = float(cnt) / float(len(key) + 1 + len(query) - cnt)
      if score >= bindex_thresh and edit_distance(key, word) <= 2:
        cands.append(key)
    if len(cands) == 0 and word in lang:
      cands.append(word)
  
  cands = [(cand, edit_distance_plus(cand, word)[1]) for cand in cands]
  cands = sorted(cands, key=lambda x: lang[x[0]] + x[1])
  cands = [cand for cand, score in cands[:cand_topk_cur]]
  return cands
  
def lang_model(word1, word2):
  """
  Compute the score of language model
  """
  global lang
  if word1 == '\0':
    return -math.log(lang[word2])
  else: 
    #TODO smooth  ??
    if (word1, word2) in lang:
      return -math.log(lang[word1, word2])
    else:
      #print >> sys.stderr, "not exists %s,%s in lang_model" % (word1, word2)
      return -math.log(lang[word2])
      
def noisy_model(err):
  global edit_model
  err_type, ch1, ch2 = err
  if err_type == 'c':
    return 0
  if err in edit_model:
    err_cnt = edit_model[err]
  else:
    err_cnt = 1
  if err_type == 't':
    cond_cnt = edit_model[(ch1, ch2)] if (ch1, ch2) in edit_model else 1
  else:
    cond_cnt = edit_model[ch1] if ch1 in edit_model else 1
  return -math.log(float(err_cnt) / cond_cnt)
def channel_model(correct, wrong):
  """
  Noisy edit model
  """
  global use_uniform

  if use_uniform:
    score = - edit_distance(correct, wrong) * math.log(uniform_error_rate)
  else:
    dist, score, path = edit_distance_plus(correct, wrong)
  return score

def read_query_data(queries_loc):
  """
  all three files match with corresponding queries on each line
  """
  queries = []
  with open(queries_loc) as f:
    for line in f:
      queries.append(line.rstrip())
  return queries
  
def load_models():
  global lang, bindex, word_dict, edit_model

  model_dir = './model'
  lang = unserialize_data(model_dir + os.sep + "language_model")
  edit_model = unserialize_data(model_dir + os.sep + "edit_model")
  bindex = unserialize_data(model_dir + os.sep + "index")
  word_dict = unserialize_data(model_dir + os.sep + "word")

  words = word_dict.values()
  words.sort()
  global matcher
  matcher = automata.Matcher(words)

def gen_split_candidate(word):
  global lang
  cands = []
  for i in range(1, len(word)):
    if word[0:i] in lang and word[i:] in lang:
      cands.append(word[0:i] + ' ' + word[i:])
  return cands

def do_inference(original_qry):
  qry = original_qry.split()
  global lang, cand_topk_prev

  markov = {}
  markov[-1] = [('\0', 0, [])] 
  for i in range(len(qry)):
    markov[i] = []
    cands = [("normal", cand) for cand in gen_candidate(qry[i])] #normal
    cands += [("split", cand) for cand in gen_split_candidate(qry[i])] #split "cand1 cand2"
    if i > 0: #combined
      cands += [("combined", cand) for cand in gen_candidate(qry[i - 1] + qry[i])]

    #normalization for wrong spelling
    score_channel_wrong_total = 0
    for cand_type, cand in cands:
      if cand_type == 'combined':
        score_channel_wrong_total += math.exp(-channel_model(cand, qry[i - 1] + ' ' + qry[i]))
      elif cand != qry[i]:
        score_channel_wrong_total += math.exp(-channel_model(cand, qry[i]))
    if score_channel_wrong_total > 0:
      log_score_channel_wrong_total = -math.log(score_channel_wrong_total)
    log_spell_correct_rate = - math.log(spell_correct_rate)
    log_spell_wrong_rate = - math.log(1 - spell_correct_rate)
    
    for cand_type, cand in cands:
      if cand_type == 'combined':
        score_channel = log_spell_wrong_rate + channel_model(cand, qry[i - 1] + ' ' + qry[i]) - log_score_channel_wrong_total
        prev_idx = i - 2
      else:
        if cand != qry[i]:
          score_channel = log_spell_wrong_rate + channel_model(cand, qry[i]) - log_score_channel_wrong_total
        else:
          score_channel = log_spell_correct_rate
        prev_idx = i - 1

      score = float("inf")
      best_sequence = []
      if cand_type == 'split':
        cand_bigram = cand.split()
      for prev_word, old_score, sequence in markov[prev_idx]: 
        if cand_type == 'split':
          score_lang = lang_model(prev_word, cand_bigram[0]) + lang_model(cand_bigram[0], cand_bigram[1])
        else:
          score_lang = lang_model(prev_word, cand)
        if score > old_score + score_lang + score_channel:
          score = old_score + score_lang + score_channel
          best_sequence = sequence + [cand]
      if score < float('inf'):
        if cand_type == 'split':
          markov[i].append((cand_bigram[1], score, best_sequence))
        else:
          markov[i].append((cand, score, best_sequence))
      
    result = sorted(markov[i], key=lambda x: x[1]) #score
    markov[i] = result[:cand_topk_prev]

  score = float("inf")
  best_sequence = []
  for cand, total_score, sequence in markov[len(qry) - 1]:
    if score > total_score:
      score = total_score
      best_sequence = sequence
  if score < float('inf'):
    return ' '.join(best_sequence)
  else:
    return ' '.join(qry)

def debuginit():
  global runmode, use_uniform, use_automata
  #parameter

  runmode = 'debug'
  queries_loc = 'data/queries.txt'
  gold_loc = 'data/gold.txt'
  google_loc = 'data/google.txt'
  
  question = read_query_data(queries_loc)
  use_uniform = False
  use_automata = True
  answer = read_query_data(gold_loc)
  google = read_query_data(google_loc)

  load_models()
  
  question = ['s']
  answer = ['s']
  google = ['s']
  
  acc = len(question)
  for i in range(len(question)):
    if i % 10 == 0:
      print >> sys.stderr, "Progess %d" % i
    ans = do_inference(question[i])
    
    if runmode == 'debug' and ans != answer[i]:
      acc -= 1
      print >> sys.stderr, question[i]
      print >> sys.stderr, ans
      print >> sys.stderr, answer[i]
      print >> sys.stderr, '----------------------------'
  if runmode == 'debug':
    print >> sys.stderr, 'Accuracy: %d' % acc
    

if __name__ == '__main__':

  global runmode, use_uniform, use_automata
  #parameter

  groundtruth = False
  if len(sys.argv) == 1: #debug mode 
    runmode = 'debug'
    groundtruth = True
    queries_loc = 'data/queries.txt'
    gold_loc = 'data/gold.txt'
    google_loc = 'data/google.txt'
  elif len(sys.argv) == 2:
    runmode = sys.argv[1] 
    groundtruth = True
    queries_loc = 'data/queries.txt'
    gold_loc = 'data/gold.txt'
    google_loc = 'data/google.txt'
  elif len(sys.argv) == 3:
    runmode = sys.argv[1]
    queries_loc = sys.argv[2]
  else:
    print >> sys.stderr, 'usage: python corrector.py <uniform | empirical | extra> <queries file>'
    os._exit(-1)

  question = read_query_data(queries_loc)
  if groundtruth:
    answer = read_query_data(gold_loc)
    google = read_query_data(google_loc)

  if runmode == 'debug':
    use_uniform = False
    use_automata = True
  elif runmode == 'uniform':
    use_uniform = True
    use_automata = False
  elif runmode == 'empirical':
    use_uniform = False
    use_automata = False
  elif runmode == 'extra':
    use_uniform = False
    use_automata = True

  load_models()
  
  if groundtruth:
    acc = len(question)
    print >> sys.stderr, acc

  for i in range(len(question)):
    if i % 10 == 0:
      print >> sys.stderr, "Progess %d" % i
    ans = do_inference(question[i])
    
    if groundtruth and ans != answer[i]:
      acc -= 1
      print >> sys.stderr, question[i]
      print >> sys.stderr, ans
      print >> sys.stderr, answer[i]
      print >> sys.stderr, '----------------------------'
  if groundtruth:
    print >> sys.stderr, 'Accuracy: %d' % acc
    
