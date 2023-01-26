from nltk.corpus import stopwords
from collections import Counter
import regex as re
from tqdm import tqdm
import string
import math
import statistics as stat
usable = {}
uids = {}
from nltk.stem import WordNetLemmatizer
import sys

# Helper function
def proc_dep(dep):
    lines = dep.split('\n')[9:]
    tokens = [t.split('\t')[1].lower() for t in lines]
    postags = [t.split('\t')[3] for t in lines]
    nertags = [t.split('\t')[-1] for t in lines]
    deprels = [t.split('\t')[7] for t in lines]
    parents = [int(t.split('\t')[6])-1 for t in lines]
    nums = [int(t.split('\t')[0])-1 for t in lines]
    return lines, tokens, postags, nertags, deprels, parents, nums

with open('fauci.conllu') as fcl:
    for li, l in enumerate(fcl.read().split('\n\n')[:-1]):
        lines, tokens, postags, nertags, deprels, parents, nums = proc_dep(l)
        ops = l.split('\n')[:9]
        followers, verified = ops[3], ops[4]
        follow_count, veri_bool = int(ops[3].split('# followers = ')[1]), ops[4]=='# verified = True'
        pol = float(ops[-1].split('# county = ')[1])
        uid = ops[2]
        if follow_count <= 10000 and not veri_bool and 'fauci' in tokens:
            usable[li] = pol
            uids[li] = uid

lm = WordNetLemmatizer()
c = Counter()
cd = {}
ce = {}
stops = set(stopwords.words('english'))
x = 0
user_pat = re.compile(r'@[A-Za-z0-9_]{1,15}')
punct = string.punctuation.replace(' ','')
for y in sorted(usable):
    flag = True
    with open(f'cie_outputs/output_{y}') as cout:
        co = cout.read()
        if co.split('\n')[-2].startswith('#'):
            flag = False
        if flag:
            s = co.split('Detected')[1].split('\n')[1:]
        else:
            s = []
    dx = []
    for i in s:
        if i.startswith('#') and 'fauci' in i.lower():
            if 'SVC' in i:
                sub = i.split('S: ')[1].split('@')[0]
                if 'C: ' not in i:
                    continue
                comp = i.split('C: ')[1].split('@')[0]
                sub = sub.translate(str.maketrans('', '', punct)).strip(' ')
                comp = comp.translate(str.maketrans('', '', punct)).strip(' ')
                if 'fauci' in sub.lower():
                    t = lm.lemmatize(comp.lower())
                else:
                    t = lm.lemmatize(sub.lower())
                if t not in stops:
                    c = c + Counter([t])
                    if t not in cd:
                        cd[t] = [usable[y]]
                    else:
                        cd[t] = cd[t] + [usable[y]]
                    if t not in ce:
                        ce[t] = [uids[y]]
                    else:
                        ce[t] = ce[t] + [uids[y]]
                    dx += [f'IS_A({t})']
    if 'hist' in sys.argv:
        print(len(d))
    elif 'list' in sys.argv:
        print('\t'.join([str(y)]+dx))        
if 'hist' not in sys.argv:
    print(len([i for i in c.elements()]))
    for i in c.most_common():
        if i[1] > 1:
            std = stat.stdev(cd[i[0]])
            se = std / math.sqrt(i[1])
            m = stat.mean(cd[i[0]])
            uniques = len(set(ce[i[0]]))
        
            if se != 0:
                tsc = (m-22.8)/se
            if uniques >= 20 and abs(tsc) > 2:
                print(f'{i[0]}\t{i[1]}\t{uniques}\t{tsc}')
