from nltk.corpus import stopwords
from collections import Counter
import regex as re
from tqdm import tqdm
import string
import math
import statistics as stat
usable = {}
uids = {}
c1 = Counter()
c2 = Counter()
from nltk.stem import WordNetLemmatizer
lm = WordNetLemmatizer()
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

cd = {}
ce = {}
dd = {}
de = {}
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
    ha, hb = [], []
    for i in s:
        if i.startswith('#') and 'fauci' in i.lower():
            if 'SV' in i and 'SVC' not in i:
                sub = i.split('S: ')[1].split('@')[0]
                verb = i.split('V: ')[1].split('@')[0]
                v = lm.lemmatize(verb.lower(), pos='v')
                if '"has"' in verb:
                    continue
                if v in stops:
                    continue
                if 'fauci' in sub.lower():
                    c1 = c1 + Counter([v])
                    if v not in cd:
                        cd[v] = [usable[y]]
                    else:
                        cd[v] = cd[v] + [usable[y]]
                    if v not in ce:
                        ce[v] = [uids[y]]
                    else:
                        ce[v] = ce[v] + [uids[y]]
                    ha += [f'AS_AGENT({v})']
                else:
                    c2 = c2 + Counter([v])
                    if v not in dd:
                        dd[v] = [usable[y]]
                    else:
                        dd[v] = dd[v] + [usable[y]]
                    if v not in de:
                        de[v] = [uids[y]]
                    else:
                        de[v] = de[v] + [uids[y]]
                    hb += [f'AS_PATIENT({v})']
    if 'hist' in sys.argv:
        print(len(ha),len(hb))
    elif 'list' in sys.argv:
        print('\t'.join([str(y)]+ha+hb))        
if 'hist' not in sys.argv:
    print(len([i for i in c1.elements()]))
    for i in c1.most_common():
        if i[1] > 1:
            std = stat.stdev(cd[i[0]])
            se = std / math.sqrt(i[1])
            m = stat.mean(cd[i[0]])
            uniques = len(set(ce[i[0]]))
            if se != 0:
                tsc = (m-22.8)/se
            if uniques >= 20 and abs(tsc) > 2:
                print(f'{i[0]}\t{i[1]}\t{uniques}\t{tsc}')
    print(len([i for i in c2.elements()]))
    for i in c2.most_common():
        if i[1] > 1:
            std = stat.stdev(dd[i[0]])
            se = std / math.sqrt(i[1])
            m = stat.mean(dd[i[0]])
            uniques = len(set(de[i[0]]))
            if se != 0:
                tsc = (m-22.8)/se
            if uniques >= 20 and abs(tsc) > 2:
                print(f'{i[0]}\t{i[1]}\t{uniques}\t{tsc}')
