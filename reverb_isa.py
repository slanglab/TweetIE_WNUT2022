from nltk.corpus import stopwords
from collections import Counter, defaultdict
import regex as re
import math
import statistics as stat
import sys
usable = {}
uids = {}
ee = defaultdict(str)
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
            ee[li] = l.split('\n')[1].split('# text =')[1]
c = Counter()
stops = set(stopwords.words('english'))
x = 0
cd = {}
ce = {}
dx = {}
for y in range(100000):
    dx[y] = []
user_pat = re.compile(r'@[A-Za-z0-9_]{1,15}')
with open('reverb_output') as rout:
    for line in rout:
        l = line.split('\t')
        y = int(l[0].split('fauci')[-1])
        if not y in usable:
            continue
        a,r,b = l[-3], l[-2], l[-1]
        a = user_pat.sub('@user',a)
        b = user_pat.sub('@user',b)
        if 'fauci' in a.split() or 'fauci' in b.split():
            a0, a1 = int(l[5]), int(l[6])
            r0, r1 = int(l[7]), int(l[8])
            b0, b1 = int(l[9]), int(l[10])
            chunks = l[-4].split(' ')
            if r == 'be':
                if 'fauci' in a.split():
                    bb = [bi.lower() for bi in b.split() if bi.lower() not in stops]
                    if len(bb) > 0:
                        c = c + Counter([' '.join(bb)])
                        t = ' '.join(bb)
                        if t not in cd:
                            cd[t] = [usable[y]]
                        else:
                            cd[t] = cd[t] + [usable[y]]
                        if t not in ce:
                            ce[t] = [uids[y]]
                        else:
                            ce[t] = ce[t] + [uids[y]]
                        dx[y] += [f'IS_A({t})']
                else:
                    aa = [ai.lower() for ai in a.split() if ai.lower() not in stops]
                    if len(aa) > 0:
                        c = c + Counter([' '.join(aa)])               
                        t = ' '.join(aa)
                        if t not in cd:
                            cd[t] = [usable[y]]
                        else:
                            cd[t] = cd[t] + [usable[y]]
                        if t not in ce:
                            ce[t] = [uids[y]]
                        else:
                            ce[t] = ce[t] + [uids[y]]
                        dx[y] += [f'IS_A({t})']
if 'hist' in sys.argv:
    for y in sorted(usable):
        print(len(dx[y]))
elif 'list' in sys.argv:
    for y in sorted(usable):
        print('\t'.join([str(y)]+dx[y]))        
elif 'text' in sys.argv:
    for y in sorted(usable):
        print(f'{y}\t{ee[y]}')        
else:
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
