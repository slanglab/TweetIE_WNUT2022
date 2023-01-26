from collections import Counter, defaultdict
from tqdm import tqdm
import random
import string
import statistics as stat
import math
from os import system
import nltk
import sys
from nltk.corpus import stopwords
stops = set(stopwords.words('english'))
lemma = nltk.wordnet.WordNetLemmatizer()

cnt = {
    'is_a_n':Counter(),
    'is_a_adj':Counter(),
    'has_a':Counter(),
    'as_agent':Counter(),
    'as_patient':Counter(),
    'as_conjunct':Counter()
}
pols = {
    'is_a_n':defaultdict(list),
    'is_a_adj':defaultdict(list),
    'has_a':defaultdict(list),
    'as_agent':defaultdict(list),
    'as_patient':defaultdict(list),
    'as_conjunct':defaultdict(list)
}
pops = {
    'is_a_n':defaultdict(list),
    'is_a_adj':defaultdict(list),
    'has_a':defaultdict(list),
    'as_agent':defaultdict(list),
    'as_patient':defaultdict(list),
    'as_conjunct':defaultdict(list)
}
users = {
    'is_a_n':defaultdict(list),
    'is_a_adj':defaultdict(list),
    'has_a':defaultdict(list),
    'as_agent':defaultdict(list),
    'as_patient':defaultdict(list),
    'as_conjunct':defaultdict(list)
}
flws = {
    'is_a_n':defaultdict(list),
    'is_a_adj':defaultdict(list),
    'has_a':defaultdict(list),
    'as_agent':defaultdict(list),
    'as_patient':defaultdict(list),
    'as_conjunct':defaultdict(list)   
}
def altprint(s, mode='bold'):
    prefix = {
            'purple_bg': '\033[105m',
            'purple': '\033[95m',
            'cyan': '\033[96m',
            'grey': '\033[90m',
            'blue': '\033[94m',
            'green': '\033[92m',
            'yellow': '\033[93m',
            'red': '\033[91m',
            'red_bg': '\033[41m',
            'bold': '\033[1m'
    }[mode]
    print(prefix+s+'\033[0m')

# Helper function
def proc_dep(dep):
    lines = dep.split('\n')[9:]
    tokens = [t.split('\t')[1] for t in lines]
    postags = [t.split('\t')[3] for t in lines]
    nertags = [t.split('\t')[-1] for t in lines]
    deprels = [t.split('\t')[7] for t in lines]
    parents = [int(t.split('\t')[6])-1 for t in lines]
    nums = [int(t.split('\t')[0])-1 for t in lines]
    return lines, tokens, postags, nertags, deprels, parents, nums

# Helper function
def proc_prefixes(dep):
    prefixes = dep.split('\n')[:9]
    twid = prefixes[0].split('# id = ')[1]
    verified = prefixes[-5].split('# verified = ')[1]=='True'
    followers = int(prefixes[-6].split('# followers = ')[1])
    place = prefixes[-3].split('# name = ')[1]
    pop = int(prefixes[-2].split('# population =')[1])
    pol = float(prefixes[-1].split('# county =')[1])
    txt = prefixes[1].split('# text = ')[1]
    uid = int(prefixes[2].split('# user = ')[1])    
    return prefixes, twid, verified, followers, place, pop, pol, txt, uid

# For a given list of nodes, get the root of the subtree they are contained by
def find_local_root(dep, indices):
    lines, tokens, postags, nertags, deprels, parents, nums = proc_dep(dep)
    potential_roots = set([i for i in indices])
    for i in indices:
        if parents[i] in indices:
            potential_roots.remove(i)
    if len(potential_roots) == 1:
        return list(potential_roots)[0]
    distance_dict = defaultdict(list)
    for i in potential_roots:
        curr = i
        distance = 0
        while deprels[curr] not in ['root', 'parataxis']:
            distance += 1
            curr = parents[curr]
        distance_dict[distance].append(i)
    potential_roots = set(distance_dict[min(distance_dict)])
    return list(potential_roots)[0]

# Get the span of tokens indices for the target's name using POS tags, flat relations, and NER tags.
def get_names(dep, lname='fauci'):
    punct_set = set(string.punctuation+"‘’“”")
    lines, tokens, postags, nertags, deprels, parents, nums = proc_dep(dep)
    persons = [j for i,j in zip(tokens,nums) if i.lower()==lname]
    name_enc = [0 for i in tokens]
    for fi, f in enumerate(persons):
        curr = f
        exclude = set()
        while deprels[curr] == 'conj' and deprels[parents[curr]]=='flat':
            curr = parents[curr]
            exclude.add(parents[curr])
            if deprels[curr+1]=='cc':
                exclude.add(curr+1)
        while deprels[curr] == 'flat':
            curr = parents[curr]
        for x, xi in zip(nertags[:curr], nums[:curr]):
            if 'B-PER' == x and '@USER'!=tokens[xi]:
                if set(nertags[xi+1:curr+1]) == set(['I-PER']) and 'vocative' not in set(deprels[xi:curr]):
                    curr = xi
                    break
        for i,j in enumerate(lines[:min(len(lines),f+1)]):
            if i in range(curr, f+1) and i not in exclude:
                name_enc[i] = fi+1

    encoding = {}
    for i in range(1,max(name_enc)+1):
        if len([j for j,k in enumerate(name_enc) if k==i]) == 0:    # idk why i needed this
            continue
        encoding[i] = {
            'name': [j for j,k in enumerate(name_enc) if k==i],
            'name_root': find_local_root(dep, [j for j,k in enumerate(name_enc) if k==i])
        }
    return encoding

# Strict high precision rule-based gendered pronoun coreference.
def get_coref(d, assumed_gender='masc'):
    enc = get_names(d)
    lines, tokens, postags, nertags, deprels, parents, nums = proc_dep(d)
    attributed_coref = set()
    new_coref = []
    for f in enc:
        if 'vocative' in [deprels[i] for i in enc[f]['name']]:
            valid_pron = ['you', 'yours', 'yourself']
        else:
            if assumed_gender == 'masc':
                valid_pron = ['he', 'him', 'his', 'himself']
            elif assumed_gender == 'fem':
                valid_pron = ['she', 'her', 'hers', 'herself']
            elif assumed_gender == 'neuter':
                valid_pron = ['they', 'them', 'their', 'themself']
        for i,j in enumerate(lines):
            if i in enc[f]['name']:
                pass
            elif i in attributed_coref:
                pass
            elif tokens[i].lower() in valid_pron and i>max(enc[f]['name']):
                if 'you' in valid_pron:
                    oops = [k for k, n in enumerate(nertags[:i]) if 'PER' in n or deprels[k]=='vocative']                    
                else:
                    oops = [k for k, n in enumerate(nertags[:i]) if 'PER' in n and deprels[k]!='vocative']
                if len(oops) == 0 or all([ii in enc[f]['name'] for ii in oops]):
                    new_coref.append(i)
                    attributed_coref.add(i)
    for i in set(new_coref):
        enc[len(enc) + 1] = {
            'name': [i],
            'name_root': i
        }
    return enc

def procner(tag):
    if tag == 'O':
        return tag
    else:
        return tag.split('-')[1]

# Get conjuncts of any name span and expand them as if they were name spans
def get_conjuncts(d):
    enc = get_coref(d)
    lines, tokens, postags, nertags, deprels, parents, nums = proc_dep(d)
    alloc_spans = set([i for j in enc for i in enc[j]['name']])
    for f in enc:
        outgoing = [parents[j] for j in enc[f]['name'] if parents[j] not in alloc_spans and deprels[j]=='conj']
        incoming = [k for i,k in zip(deprels, nums) if i=='conj' and parents[k] in enc[f]['name']+outgoing and k not in alloc_spans|set(outgoing)]
        enc[f]['conj_roots'] = sorted(list(set(outgoing+incoming)))
        enc[f]['conjs'] = {}
        enc[f]['mc'] = {}
        enc[f]['all_conjs'] = []
        for cr in enc[f]['conj_roots']:
            span = set()
            queue = [cr]
            while len(queue):
                top = queue.pop(0)
                span.add(top)
                children = [i for i in nums if i not in span and parents[i]==top and deprels[i]=='flat']
                for c in children:
                    if c not in span|alloc_spans:
                        queue.append(c)
            span = sorted(list(span))

            # Leftward expansion
            if nertags[span[0]].startswith('I'):
                tag = nertags[span[0]].split('-')[1]
                curr = span[0]
                for x, xi in zip(nertags[:span[0]], nums[:span[0]]):
                    if 'B-'+tag == x and '@USER'!=tokens[xi]:
                        if set(nertags[xi+1:curr+1]) == set([f'I-{tag}']) and 'vocative' not in set(deprels[xi:curr]):
                            curr = xi
                            break
                span = list(range(curr, span[0])) + span

            # Rightward expansion
            if not nertags[span[-1]]=='O':
                tag = nertags[span[-1]].split('-')[1]
                curr = span[-1]
                for x, xi in zip(nertags[span[-1]:], nums[span[-1]:]):
                    if 'I-'+tag == x and '@USER'!=tokens[xi]:
                        if set(nertags[curr+1:xi+1]) == set(['I-'+tag]) and 'vocative' not in set(deprels[curr:xi]):
                            curr = xi
                            break
                span = span + list(range(span[-1]+1, curr+1))
            
            enc[f]['mc'][cr] = Counter([procner(nertags[i]) for i in span]).most_common()[0][0]
            enc[f]['conjs'][cr] = span
            enc[f]['all_conjs'] += span
    return enc

def get_appos(d):
    enc = get_conjuncts(d)
    lines, tokens, postags, nertags, deprels, parents, nums = proc_dep(d)
    for f in enc:
        outgoing_appos = [parents[j] for j in enc[f]['name'] if parents[j] not in enc[f]['name'] and deprels[j]=='appos' and deprels[parents[j]] not in ['appos', 'list']]
        incoming_appos = [k for i,k in zip(deprels, nums) if i=='appos' and parents[k] in enc[f]['name'] and k not in enc[f]['name'] and tokens[k] != '@USER' and deprels[parents[k]] not in ['appos', 'list']]
        appos = outgoing_appos+incoming_appos
        fin = [a for a in appos]
        #for a in appos:
        #    conjucts = [i for i in nums if parents[i]==a and deprels[i]=='conj']
        #    for c in conjucts:
        #        fin.append(c)
        enc[f]['appos_roots'] = sorted(list(set(fin)))
    return enc

def get_titles(d):
    enc = get_appos(d)
    lines, tokens, postags, nertags, deprels, parents, nums = proc_dep(d)
    for f in enc:
        enc[f]['title_roots'] = []
        for i,j,k in zip(nums, parents, deprels):
            if i not in enc[f]['title_roots']:
                if j in enc[f]['name']+enc[f]['appos_roots']:
                    if k=='compound':
                        if tokens[i] not in ['@USER', 'HTTPURL']:
                            enc[f]['title_roots'].append(i)
                            conjucts = [ii for ii in nums if parents[ii]==i and deprels[ii]=='conj']
                            #for c in conjucts:
                            #    enc[f]['title_roots'].append(c)
                """
                if i in enc[f]['name']+enc[f]['appos_roots']:
                    if k == 'flat':
                        if j not in enc[f]['name']+enc[f]['appos_roots']+enc[f]['title_roots']:
                            if tokens[j] not in ['@USER', 'HTTPURL']:
                                enc[f]['title_roots'].append(j)
                                conjucts = [ii for ii in nums if parents[ii]==j and deprels[ii]=='conj']
                                for c in conjucts:
                                    enc[f]['title_roots'].append(c)
                """
    return enc

def get_amods(d):
    enc = get_titles(d)
    lines, tokens, postags, nertags, deprels, parents, nums = proc_dep(d)
    for f in enc:
        enc[f]['amod_roots'] = []
        for i,j,k in zip(nums, parents, deprels):
            if j in enc[f]['name']+enc[f]['appos_roots']+enc[f]['title_roots']:
                if k == 'amod':
                    enc[f]['amod_roots'].append(i)
                    conjucts = [ii for ii in nums if parents[ii]==i and deprels[ii]=='conj']
                    #for c in conjucts:
                    #    enc[f]['amod_roots'].append(c)
    return enc

def get_nmods(d):
    enc = get_amods(d)
    lines, tokens, postags, nertags, deprels, parents, nums = proc_dep(d)
    for f in enc:
        enc[f]['nmod_roots'] = []
        for i,j,k in zip(nums, parents, deprels):
            if j in enc[f]['name']:#+enc[f]['appos_roots']+enc[f]['title_roots']:
                if k == 'nmod':
                    enc[f]['nmod_roots'].append(i)
                    conjucts = [ii for ii in nums if parents[ii]==i and deprels[ii]=='conj']
                    #for c in conjucts:
                    #    enc[f]['nmod_roots'].append(c)
    return enc

def get_preds(d):
    enc = get_nmods(d)
    lines, tokens, postags, nertags, deprels, parents, nums = proc_dep(d)
    for f in enc:
        enc[f]['padj_roots'] = []
        enc[f]['pnom_roots'] = []
        enc[f]['nom_roots'] = []
        enc[f]['verb_roots'] = []
        enc[f]['used_obl'] = []
        for i,j,k,p in zip(nums, parents, deprels, postags):
            if i in enc[f]['name']+enc[f]['appos_roots']+enc[f]['title_roots']:
                if deprels[i] in ['nsubj']:
                    queue = [j]
                    conjucts = [ii for ii in nums if parents[ii]==j and deprels[ii]=='conj']
                    #for c in conjucts:
                    #    queue.append(c)
                    for jj in queue:
                        if postags[jj] == 'ADJ':
                            enc[f]['padj_roots'].append(jj)
                        elif postags[jj] in ['PROPN', 'NOUN']:
                            enc[f]['pnom_roots'].append(jj)
                            for l in nums:
                                if parents[l]==jj and postags[l]=='ADJ' and deprels[l]=='amod':
                                    enc[f]['amod_roots'].append(l)
                                    conjucts = [ii for ii in nums if parents[ii]==l and deprels[ii]=='conj']
                                    #for c in conjucts:
                                    #    enc[f]['amod_roots'].append(c)
                        elif postags[jj] in ['VERB']:
                            enc[f]['verb_roots'].append(jj)
                if deprels[i] in ['root', 'parataxis']:
                    for l in nums:
                        if parents[l]==i and postags[l] in ['PROPN', 'NOUN'] and deprels[l]=='nsubj':
                            queue = [l]
                            conjucts = [ii for ii in nums if parents[ii]==l and deprels[ii]=='conj']
                            #for c in conjucts:
                            #    queue.append(c)
                            for jj in queue:
                                if postags[jj] in ['PROPN', 'NOUN']:
                                    enc[f]['nom_roots'].append(jj)
                                    for l in nums:
                                        if parents[l]==jj and postags[l]=='ADJ' and deprels[l]=='amod':
                                            enc[f]['amod_roots'].append(l)
                                            conjucts = [ii for ii in nums if parents[ii]==l and deprels[ii]=='conj']
                                            #for c in conjucts:
                                            #    enc[f]['amod_roots'].append(c)     
                if deprels[i] in ['obl']:
                    for l in nums:
                        if parents[i]==l and postags[l] in ['VERB']:
                            if any([k for k in nums if deprels[k]=='nsubj:pass' and parents[k]==l]):
                                enc[f]['verb_roots'].append(l)
                                enc[f]['used_obl'].append(i)
    return enc

def get_poss(d):
    enc = get_preds(d)
    lines, tokens, postags, nertags, deprels, parents, nums = proc_dep(d)
    for f in enc:
        enc[f]['poss_roots'] = []
        for i,j,k in zip(nums, parents, deprels):
            if i in enc[f]['name']+enc[f]['appos_roots']+enc[f]['title_roots']:
                if k in ['nmod:poss']:
                    if j not in enc[f]['name']+enc[f]['appos_roots']+enc[f]['title_roots']:
                        enc[f]['poss_roots'].append(j)
                        conjucts = [ii for ii in nums if parents[ii]==j and deprels[ii]=='conj']
                        #for c in conjucts:
                        #    enc[f]['poss_roots'].append(c)
    return enc

def get_args(d):
    enc = get_poss(d)
    lines, tokens, postags, nertags, deprels, parents, nums = proc_dep(d)
    for f in enc:
        enc[f]['obj_roots'] = []
        enc[f]['objs'] = []
        for i,j,k in zip(nums, parents, deprels):
            if i in enc[f]['name']+enc[f]['appos_roots']:
                if k in ['obj', 'obl', 'nsubj:pass']:
                    if k=='obl' and any([i in enc[fi]['used_obl'] for fi in enc]):
                        continue
                    xcomp = []
                    curr = j
                    while deprels[curr] in ['xcomp']:
                        if postags[curr] == 'VERB':
                            xcomp.append(curr)
                            curr = parents[curr]
                        else:
                            break
                    if postags[j] == 'VERB':
                        #verb_seq = sorted(list(set(xcomp+[curr])))
                        verb_seq = [curr]
                        if k in ['obl']:
                            try:
                                verb_seq.append([a for a,b,c in zip(nums, deprels, parents) if b=='case' and c==i][0])
                            except:
                                pass
                        for v in set(verb_seq):
                            enc[f]['obj_roots'].append(v)
                        verb_seqs = [tokens[v].lower() for v in verb_seq if len(tokens[v].strip()) > 0]
                        enc[f]['objs'].append((' '.join(verb_seqs),verb_seq))                    
    return enc

if __name__ == '__main__':
    random.seed(11230)
    saves = []
    crf = []
    if 'coref' in sys.argv:
        with open('../coref/annos') as an:
            for l in an:
                if l.startswith('# id ='):
                    saves.append(l.rstrip('\n').split('# id = ')[1])
        with open('../coref/annos') as an:
            crf = an.read().split('\n\n')
    with open('fauci.conllu') as deps_f:
        deps = deps_f.read().split('\n\n')[:-1]
        #random.shuffle(deps)
        if 'coref-acc' in sys.argv:
            xdeps = []
            for x in tqdm(saves):
                for d in deps:
                    prefixes, twid, verified, followers, place, pop, pol, txt, uid = proc_prefixes(d)
                    if twid == saves[0]:
                        xdeps.append(d)
                        saves = saves[1:]
                        break
            deps = xdeps
        loc_cnt = Counter()
        v_cnt = {}
        fl_cnt = {}
        mc = {}
        flver, fl10k, fl5k, fl1k = 0, 0, 0, 0
        tp,fp,fn = 0,0,0
        histogram_data = Counter()
        for ddi,d in tqdm(enumerate(deps)):
            prefixes, twid, verified, followers, place, pop, pol, txt, uid = proc_prefixes(d)
            lines, tokens, postags, nertags, deprels, parents, nums = proc_dep(d)
            if 'fauci' not in [i.lower() for i in tokens]:
                continue
            if followers > 10000 or verified:
                continue
            if 'dist' in sys.argv:
                print(pol)
                continue
            enc = get_args(d)
            negs = []
            for i in enc:
                for j in enc[i]['title_roots']+enc[i]['appos_roots']+enc[i]['pnom_roots']+enc[i]['amod_roots']+enc[i]['padj_roots']+enc[i]['poss_roots']+enc[i]['verb_roots']+enc[i]['nom_roots']:
                    if any([k for k in nums if parents[k]==j and tokens[k].lower() in ['no', 'nt', 'not', "n't", 'never']]):
                        negs.append(j)
                for jj in enc[i]['objs']:
                    for j in jj[1]:
                        if any([k for k in nums if parents[k]==j and tokens[k].lower() in ['no', 'nt', 'not', "n't", 'never']]):
                            negs.append(j)
            is_a_n = list(set([(tokens[j].lower(),tuple([j])) for i in enc for j in enc[i]['title_roots']+enc[i]['appos_roots']+enc[i]['pnom_roots']+enc[i]['nom_roots']]))
            is_a_adj = list(set([(tokens[j].lower(),tuple([j])) for i in enc for j in enc[i]['amod_roots']+enc[i]['padj_roots']]))
            is_a = is_a_n + is_a_adj
            has_a = list(set([(tokens[j].lower(),tuple([j])) for i in enc for j in enc[i]['poss_roots']]))

            as_agent = list(set([(tokens[j].lower(),tuple([j])) for i in enc for j in enc[i]['verb_roots']]))
            as_agent  = []
            for i in enc:
                for jj in enc[i]['verb_roots']:
                    j = [tokens[jj].lower(), [jj]]
                    if not any([i for i in as_agent if j[0]==i[0] and j[1]==i[1]]):
                        as_agent.append(j)            

            as_patient = []
            for i in enc:
                 for j in enc[i]['objs']:
                    if not any([i for i in as_patient if j[0]==i[0] and j[1]==i[1]]):
                        as_patient.append(j)            
            as_conjunct = list(set([(' '.join([tokens[k] for k in enc[i]['conjs'][j]]).lower(), enc[i]['mc'][j]) for i in enc for j in enc[i]['conj_roots']]))
            punct = string.punctuation.replace(' ', '') 
            if 'coref' in sys.argv:
                for li, l in enumerate(lines):
                    crf_line = crf[ddi].split('\n')[li+2] 
                    if crf_line.endswith('\t1') and any([li in enc[i]['name'] for i in enc]):
                        tp += 1
                    if any([li in enc[i]['name'] for i in enc]):
                        altprint(l, 'purple')
                        if crf_line.endswith('\t0'):
                            fp += 1
                    elif crf_line.endswith('\t1'):
                        fn += 1
                    print(crf_line,nertags[li])
                print(tp, fn, fp)
                input()
                system('clear')
                print(tp, fn, fp)
            if 'full' in sys.argv:
                print(pol, place)
                print(txt)
                print(negs)
                for x in is_a:
                    print(f'> IS_A {x}')
                for x in has_a:
                    print(f'> HAS_A {x}')
                for x in as_agent:
                    print(f'> AS_AGENT {x}')
                for x in as_patient:
                    print(f'> AS_PATIENT {x}')
                for x in as_conjunct:
                    print(f'> AS_CONJUNCT {x}')

                for li, l in enumerate(lines):
                    if any([li in enc[i]['name'] for i in enc]):
                        altprint(l, 'purple')
                    elif any([li in enc[i]['title_roots']+enc[i]['appos_roots'] for i in enc]):
                        altprint(l, 'cyan')
                    elif any([li in enc[i]['amod_roots'] for i in enc]):
                        altprint(l, 'blue')
                    elif any([li in enc[i]['poss_roots'] for i in enc]):
                        altprint(l, 'green')
                    elif any([li in enc[i]['pnom_roots']+enc[i]['padj_roots']+enc[i]['nom_roots'] for i in enc]):
                        altprint(l, 'red')
                    elif any([li in enc[i]['verb_roots'] for i in enc]):
                        altprint(l, 'yellow')
                    elif any([li in enc[i]['obj_roots'] for i in enc]):
                        altprint(l, 'grey')
                    elif any([li in enc[i]['all_conjs'] for i in enc]):
                        altprint(l, 'purple_bg')
                    elif any([li in enc[i]['nmod_roots'] for i in enc]):
                        altprint(l, 'red_bg')
                    else:
                        print(l)
                print()
                input()
            
                system('clear')
            if 'locations' in sys.argv:
                if followers > 10000 or verified:
                    flver += 1
                if followers > 10000:
                    fl10k +=1
                if followers > 5000:
                    fl5k +=1
                if followers > 1000:
                    fl1k += 1
                loc_cnt += Counter([place])
                if place not in v_cnt:
                    v_cnt[place] = [0,0]
                v_cnt[place] = [v_cnt[place][0]+verified, v_cnt[place][1]+1]
                if place not in fl_cnt:
                    fl_cnt[place] = [0,0]
                fl_cnt[place] = [fl_cnt[place][0]+followers, fl_cnt[place][1]+1]
            hist_counter = 0
            lister = []
            if 'out' in sys.argv or 'hist' in sys.argv or 'list' in sys.argv:
                for x in is_a_n:
                    xx = x[0].translate(str.maketrans('', '', punct)).strip(' ')
                    xi = lemma.lemmatize(xx, 'n')
                    if xi in stops:
                        continue
                    if any([i in negs for i in x[1]]):
                        xi = 'not_'+xi
                    cnt['is_a_n'] += Counter([xi])
                    pols['is_a_n'][xi].append(pol)
                    pops['is_a_n'][xi].append(pop)
                    users['is_a_n'][xi].append(uid)
                    hist_counter += 1
                    lister += [f'IS_A({xi})']
                for x in is_a_adj:
                    xx = x[0].translate(str.maketrans('', '', punct)).strip(' ')
                    xi = lemma.lemmatize(xx, 'a')
                    if xi in stops:
                        continue
                    if any([i in negs for i in x[1]]): 
                        xi = 'not_'+xi
                    cnt['is_a_adj'] += Counter([xi])
                    pols['is_a_adj'][xi].append(pol)
                    pops['is_a_adj'][xi].append(pop)
                    users['is_a_adj'][xi].append(uid)
                    hist_counter += 1
                    lister += [f'IS_A({xi})']
                for x in has_a:
                    xx = x[0].translate(str.maketrans('', '', punct)).strip(' ')
                    xi = lemma.lemmatize(xx, 'n')
                    if xi in stops:
                        continue
                    if any([i in negs for i in x[1]]):
                        xi = 'not_'+xi
                    cnt['has_a'] += Counter([xi])
                    pols['has_a'][xi].append(pol)
                    pops['has_a'][xi].append(pop)
                    users['has_a'][xi].append(uid)
                    hist_counter += 1
                for x in as_agent:
                    xx = x[0].translate(str.maketrans('', '', punct)).strip(' ')
                    xi = lemma.lemmatize(xx, 'v')
                    if xi in stops:
                        continue
                    if any([i in negs for i in x[1]]):
                        xi = 'not_'+xi
                    cnt['as_agent'] += Counter([xi])
                    pols['as_agent'][xi].append(pol)
                    pops['as_agent'][xi].append(pop)
                    users['as_agent'][xi].append(uid)
                    hist_counter += 1
                    lister += [f'AS_AGENT({xi})']
                for x in as_patient:
                    xx = x[0].translate(str.maketrans('', '', punct)).strip(' ')
                    if ' ' in xx:                        
                        xi = ' '.join([lemma.lemmatize(i, 'v') for i in xx.split(' ')])
                    else:
                        xi = lemma.lemmatize(xx, 'v') 
                    if xi in stops:
                        continue
                    if any([i in negs for i in x[1]]):
                        xi = 'not_'+xi
                    cnt['as_patient'] += Counter([xi])
                    pols['as_patient'][xi].append(pol)
                    pops['as_patient'][xi].append(pop)
                    users['as_patient'][xi].append(uid)
                    hist_counter += 1
                    lister += [f'AS_PATIENT({xi})']
                for xi in as_conjunct:
                    x,y = xi[0], xi[1]
                    if x == '@user':
                        continue
                    xx = x.translate(str.maketrans('', '', punct)).strip(' ')
                    if xx in stops:
                        continue
                    cnt['as_conjunct'] += Counter([xx])
                    pols['as_conjunct'][xx].append(pol)
                    pops['as_conjunct'][xx].append(pop)
                    users['as_conjunct'][xx].append(uid)
                    if xx not in mc:
                        mc[xx] = Counter()
                    mc[xx] = mc[xx] + Counter([y])
                    hist_counter += 1
                histogram_data = histogram_data + Counter([hist_counter])
            if 'list' in sys.argv:
                print('\t'.join([str(ddi)]+lister))
    if 'locations' in sys.argv:
        print(sum([v_cnt[i][0] for i in v_cnt]))
        print(flver, fl10k, fl5k, fl1k)
        with open('locations.csv', 'w+') as loccsv:
            for i in loc_cnt.most_common():
                loccsv.write('\t'.join([str(i) for i in [i[0], i[1], v_cnt[i[0]][0]/v_cnt[i[0]][1], fl_cnt[i[0]][0]/fl_cnt[i[0]][1]]])+'\n')

    if 'hist' in sys.argv:
        for i in sorted(histogram_data):
            print(i, histogram_data[i])

    if 'out' in sys.argv:
        ttotal = 0
        tunique = 0
        tcritical = 0
        for y in ['is_a_n','is_a_adj','has_a','as_agent','as_patient','as_conjunct']:
            total = 0
            unique = 0
            critical = 0
            with open(f'{y}.tsv', 'w+') as out:
                if y not in 'as_conjunct':
                    for i in cnt[y].most_common():
                        if i[1] == 1:
                            std = 0
                            se = 0
                        else:
                            std = stat.stdev(pols[y][i[0]])
                            se = std / math.sqrt(i[1])
                            if se == 0:
                                tsc = 0
                            else:
                                tsc = (stat.mean(pols[y][i[0]])-22.8) / se
                        out.write('\t'.join([str(xx) for xx in [
                            i[0],
                            i[1],
                            len(set(users[y][i[0]])),
                            stat.mean(pols[y][i[0]]),
                            std,
                            se,
                            tsc                       
                        ]])+'\n')
                        if len(set(users[y][i[0]]))>20 and se:
                            if abs((stat.mean(pols[y][i[0]])-22.8)/se) > 2:
                                critical += 1
                        total += i[1]
                        unique += 1
                else:
                    for i in cnt[y].most_common():
                        if i[1] == 1:
                            std = 0
                            se = 0
                        else:
                            std = stat.stdev(pols[y][i[0]])
                            se = std / math.sqrt(i[1])
                            if se == 0:
                                tsc = 0
                            else:
                                tsc = (stat.mean(pols[y][i[0]])-22.8) / se
                        out.write('\t'.join([str(xx) for xx in [
                            i[0],
                            mc[i[0]].most_common()[0][0],
                            i[1],
                            len(set(users[y][i[0]])),
                            stat.mean(pols[y][i[0]]),
                            std,
                            se,
                            tsc                       
                        ]])+'\n')
                        if len(set(users[y][i[0]]))>20 and se:
                            if abs((stat.mean(pols[y][i[0]])-22.8)/se) > 2:
                                critical += 1
                        total += i[1]
                        unique += 1
            print(y,total, unique, critical)
            ttotal+= total
            tunique += unique
            tcritical += critical
        print('total', ttotal, tunique, tcritical)
