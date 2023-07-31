from collections import Counter, defaultdict
import random
import string
import statistics as stat
import math
from os import system
import nltk
import sys
from nltk.corpus import stopwords

class Pipeline:
    def __init__(self, conll_file, ner_conll=None, pos_conll=None, lname='fauci', assumed_gender='masc', filter_stops=True, stoplist=None, lemmatize=True, lemmatizer=None, assume_intersective_adj=True, assume_intersective_title=False, dep_prefixes=9, ner_prefixes=0, pos_prefixes=0):
        # Config, inputs
        self.lname = lname
        self.assumed_gender = assumed_gender
        self.filter_stops = filter_stops
        if stoplist == None:
            self.stops = set(stopwords.words('english'))
        else:
            self.stops = stoplist
        self.lemmatize = lemmatize
        if lemmatizer == None:
            self.lemma = nltk.wordnet.WordNetLemmatizer()
        else:
            self.lemma = lemmatizer
        self.assume_intersective_adj = True
        self.assume_intersective_title = False
        self.num_prefixes = dep_prefixes
        self.ner_prefixes = ner_prefixes
        self.pos_prefixes = pos_prefixes

        if ner_conll == None and pos_conll == None:
            with open(conll_file,encoding="utf8") as conll:
                self.inputs = conll.read().split('\n\n')[:-1]

            def dep_helper(self, dep):
                lines = dep.split('\n')[self.num_prefixes:]
                tokens = [t.split('\t')[1] for t in lines]
                postags = [t.split('\t')[3] for t in lines]
                nertags = [t.split('\t')[-1] for t in lines]
                deprels = [t.split('\t')[7] for t in lines]
                parents = [int(t.split('\t')[6])-1 for t in lines]
                nums = [int(t.split('\t')[0])-1 for t in lines]
                return lines, tokens, postags, nertags, deprels, parents, nums

            self._helper = dep_helper
        
        elif pos_conll == None:
            with open(conll_file,encoding="utf8") as conll:
                inputs = conll.read().split('\n\n')[:-1]
            with open(ner_conll,encoding="utf8") as ner_conll:
                ners = conll.read().split('\n\n')[:-1]
            self.inputs = zip(inputs, ners)

            def dep_helper(self, dep):
                d,n = dep[0], dep[1]
                lines = d.split('\n')[self.num_prefixes:]
                nerlines = n.split('\n')[self.ner_prefixes:]
                tokens = [t.split('\t')[1] for t in lines]
                postags = [t.split('\t')[3] for t in lines]
                nertags = [t.split('\t')[1] for t in nerlines]
                deprels = [t.split('\t')[7] for t in lines]
                parents = [int(t.split('\t')[6])-1 for t in lines]
                nums = [int(t.split('\t')[0])-1 for t in lines]
                return lines, tokens, postags, nertags, deprels, parents, nums
            self._helper = dep_helper

        else:
            with open(conll_file,encoding="utf8") as conll:
                inputs = conll.read().split('\n\n')[:-1]
            with open(ner_conll,encoding="utf8") as ner_conll:
                ners = conll.read().split('\n\n')[:-1]
            with open(pos_conll,encoding="utf8") as ner_conll:
                poss = conll.read().split('\n\n')[:-1]
            self.inputs = zip(inputs, ners, poss)

            def dep_helper(self, dep):
                d,n,p = dep[0], dep[1], dep[2]
                lines = d.split('\n')[self.num_prefixes:]
                nerlines = n.split('\n')[self.ner_prefixes:]
                poslines = p.split('\n')[self.pos_prefixes:]
                tokens = [t.split('\t')[1] for t in lines]
                postags = [t.split('\t')[1] for t in poslines]
                nertags = [t.split('\t')[1] for t in nerlines]
                deprels = [t.split('\t')[7] for t in lines]
                parents = [int(t.split('\t')[6])-1 for t in lines]
                nums = [int(t.split('\t')[0])-1 for t in lines]
                return lines, tokens, postags, nertags, deprels, parents, nums
            self._helper = dep_helper

    def get_dep_info(self, dep):
        return self._helper(self, dep)

    def get_ner_info(self, tag):
        if tag == 'O':
            return tag
        else:
            return tag.split('-')[1]

    # For a given list of nodes, get the root of the subtree they are contained by
    def find_local_root(self, dep, indices):
        lines, tokens, postags, nertags, deprels, parents, nums = self.get_dep_info(dep)
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
    def process_names(self, dep):
        punct_set = set(string.punctuation+"‘’“”")
        lines, tokens, postags, nertags, deprels, parents, nums = self.get_dep_info(dep)
        persons = [j for i,j in zip(tokens,nums) if i.lower()==self.lname]
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
            if len([j for j,k in enumerate(name_enc) if k==i]) == 0:
                continue
            encoding[i] = {
                'name': [j for j,k in enumerate(name_enc) if k==i],
                'name_root': self.find_local_root(dep, [j for j,k in enumerate(name_enc) if k==i])
            }
        return encoding

    # Strict high precision rule-based gendered pronoun coreference.
    def process_coref(self,enc,d):
        lines, tokens, postags, nertags, deprels, parents, nums = self.get_dep_info(d)
        attributed_coref = set()
        new_coref = []
        for f in enc:
            if 'vocative' in [deprels[i] for i in enc[f]['name']]:
                valid_pron = ['you', 'yours', 'yourself']
            else:
                if self.assumed_gender == 'masc':
                    valid_pron = ['he', 'him', 'his', 'himself']
                elif self.assumed_gender == 'fem':
                    valid_pron = ['she', 'her', 'hers', 'herself']
                elif self.assumed_gender == 'neuter':
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

    # Get conjuncts of any name span and expand them as if they were name spans
    def process_conjuncts(self, enc, d):
        lines, tokens, postags, nertags, deprels, parents, nums = self.get_dep_info(d)
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
                
                enc[f]['mc'][cr] = Counter([self.get_ner_info(nertags[i]) for i in span]).most_common()[0][0]
                enc[f]['conjs'][cr] = span
                enc[f]['all_conjs'] += span
        return enc

    # Placeholder
    def process_appositions(self, enc, d):
        lines, tokens, postags, nertags, deprels, parents, nums = self.get_dep_info(d)
        for f in enc:
            outgoing_appos = [parents[j] for j in enc[f]['name'] if parents[j] not in enc[f]['name'] and deprels[j]=='appos' and deprels[parents[j]] not in ['appos', 'list']]
            incoming_appos = [k for i,k in zip(deprels, nums) if i=='appos' and parents[k] in enc[f]['name'] and k not in enc[f]['name'] and tokens[k] != '@USER' and deprels[parents[k]] not in ['appos', 'list']]
            appos = outgoing_appos+incoming_appos
            fin = [a for a in appos]
            enc[f]['appos_roots'] = sorted(list(set(fin)))
        return enc

    # Placeholder
    def process_titles(self, enc, d):
        lines, tokens, postags, nertags, deprels, parents, nums = self.get_dep_info(d)
        for f in enc:
            enc[f]['title_roots'] = []
            for i,j,k in zip(nums, parents, deprels):
                if i not in enc[f]['title_roots']:
                    if j in enc[f]['name'] or (self.assume_intersective_title and j in enc[f]['appos_roots']):
                        if k=='compound':
                            if tokens[i] not in ['@USER', 'HTTPURL']:
                                enc[f]['title_roots'].append(i)
                                conjucts = [ii for ii in nums if parents[ii]==i and deprels[ii]=='conj']
        return enc

    # Placeholder
    def process_amods(self, enc, d):
        lines, tokens, postags, nertags, deprels, parents, nums = self.get_dep_info(d)
        for f in enc:
            enc[f]['amod_roots'] = []
            for i,j,k in zip(nums, parents, deprels):
                if j in enc[f]['name'] or (self.assume_intersective_adj and j in enc[f]['appos_roots']+enc[f]['title_roots']):
                    if k == 'amod':
                        enc[f]['amod_roots'].append(i)
                        conjucts = [ii for ii in nums if parents[ii]==i and deprels[ii]=='conj']
        return enc

    # Placeholder
    def process_predicates(self, enc, d):
        lines, tokens, postags, nertags, deprels, parents, nums = self.get_dep_info(d)
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
                        for jj in queue:
                            if postags[jj] == 'ADJ':
                                enc[f]['padj_roots'].append(jj)
                            elif postags[jj] in ['PROPN', 'NOUN']:
                                enc[f]['pnom_roots'].append(jj)
                                for l in nums:
                                    if parents[l]==jj and postags[l]=='ADJ' and deprels[l]=='amod':
                                        if self.assume_intersective_adj:
                                            enc[f]['amod_roots'].append(l)
                                            conjucts = [ii for ii in nums if parents[ii]==l and deprels[ii]=='conj']
                            elif postags[jj] in ['VERB']:
                                enc[f]['verb_roots'].append(jj)
                    if deprels[i] in ['root', 'parataxis']:
                        for l in nums:
                            if parents[l]==i and postags[l] in ['PROPN', 'NOUN'] and deprels[l]=='nsubj':
                                queue = [l]
                                conjucts = [ii for ii in nums if parents[ii]==l and deprels[ii]=='conj']
                                for jj in queue:
                                    if postags[jj] in ['PROPN', 'NOUN']:
                                        enc[f]['nom_roots'].append(jj)
                                        for l in nums:
                                            if parents[l]==jj and postags[l]=='ADJ' and deprels[l]=='amod':
                                                if self.assume_intersective_adj:
                                                    enc[f]['amod_roots'].append(l)
                                                    conjucts = [ii for ii in nums if parents[ii]==l and deprels[ii]=='conj']
                    if deprels[i] in ['obl']:
                        for l in nums:
                            if parents[i]==l and postags[l] in ['VERB']:
                                if any([k for k in nums if deprels[k]=='nsubj:pass' and parents[k]==l]):
                                    enc[f]['verb_roots'].append(l)
                                    enc[f]['used_obl'].append(i)
        return enc

    # Placeholder
    def process_possesives(self, enc, d):
        lines, tokens, postags, nertags, deprels, parents, nums = self.get_dep_info(d)
        for f in enc:
            enc[f]['poss_roots'] = []
            for i,j,k in zip(nums, parents, deprels):
                if i in enc[f]['name']:
                    if k in ['nmod:poss']:
                        if j not in enc[f]['name']:
                            enc[f]['poss_roots'].append(j)
                            conjucts = [ii for ii in nums if parents[ii]==j and deprels[ii]=='conj']
        return enc

    # Placeholder
    def process_arguments(self, enc, d):
        lines, tokens, postags, nertags, deprels, parents, nums = self.get_dep_info(d)
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

    # Placeholder
    def post_process(self, enc, d):
        lines, tokens, postags, nertags, deprels, parents, nums = self.get_dep_info(d)
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
        hist_counter = 0
        lister = []
        for x in is_a_n:
            xx = x[0].translate(str.maketrans('', '', punct)).strip(' ')
            if self.lemmatize:
                xi = self.lemma.lemmatize(xx, 'n')
            else:
                xi = xx
            if self.filter_stops and xi in stops:
                continue
            if any([i in negs for i in x[1]]):
                xi = 'not_'+xi
            lister += [(f'IS_A({xi})', x)]
        for x in is_a_adj:
            xx = x[0].translate(str.maketrans('', '', punct)).strip(' ')
            if self.lemmatize:
                xi = self.lemma.lemmatize(xx, 'a')
            else:
                xi = xx
            if self.filter_stops and xi in stops:
                continue
            if any([i in negs for i in x[1]]): 
                xi = 'not_'+xi
            lister += [(f'IS_A({xi})', x)]
        for x in has_a:
            xx = x[0].translate(str.maketrans('', '', punct)).strip(' ')
            if self.lemmatize:
                xi = self.lemma.lemmatize(xx, 'n')
            else:
                xi = xx
            if self.filter_stops and xi in stops:
                continue
            if any([i in negs for i in x[1]]):
                xi = 'not_'+xi
            lister += [(f'HAS_A({xi})', x)]
        for x in as_agent:
            xx = x[0].translate(str.maketrans('', '', punct)).strip(' ')
            if self.lemmatize:
                xi = self.lemma.lemmatize(xx, 'v')
            else:
                xi = xx
            if self.filter_stops and xi in stops:
                continue
            if any([i in negs for i in x[1]]):
                xi = 'not_'+xi
            lister += [(f'AS_AGENT({xi})',x)]
        for x in as_patient:
            xx = x[0].translate(str.maketrans('', '', punct)).strip(' ')
            if ' ' in xx:     
                if self.lemmatize:
                    xi = ' '.join([self.lemma.lemmatize(i, 'v') for i in xx.split(' ')])
                else:
                    xi = xx
            else:
                if self.lemmatize:
                    xi = self.lemma.lemmatize(xx, 'v')
                else:
                    xi = xx
            if self.filter_stops and xi in stops:
                continue
            if any([i in negs for i in x[1]]):
                xi = 'not_'+xi
            lister += [(f'AS_PATIENT({xi})',x)]
        for xi in as_conjunct:
            x,y = xi[0], xi[1]
            if x == '@user':
                continue
            xx = x.translate(str.maketrans('', '', punct)).strip(' ')
            if self.filter_stops and xx in stops:
                continue
            lister += [(f'AS_CONJUNCT({xx})',xi)]
        return lister 
    
    def run(self, entry_num):
        entry = self.inputs[entry_num]
        name_encoding = self.process_names(entry)
        coref_encoding = self.process_coref(name_encoding, entry)
        conjunct_encoding = self.process_conjuncts(coref_encoding, entry)
        apposition_encoding = self.process_appositions(conjunct_encoding, entry)
        title_encoding = self.process_titles(apposition_encoding, entry)
        amod_encoding = self.process_amods(title_encoding, entry)
        predicate_encoding = self.process_predicates(amod_encoding, entry)
        possesive_encoding = self.process_possesives(predicate_encoding, entry)
        argument_encoding = self.process_arguments(possesive_encoding, entry)
        return self.post_process(argument_encoding, entry)