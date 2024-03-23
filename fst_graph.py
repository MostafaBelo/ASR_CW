import math
import openfst_python as fst
from language_model import LanguageModel
from minimizations import *

weight_type = 'log'
NLL_ZERO = 1e10

from subprocess import check_call
from IPython.display import Image
def draw_f(f):
    f.draw('tmp.dot', portrait=True)
    check_call(['dot','-Tpng','-Gdpi=600','tmp.dot','-o','tmp.png'])
    return Image(filename='tmp.png')

class WeightGenerator:
    weight_type=weight_type
    lm = LanguageModel()
    
    def __init__(self, weight_type='log'):
        self.weight_type = weight_type
    
    def log_to_prob(self, w):
        return math.e ** (-w)
    def prob_to_log(self, w):
        if w == 1:
            return 0
        elif w == 0:
            return NLL_ZERO
        return -math.log(w)
    
    def defloat(self, w):
        return float(w)
    def enfloat(self, w):
        return fst.Weight(self.weight_type, w)
    
    def prob(self, w):
        w = self.defloat(w)
        if self.weight_type == 'log':
            w = self.log_to_prob(w)
        
        return w
    
    def weighted(self, w):
        if weight_type == 'tropical':
            return w
        
        return self.enfloat(self.prob_to_log(w))
    
    def get_possible(self, f, state):
        w = 0
        for arc in f.arcs(state):
            w += self.prob(float(arc.weight))
            
        if w >= 1:
            return 0
        elif w < 0:
            return 1
        return 1 - w
    
    def get_Zero(self):
        return self.weighted(0)
    def get_One(self):
        return self.weighted(1)
    
    def get_self_loop(self):
        return self.weighted(0.1)
    
    def get_split(self, f, state, N):
        w = self.get_possible(f, state)
        w /= N
        return self.weighted(w)
    
    def get_word_entry(self, word):
        return self.weighted(self.lm.get_start_word_prob(word))
    
    def get_word_mid(self, word):
        return self.weighted(self.lm.get_word_prob(word))
    
    def get_word_exit(self, word):
        notFinal = 1-(self.lm.get_end_word_prob(word))
        notSilence = 1-self.get_silence_split(word)
        return self.weighted(notSilence * notFinal)
    
    def get_word_silence(self, word):
        notFinal = 1-(self.lm.get_end_word_prob(word))
        Silence = self.get_silence_split(word)
        return self.weighted(Silence * notFinal)
    
    def get_silence_split(self, word):
        return .5
    
    def get_word_end(self, word):
        isFinal = self.lm.get_end_word_prob(word)
        return self.weighted(isFinal)
    
    def get_word_entry_in_subset(self, word, words):
        prob1 = self.lm.get_start_word_freq(word)
        prob2 = self.lm.get_start_word_freq(words)
        return self.weighted(prob1 / prob2)
    
    def get_word_mid_in_subset(self, word, words):
        prob1 = self.lm.get_word_freq(word)
        prob2 = self.lm.get_word_freq(words)
        return self.weighted(prob1 / prob2)
    
weighter = WeightGenerator()

def parse_lexicon(lex_file):
    """
    Parse the lexicon file and return it in dictionary form.
    
    Args:
        lex_file (str): filename of lexicon file with structure '<word> <phone1> <phone2>...'
                        eg. peppers p eh p er z

    Returns:
        lex (dict): dictionary mapping words to list of phones
    """
    
    lex = {}  # create a dictionary for the lexicon entries
    with open(lex_file, 'r') as f:
        for line in f:
            line = line.split()  # split at each space
            lex[line[0]] = line[1:]  # first field the word, the rest is the phones
    return lex

lex = parse_lexicon('lexicon.txt')


def generate_symbol_tables(lexicon, n=3):
    '''
    Return word, phone and state symbol tables based on the supplied lexicon
    
    Args:
        lexicon (dict): lexicon to use, created from the parse_lexicon() function
        n (int): number of states for each phone HMM
        
    Returns:
        word_table (fst.SymbolTable): table of words
        phone_table (fst.SymbolTable): table of phones
        state_table (fst.SymbolTable): table of HMM phone-state IDs
    '''
    state_table = fst.SymbolTable()
    phone_table = fst.SymbolTable()
    word_table = fst.SymbolTable()
    
    # your code here
    # .add_symbol('<eps>')
    word_table.add_symbol('<eps>')
    for word in lexicon:
        word_table.add_symbol(word)
        
    phone_table.add_symbol('<eps>')
    for word in list(lexicon):
        for phone in lexicon[word]:
            phone_table.add_symbol(phone)
    phone_table.add_symbol('sil')
    
    state_table.add_symbol('<eps>')
    for word in list(lexicon):
        for phone in lexicon[word]:
            for i in range(n):
                state_table.add_symbol(f"{phone}_{i+1}")
    state_table.add_symbol('sil_1')
    state_table.add_symbol('sil_2')
    state_table.add_symbol('sil_3')
    state_table.add_symbol('sil_4')
    state_table.add_symbol('sil_5')

    return word_table, phone_table, state_table

word_table, phone_table, state_table = generate_symbol_tables(lex)


def generate_phone_wfst(f, start_state, phone, n, word = '', isLast=False):
    """
    Generate a WFST representing an n-state left-to-right phone HMM.
    
    Args:
        f (fst.Fst()): an FST object, assumed to exist already
        start_state (int): the index of the first state, assumed to exist already
        phone (str): the phone label 
        n (int): number of emitting states of the HMM
        
    Returns:
        the final state of the FST
    """
    
    current_state = start_state
    eps = phone_table.find('<eps>')
    out = word_table.find(word)
    
    for i in range(1, n+1):
    
        in_label = state_table.find('{}_{}'.format(phone, i))
        
        next_state = f.add_state()
        f.add_arc(current_state, fst.Arc(in_label, eps, weighter.get_self_loop(), current_state))
        if (isLast and i >= n):
            f.add_arc(current_state, fst.Arc(in_label, out, weighter.get_split(f, current_state, 1), next_state))
        else:
            f.add_arc(current_state, fst.Arc(in_label, eps, weighter.get_split(f, current_state, 1), next_state))
        
        current_state = next_state
    
    return current_state


def generate_word_wfst(f, start_state, word, n, phone_list=None):
    """ Generate a WFST for any word in the lexicon, composed of n-state phone WFSTs.
        This will currently output phone labels.  
    
    Args:
        f (fst.Fst()): an FST object, assumed to exist already
        start_state (int): the index of the first state, assumed to exist already
        word (str): the word to generate
        n (int): states per phone HMM
        
    Returns:
        the constructed WFST
    
    """

    current_state = start_state
    phone_list = lex[word] if phone_list == None else phone_list
    for phone in phone_list[:-1]:
        current_state = generate_phone_wfst(f, current_state, phone, n)
    current_state = generate_phone_wfst(f, current_state, phone_list[-1], n, word, True)
    
    f.set_final(current_state)
    
    return current_state

def add_word(f, n, init_state, start_state, silence_start, silence_end, word, phone_list=None):
    new_start_state = f.add_state()
    f.add_arc(init_state, fst.Arc(state_table.find("<eps>"), phone_table.find("<eps>"), weighter.get_word_entry(word), new_start_state))
    
    f.add_arc(start_state, fst.Arc(state_table.find("<eps>"), phone_table.find("<eps>"), weighter.get_word_mid(word), new_start_state))
    
    last_state = generate_word_wfst(f, new_start_state, word, n, phone_list)
    prob_final = weighter.get_word_end(word)
    f.set_final(last_state, prob_final)
    
    f.add_arc(last_state, fst.Arc(state_table.find("<eps>"), phone_table.find("<eps>"), weighter.get_word_exit(word), start_state))
    f.add_arc(last_state, fst.Arc(state_table.find("<eps>"), phone_table.find("<eps>"), weighter.get_word_silence(word), silence_start))
    
def generate_from_node(f, node, start_state, silence_start, phone_start, n):    
    if node.data == "root":
        return
    elif node.data == "eps":
        word = node.out_lbl
        f.add_arc(phone_start, fst.Arc(state_table.find("<eps>"), phone_table.find("<eps>"), weighter.get_word_exit(word), start_state))
        f.set_final(phone_start, weighter.get_word_end(word))
        return
    
    phone = node.data
    phone_word = node.getLastWord()
    isLast = node.isLast()
    last_state = generate_phone_wfst(f, phone_start, phone, n, phone_word, isLast)
    
    if node.is_branching():
        for child in node.children:
            word = child.words_below
            words = node.words_below
            
            child_transition = f.add_state()
            out_lbl = phone_table.find("<eps>")
            if child.data == "eps":
                out_lbl = word_table.find(child.out_lbl)
            f.add_arc(last_state, fst.Arc(state_table.find("<eps>"), out_lbl, weighter.get_word_mid_in_subset(word, words), child_transition))
            
            generate_from_node(f, child, start_state, silence_start, child_transition, n)
    else:
        if node.out_lbl != "":
            word = node.out_lbl
            f.add_arc(last_state, fst.Arc(state_table.find("<eps>"), phone_table.find("<eps>"), weighter.get_word_exit(word), start_state))
            f.add_arc(last_state, fst.Arc(state_table.find("<eps>"), phone_table.find("<eps>"), weighter.get_word_silence(word), silence_start))
            f.set_final(last_state, weighter.get_word_end(word))
        
        for child in node.children:
            generate_from_node(f, child, start_state, silence_start, last_state, n)

def generate_recog_from_tree(n = 3):
    root = tree_root
    
    f = fst.Fst(weight_type)
    
    # create a single start state
    start_state = f.add_state()
    f.set_start(start_state)
    
    silence_start = f.add_state()
    silence_end = silence_model(f, silence_start)
    sil_symbol = state_table.find("sil_5")
    f.add_arc(silence_end, fst.Arc(sil_symbol, 0, weighter.get_split(f, silence_end, 1), start_state))
    
    for child in root.children:
        word_start = f.add_state()
        word = child.words_below
        f.add_arc(start_state, fst.Arc(state_table.find("<eps>"), phone_table.find("<eps>"), weighter.get_word_mid(word), word_start))
        generate_from_node(f, child, start_state, silence_start, word_start, n)
        
    return f
    
def generate_word_sequence_recognition_wfst(n = 3):
    """ generate a HMM to recognise any sequence of words in the lexicon
    
    Args:
        n (int): states per phone HMM

    Returns:
        the constructed WFST
    
    """
    
    f = fst.Fst(weight_type)
    
    init_state = f.add_state()
    f.set_start(init_state)
    
    # create a single start state
    start_state = f.add_state()
    
    silence_start = f.add_state()
    silence_end = silence_model(f, silence_start)
    sil_symbol = state_table.find("sil_5")
    f.add_arc(silence_end, fst.Arc(sil_symbol, 0, weighter.get_split(f, silence_end, 1), start_state))
    
    N = len(lex.keys())
    
    for word in lex.keys():
        add_word(f, n, init_state, start_state, silence_start, silence_end, word, None)
#         new_start_state = f.add_state()
#         f.add_arc(init_state, fst.Arc(state_table.find("<eps>"), phone_table.find("<eps>"), weighter.get_word_entry(word), new_start_state))
#         f.add_arc(start_state, fst.Arc(state_table.find("<eps>"), phone_table.find("<eps>"), weighter.get_word_mid(word), new_start_state))
#         last_state = generate_word_wfst(f, new_start_state, word, n)
#         f.set_final(last_state)
#         f.add_arc(last_state, fst.Arc(state_table.find("<eps>"), phone_table.find("<eps>"), weighter.get_word_exit(word), start_state))
#         f.add_arc(last_state, fst.Arc(state_table.find("<eps>"), phone_table.find("<eps>"), weighter.get_word_silence(word), silence_start))
        
    
    add_word(f, n, init_state, start_state, silence_start, silence_end, "a", ["ah"])
    add_word(f, n, init_state, start_state, silence_start, silence_end, "the", ["dh", "ah"])
    
    return f


lex = parse_lexicon('lexicon.txt')
word_table, phone_table, state_table = generate_symbol_tables(lex)  # we won't use state_table in this lab

def generate_L_wfst(lex):
    """ Express the lexicon in WFST form
    
    Args:
        lexicon (dict): lexicon to use, created from the parse_lexicon() function
    
    Returns:
        the constructed lexicon WFST
    
    """
    L = fst.Fst()
    
    # create a single start state
    start_state = L.add_state()
    L.set_start(start_state)
    
    for (word, pron) in lex.items():
        
        current_state = start_state
        for (i,phone) in enumerate(pron):
            next_state = L.add_state()
            
            if i == len(pron)-1:
                # add word output symbol on the final arc
                L.add_arc(current_state, fst.Arc(phone_table.find(phone), \
                                                 word_table.find(word), None, next_state))
            else:
                L.add_arc(current_state, fst.Arc(phone_table.find(phone),0, None, next_state))
            
            current_state = next_state
                          
        L.set_final(current_state)
        L.add_arc(current_state, fst.Arc(0, 0, None, start_state))
        
    L.set_input_symbols(phone_table)
    L.set_output_symbols(word_table)                      
    
    return L


def create_wfst(n = 3):
#     f = generate_word_sequence_recognition_wfst(n)
    f = generate_recog_from_tree(n)
    f.set_input_symbols(state_table)
    f.set_output_symbols(word_table)

    return f

def silence_model(f, start_state):
    s1 = start_state    
    s2 = f.add_state()
    s3 = f.add_state()
    s4 = f.add_state()
    s5 = f.add_state()
    
    sil1 = state_table.find("sil_1")
    sil2 = state_table.find("sil_2")
    sil3 = state_table.find("sil_3")
    sil4 = state_table.find("sil_4")
    sil5 = state_table.find("sil_5")
    
    f.add_arc(s1, fst.Arc(sil1, 0, weighter.get_self_loop(), s1))
    f.add_arc(s1, fst.Arc(sil1, 0, weighter.get_split(f, s1, 1), s2))
    
    w = weighter.get_split(f, s2, 3)
    f.add_arc(s2, fst.Arc(sil2, 0, w, s2))
    f.add_arc(s2, fst.Arc(sil2, 0, w, s3))
    f.add_arc(s2, fst.Arc(sil2, 0, w, s4))
    
    w = weighter.get_split(f, s3, 3)
    f.add_arc(s3, fst.Arc(sil3, 0, w, s2))
    f.add_arc(s3, fst.Arc(sil3, 0, w, s3))
    f.add_arc(s3, fst.Arc(sil3, 0, w, s4))
    
    w = weighter.get_split(f, s4, 4)
    f.add_arc(s4, fst.Arc(sil4, 0, w, s2))
    f.add_arc(s4, fst.Arc(sil4, 0, w, s3))
    f.add_arc(s4, fst.Arc(sil4, 0, w, s4))
    f.add_arc(s4, fst.Arc(sil4, 0, w, s5))
    
    f.add_arc(s5, fst.Arc(sil5, 0, weighter.get_self_loop(), s5))
    
    return s5
