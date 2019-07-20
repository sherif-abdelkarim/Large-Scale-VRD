from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.corpus import words

english_vocab = set(w.lower() for w in words.words())
# import enchant
# d = enchant.Dict("en_US")


wnl = WordNetLemmatizer()

# def isenglishword(word):
#     return  isenglishword_wordnet(word) or isenglishword_enchant(word)or word in english_vocab

def isenglishword_wordnet(word):
    return  not not wn.synsets(word)

def get_noun_synsets_wordnet(word):
    noun_synsets = []
    ss = wn.synsets(word, pos=wn.NOUN)
    for ss_i in ss:
        noun_synsets.append(ss_i.name())
    return noun_synsets
        

# def spellcheck_suggest_enchant(word):
#     return d.suggest(word)


# def isenglishword_enchant(word):
#     return d.check(word)

def isplural(word):
    lemma = wnl.lemmatize(word, 'n')
    plural = True if word is not lemma else False
    return plural, lemma

def lemma_to_synset(lemma_keys):
    output = []
    for i in range(0, len(lemma_keys)):
        synseti = wn.lemma_from_key(lemma_keys[i]).synset().name()
        output.append(synseti)
    return output

def lemma_to_synset_one_word(lemma_key):
    try:
        synseti = wn.lemma_from_key(lemma_key).synset().name()
        return synseti
    except:
        return "nil"

def isA (sense1, sense2):
    wn_sense1 =  wn.synset(sense1);
    wn_sense2 =  wn.synset(sense2);
    s_hypernyms = wn_sense1.lowest_common_hypernyms(wn_sense2)
    s_hypernym = s_hypernyms[0]
    if wn_sense1==s_hypernym:
        return 1, 0
    elif wn_sense2==s_hypernym:
        return 1, 1
    else:
        return 0, 0

# print get_noun_synsets_wordnet('dog')

