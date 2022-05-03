# -*- coding: utf-8 -*-
"""
Created on Wed Sep 1 2021

@author: Joe L/Joe HC/Steven H

All the text cleaning functions for data pre-processing.

"""

from typing import Optional, Any, Union
import re
import nltk
from spacy.language import Language
from spacy.tokens.doc import Doc
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from googletrans import Translator
import pathlib as plib

STOP_WORDS = nltk.corpus.stopwords.words('english')


############# Clean all escape characters #############

def standardize_whitespace(text):
    """Replace every instance of whitespace(s) with single space ' '
    and remove leading and trailing spaces in each cell"""

    return re.sub(r'\s+', ' ', text).lstrip().rstrip()


############# Lower case all characters #############

def lowercase(string):
    """All letters in string input will be put into lower case"""

    return string.lower()


############# Generate n-grams #############

def n_grams(word_list, n):
    """
        Runs through a list of words and pairs them sequentially.

        E.g. ['blah','clah','flah'] -> [['blah','clah'],['clah','flah']]
    """

    output = [[word_list[i] for i in range(j, j + n)] for j in range(len(word_list) - n + 1)]

    return output


############# Pandas: select rows through regex #############

def select_rows_regex(sub_df, pattern):
    """
        Use with conditional selection syntax in pandas dataframe. E.g.

            df.loc[select_rows_regex(sub_df,pattern),:]

            or

            df[select_rows_regex(sub_df,pattern)]

        Returns records of a dataframe whose columns, specified by sub_df,
        contains any instance of the regex pattern specified in pattern.
    """

    output = sub_df.applymap(lambda x: re.search(pattern, x) is not None).any(axis=1)

    return output


############# Expanding Contractions #############

def expand_contractions(text: str) -> str:

    contraction_lookup = {
        "n't": 'not',
        "'ve": 'have',
        "'ll": 'will', 
        "'re": 'are', 
        "'d": 'would',
        "'m": 'am'
        }

    matches = re.finditer(r'\w+\'\w{1,2}\W', text)

    substitutions = dict()
    for match in matches:

        temp_text = match.group()
        for contraction, expansion in contraction_lookup.items():
            temp_text = re.sub(
                contraction, 
                f' {expansion}', 
                temp_text)
        
        substitutions.update({match.group(): temp_text})

    output = text
    for original, substitution in substitutions.items():
        output = output.replace(original, substitution)

    return output


############# Tokenizers #############





############# Lemmatize text #############


def compare_raw_lemma(df, row_idx, colname, toksuffix):
    """
    This function is used to review lemmatization:
    It compares lemmatized text from a specific DataFrame
    cell with the non-lemmatized counterpart.
    """
    # non-lemmatized:
    print(df.loc[row_idx, colname])

    # lemmatized - recombined into a single string
    print(' '.join(df.loc[row_idx, colname + toksuffix]))

    return None

def get_list_ngrams(text, cleanup, token, pc_bigrams):
    """
    Fits a vectorizer to the data and finds the most common bigrams to reappend
    :param text: combined petition text
    :param cleanup: cleaner chain
    :param token: tokenizer chain
    :param pc_bigrams: top percentage of bigrams to take
    :return: list of top bigrams
    """
    tfidf_bi = TfidfVectorizer(preprocessor=cleanup.apply,
                            tokenizer=token.apply,
                            ngram_range=(2, 2),
                            sublinear_tf=True)

    tfidf_bi.fit_transform(text)
    tfidf_matrix = tfidf_bi.fit_transform(text)
    scored_bigrams = pd.DataFrame(tfidf_matrix.toarray(), columns = tfidf_bi.get_feature_names())
    scored_bigrams_listed = pd.melt(scored_bigrams, var_name="Bigram", value_name="Score")
    scored_bigrams_sorted = scored_bigrams_listed.groupby('Bigram').Score.sum().reset_index()
    scored_bigrams_sorted.sort_values("Score", inplace=True, ascending=False)
    scored_bigrams_sorted = scored_bigrams_sorted.reset_index()
    five_pc = round(pc_bigrams * scored_bigrams_sorted.shape[0])
    top_five_bigrams = scored_bigrams_sorted['Bigram'][0:five_pc]

    bigram_list = top_five_bigrams.tolist()

    return bigram_list

def bigram_replacer(top, text):
    # to be applied as lambda func replacing ' ' with _ for bigrams
    for bigram in top:
        bigram = str(bigram)
        top_split = bigram.split()
        if len(top_split[0]) > 1 or len(top_split[1]) > 1:
            text = text.replace(bigram, ' ' + top_split[0] + '_' + top_split[1] + ' ')
    return text

TRANSLATOR = Translator()


WELSH_STOPWORDS = []
file = plib.Path(__file__).parent.parent.parent.joinpath('data', 'welsh_stopwords.txt')

with open(file) as f:
    lines = f.readlines()
    for line in lines:
        WELSH_STOPWORDS.append(line[0:-1])
    f.close()

def translate_text(cy_text):
    cy_text = standardize_whitespace(cy_text)
    cy_text = cy_text.lower()
    is_welsh = False
    try:
        is_welsh = (TRANSLATOR.detect(cy_text).lang == 'cy')
    except:
        pass

    if is_welsh:
        cy_tokens = cy_text.split()
        cy_clean = [word for word in cy_tokens if word not in WELSH_STOPWORDS]
        en_clean = []
        for cy_word in cy_clean:
            try:
                en = TRANSLATOR.translate(cy_word, dest='en', src='cy')
                en_word = en.text
                en_clean.append(en_word)

            except:
                pass

        en_text = ' '.join(en_clean)

    else:
        en_text = cy_text

    return en_text