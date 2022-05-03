###############################
########### Imports ###########

import pandas as pd
import spacy
from scripts.functions.utils import ChainerCleanList
from scripts.functions.text_cleaning import expand_contractions, standardize_whitespace
from scripts.functions.spacy_utils import SpacyModifyPipeline, SpacyExtendedPipeline, SpacyTfidf

########### get data ###########
petitions = pd.read_csv(r'data/petitions_sample_text.csv')
petitions.fillna({'action': '', 'background': '', 'additional_details': ''}, inplace = True)
petitions['combined'] = petitions['action'].str.cat(petitions[['background', 'additional_details']], sep = ' ')

########### text preprocessing setup ###########
preprocessor = ChainerCleanList(
    standardize_whitespace,
    expand_contractions
)

nlp = spacy.load('en_core_web_sm') # pre-trained pipeline

nlp_modded = SpacyModifyPipeline(
    nlp,
    stop_words = True
)

nlp_extended = SpacyExtendedPipeline(
    nlp,
    preprocessor,
    stop_words = True,
    lemmatize = True
)

petitions_corpus = preprocessor(petitions.at[3, 'combined'])
petitions_corpus = nlp_modded(petitions_corpus)

petitions_token = nlp_extended(petitions.loc[3, 'combined'])


tfidf = SpacyTfidf(
    nlp,
    preproc_function = preprocessor,
    stop_words = True,
    lemmatize = True,
    gensim_output = True
)
petitions_tfidf = tfidf.fit_transform(petitions['combined'])

print("Petition text vectorized")