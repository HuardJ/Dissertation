###############################
########### Imports ###########

import logging
import pickle
import pandas as pd

from scripts.functions.spacy_utils import SpacyGensimBOW
from scripts.functions.tuning_pipelines import gensim_model_holder

from gensim.models.tfidfmodel import TfidfModel
from gensim.models import LdaModel
from scripts.functions.tuning_pipelines import GensimPipeline

from scripts.functions.model_extractions import GensimTopicModelExtractor

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

path = "configs/base_config.pkl"
if path:
    try:
        with open(path, 'rb') as file:
            config = pickle.load(file = file)

    except FileNotFoundError:
        print('-- WARNING: User config file failed to load.\n')

    else:
        for key, value in config.items():
            config[key] = value
else:
    logging.warning("-- Custom user config were not found.")

logging.info('Loadinging data: data/petitions_sample_text.csv')
########### get data ###########
petitions = pd.read_csv(r'data/petitions_sample_text.csv')
petitions.fillna({'action': '', 'background': '', 'additional_details': ''}, inplace = True)
petitions['combined'] = petitions['action'].str.cat(petitions[['background', 'additional_details']], sep = ' ')

petitions_corpus = petitions['combined']

logging.info("Initializing text processing classes: \n")
########### text preprocessing setup ###########
preprocessor = SpacyGensimBOW(
    **config.get('preprocessor')
)
############## Vectorizer Setup ##############
tfidf = gensim_model_holder(
    TfidfModel, 
    **config.get('vectorization')
)
############## Model Setup ##############
lda = gensim_model_holder(
    LdaModel, 
    **config.get('model')
)

############## Passing to Pipeline ##############

topic_model = GensimPipeline(
    preprocessing = preprocessor, 
    vectorization = tfidf, 
    model = lda
)

logging.info("Text processing classes built. Applying to data:")
############## Running Model ##############

outputs = topic_model.apply_full_pipeline(petitions_corpus)

extractor = GensimTopicModelExtractor(
    texts = petitions[['created_at', 'combined']], 
    topic_model_outputs = outputs, 
    topic_model = topic_model
)

unused = """
pickle_dir = r'data/pickles/{}.pkl'

pet_topics.to_pickle(pickle_dir.format(r'petition_topics'), None)
model_set.doc_topic_weights.to_pickle(pickle_dir.format(r'document_topics_weights'))

with open(pickle_dir.format('model_set'), 'wb') as pkl:
    pickle.dump(model_set, pkl)
pkl.close()

logging.info('Pickles set up')
"""