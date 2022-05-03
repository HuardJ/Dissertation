###############################
########### Imports ###########

from numpy import vectorize
import pandas as pd
import spacy
from scripts.functions.utils import ChainerCleanList
from scripts.functions.text_cleaning import expand_contractions, standardize_whitespace
from scripts.functions.spacy_utils import SpacyGensimBOW
from gensim.models.tfidfmodel import TfidfModel
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel
from scripts.functions.tuning_pipelines import GensimPipeline, TuningPipeline
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


########### get data ###########
petitions = pd.read_csv(r'data/petitions_sample_text.csv')
petitions.fillna({'action': '', 'background': '', 'additional_details': ''}, inplace = True)

petitions['combined'] = petitions['action'].str.cat(
    petitions[['background', 'additional_details']], 
    sep = ' '
)

########### text preprocessing setup ###########
preproc = ChainerCleanList(
    standardize_whitespace,
    expand_contractions
)

nlp = spacy.load('en_core_web_sm') # pre-trained pipeline
preprocessor = SpacyGensimBOW(
    nlp, 
    preproc_function = preproc, 
    stop_words = True, 
    lemmatize = True
)

########### Topic modelling set up ###########

tfidf = TfidfModel(
    smartirs = 'ltc'
)

lda = LdaModel(
        chunksize = 512,
        update_every = 1,
        passes = 50
)

########### Topic modelling pipeline set up ###########

pipeline = GensimPipeline(
    preprocessing = preprocessor, 
    vectorization = tfidf,
    model = lda
)

### hyper parameter set-up
hyperparameters = {
    'preprocessing__stop_words': [True, False],
    'vectorization__smartirs': ['ltc', 'ttc', 'ttx'],
    'model__num_topics': list(range(0, 10, 2))
}

grid_search = TuningPipeline(
    topic_pipeline = pipeline,
    hyperparameter_grid = hyperparameters
)

model_coherences = grid_search.gridsearch(petitions['combined'])

print('Training Done')

########### Aggregate Perplexities ###########
"""
# convert perplexity records to df
model_coherences = pd.DataFrame(model_coherences)

# convert list of Perplexity scores to individual rows
model_coherences.index = model_coherences.index * 50 # original index values to match first index values of first element in perplexity_series
perplexity_series = model_coherences.pop('Perplexity')
perplexity_series = perplexity_series.explode(ignore_index = True)
model_coherences = model_coherences.join(perplexity_series, how = 'outer') # join on index values

model_coherences['topic_numbers'].fillna(method = 'ffill', inplace = True) # fill topic_numbers for records with empty rows after join

# add pass number for each row
model_coherences['passes'] = pd.Series(range(len(model_coherences))) % 50
model_coherences['passes'] += 1

# reorder columns
model_coherences = model_coherences[['topic_numbers', 'passes', 'Perplexity']]

model_coherences.to_csv(r'investigation_outputs/bound_perplexities.csv', index = False)

# visualize results
import matplotlib.pyplot as plt

convergence_df = model_coherences.loc[
    model_coherences.topic_numbers <= 4
]

fig, ax = plt.subplots(figsize = (12, 8))
ax.set_title(
    'Convergence of Topic Numbers 2 to 4',
    fontsize = 16
)
ax.set_ylabel(
    'Perplexity',
    fontsize = 12
)
ax.xaxis.label.set_size(12)

for label, df in convergence_df.groupby('topic_numbers'):
    df.plot(
        x = 'passes',
        y = 'Perplexity',
        label = str(int(label)),
        ax = ax
    )
    
legend = ax.get_legend()
legend.set_title('Topic  Numbers', prop = {'size': 14})
for text in legend.get_texts():
    text.set_fontsize(12)

fig.savefig(r'investigation_outputs/Topic_convergence_examples.png')
"""