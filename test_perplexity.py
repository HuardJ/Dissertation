###############################
########### Imports ###########

import pandas as pd
import spacy
from scripts.functions.text_cleaning import expand_contractions
from scripts.functions.spacy_utils import SpacyTfidf
from gensim.models import LdaModel
from gensim.models.callbacks import PerplexityMetric
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


########### get data ###########
petitions = pd.read_csv(r'data/petitions_sample_text.csv')
petitions.fillna({'action': '', 'background': '', 'additional_details': ''}, inplace = True)
petitions['combined'] = petitions['action'].str.cat(petitions[['background', 'additional_details']], sep = ' ')

########### split combined data ###########

petitions_corpus = petitions['combined'].sample(frac = 0.9, random_state = 42)

hold_out = petitions.loc[
    petitions.index.difference(petitions_corpus.index), 
    'combined'
]

########### text preprocessing setup ###########
nlp = spacy.load('en_core_web_sm') # pre-trained pipeline
vectorizer = SpacyTfidf(
    nlp, 
    preproc_function = expand_contractions, 
    stop_words = True, 
    lemmatize = True,
    gensim_output = True
)

petitions_corpus = vectorizer.fit_transform(petitions_corpus.values)
hold_out = vectorizer.transform(hold_out.values)
print("Petition text vectorized")

########### Topic modelling ###########

perplexity_callback = PerplexityMetric(hold_out, 'shell')

topic_ranges = range(2,21)
model_perplexities = []
for topic_num in topic_ranges:
    print(f'Fitting LDA Model to {topic_num} topics.')

    # train model
    lda = LdaModel(
        petitions_corpus,
        topic_num, 
        {v: k for k, v in vectorizer.vocabulary_.items()}, 
        chunksize = len(petitions_corpus),
        update_every = 1,
        passes = 50,
        # eval_every = 5, # this is pointless because LdaModel defaults eval_every to effectively 1 when chunk size equals corpus size 
        #iterations = 100, # iterations set high to ensure gamma threshold used as stopping method
        callbacks = [perplexity_callback]
    )

    model_perplexities.append(lda.metrics)

print('Training Done')

########### Aggregate Perplexities ###########

# convert perplexity records to df
for topic_num, model in zip(topic_ranges, model_perplexities):
    model['topic_numbers'] = topic_num
model_perplexities = pd.DataFrame(model_perplexities)

# convert list of Perplexity scores to individual rows
model_perplexities.index = model_perplexities.index * 50 # original index values to match first index values of first element in perplexity_series
perplexity_series = model_perplexities.pop('Perplexity')
perplexity_series = perplexity_series.explode(ignore_index = True)
model_perplexities = model_perplexities.join(perplexity_series, how = 'outer') # join on index values

model_perplexities['topic_numbers'].fillna(method = 'ffill', inplace = True) # fill topic_numbers for records with empty rows after join

# add pass number for each row
model_perplexities['passes'] = pd.Series(range(len(model_perplexities))) % 50
model_perplexities['passes'] += 1

# reorder columns
model_perplexities = model_perplexities[['topic_numbers', 'passes', 'Perplexity']]

model_perplexities.to_csv(r'investigation_outputs/bound_perplexities.csv', index = False)

# visualize results
import matplotlib.pyplot as plt

convergence_df = model_perplexities.loc[
    model_perplexities.topic_numbers <= 4
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