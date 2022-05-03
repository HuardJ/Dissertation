# -*- coding: utf-8 -*-
"""
Created on Thur Apr 28 2022

@author: Joe HC

Classes to extract and hold key topic modelling  outputs.
"""
from dataclasses import dataclass
from abc import abstractmethod, ABC
from typing import Any, Union, Optional
import pandas as pd
from .tuning_pipelines import AbstractModellingPipeline

@dataclass
class TopicModelOutputs:
    """
    Standard data class, holds key features from topic models
    for use in dashboard.
    """
    feature_names: dict[int, str]
    doc_topic_weights: pd.DataFrame
    topic_term_weights: list[dict[str, float]]
    doc_topic_summary: Optional[pd.DataFrame]
    docs_unclassified: Optional[pd.DataFrame]


class AbstractTopicModelExtractor(ABC):
    @abstractmethod
    def __init__(
        self, 
        topic_model_outputs: dict[str, Any], 
        topic_model: AbstractModellingPipeline
    ):
        pass


class GensimTopicModelExtractor(AbstractTopicModelExtractor):
    """
    Extractor for Gensim based topic models.
    Components are extracted to a TopicModelOutputs class.
    """
    def __init__(
        self, 
        texts: Union[pd.Series, pd.DataFrame], 
        topic_model_outputs: dict[str, Any], 
        topic_model: AbstractModellingPipeline
    ):

        self.texts = texts

        self.feature_names = dict(topic_model.model.id2word)

        self.doc_topic_weights = pd.DataFrame(topic_model_outputs.get('model'))
        self.doc_topic_weights = self.doc_topic_weights.applymap(lambda el: el[-1])

        self.topic_term_weights = topic_model.model.show_topics(
            num_topics = topic_model.model.num_topics, 
            num_words = len(topic_model.model.id2word), 
            formatted = False
        )
        self.topic_term_weights = [
            dict(el[-1]) for el in self.topic_term_weights
        ]

    def get_data_class(self):
        return TopicModelOutputs(
            feature_names = self.feature_names, 
            doc_topic_weights = self.doc_topic_weights, 
            topic_term_weights = self.topic_term_weights, 
            doc_topic_summary = self.texts, 
            docs_unclassified = None
        )