# -*- coding: utf-8 -*-
"""
Created on Wed Sep 1 2021

@author: Joe HC

Classes and functions for utilizing spacy with sklearn and Gensim.
"""

from .child_argument_handlers import \
    ModifierInstanceArgHandler, \
    OptionalBoolCollectionArgHandler, \
    OptionalArgHandler, \
    BasicArgHandler
from .descriptor_modifier import BaseDescriptorModifier
from typing import Optional, Any, Union, Collection
from inspect import isabstract
from abc import ABC, abstractmethod
from collections.abc import Callable
from spacy.language import Language
from spacy.tokens.doc import Doc
from spacy.tokens import Token
from pandas import Series
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from .utils import ChainerCleanList
from ..constants import currency
from scipy.sparse import csr_matrix
from gensim.matutils import Sparse2Corpus
from gensim.corpora import Dictionary

######################
### Descriptor Modifier Classes:

# Descriptor Modifier class for spacy language pipeline modifier arguments
class SpacyModifyPipelineDescriptor(BaseDescriptorModifier):

    def update_object(
        self,
        obj
    ) -> None:
        obj.modify_language()


# Descriptor Modifier class for extensions to the spacy language pipeline
class SpacyExtendPipelineDescriptor(BaseDescriptorModifier):

    def update_object(
        self,
        obj
    ) -> None:
        obj.construct_pipeline()


# Descriptor Modifier class for spacy language pipeline arguments
# that don't require _modifier component
class SpacyNoModifierDescriptor(SpacyExtendPipelineDescriptor):

    def __init__(
        self
    ):
        pass

    def __set_name__(self, owner, name):
        self.private_name = '_' + name

    def update_object(
        self,
        obj
    ) -> None:
        return obj

    def __set__(self, obj, value) -> None:
        """
        This method returns a specific class from a collection of child classes.
        Allows for multiple options to be passed to a class without breaking SOLID principles.
        """
        setattr(obj, self.private_name, value)

######################
### Modifier Classes - Used to customize each component on the preprocessing pipeline:

### Custom Tokenizers
class CustomTokenizer(ABC):
    """
    All tokenizers must inherit this class to be used with spaCy's pretrained pipelines.

    All tokenizers must have __init__ and __call__ methods defined.
    """
    @abstractmethod
    def __init__():
        pass

    @abstractmethod
    def __call__():
        pass


class SpacyRegexTokenizer(CustomTokenizer):

    """
    Custom Tokenizer class, compatible with spacy's
    pre-trained pipelines.
    """

    def __init__(
        self, 
        language: Language, 
        pattern: str = r'\w+(-\w+)?(\'\w+)?'
        ):

        self.pattern = pattern
        self.vocab = language.vocab

    def __call__(self, text: str) -> Doc:

        tokens = re.finditer(self.pattern, text)
        tokens = [el.group() for el in tokens]

        return Doc(self.vocab, tokens)


######################

### Spacy Language Pipeline Modifiers
class BaseSpacyModifier(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __call__(
        language_model: Language, 
        ) -> Language:
        pass


class DefaultModifier(BaseSpacyModifier):
    _key = 'default'

    def __init__(
        self, 
        custom_tokenizer: Any = None
        ):
        pass

    def __call__(
        self,
        language_model: Language,
        ) -> Language:
        """
        Returns language_model unmodified
        """
        return language_model

######################
# Tokenize modifier

class TokenModifier(BaseSpacyModifier):
    _key = 'CustomTokenizer'
    
    def __init__(
        self, 
        custom_tokenizer: CustomTokenizer, 
        ):
        self.custom_tokenizer = custom_tokenizer

    def __call__(
        self,
        language_model: Language,
        ) -> Language:
        """
        Passes a custom tokizer to the spacy language model.
        """
        language_model.tokenizer = self.custom_tokenizer

        return language_model


######################
### Stopwords modifier

class BaseSpacyStopWords(ABC):
    @abstractmethod
    def __init__(
        self, 
        stop_words
        ):
        pass

    @abstractmethod
    def __call__(
        self,
        language_model: Language, 
        ) -> Language:
        pass


class NoStopWords(BaseSpacyStopWords):
    _key = 'False'
    
    def __init__(
        self,
        stop_words: None = None
        ):
        self.stop_words = stop_words

    def __call__(
        self,
        language_model: Language, 
        ) -> Language:
        """
        Removes all stopwords from language model
        """
        language_model.Defaults.stop_words = set()
        
        return language_model


class DefaultStopWords(BaseSpacyStopWords):
    _key = 'True'
    
    def __init__(
        self, 
        stop_words: bool
        ):
        self.stop_words = stop_words

    def __call__(
        self,
        language_model: Language, 
        ) -> Language:
        """
        Does not alter the model
        """
        return language_model


class CustomStopWords(BaseSpacyStopWords):
    _key = str(Collection)
    
    def __init__(
        self, 
        stop_words: set[str]
        ):
        self.stop_words = stop_words

    def __call__(
        self,
        language_model: Language
        ) -> Language:
        """
        Adds custom set of stopwords to language model
        """
        language_model.Defaults.stop_words = self.stop_words
        
        return language_model


######################
### spacy ner and token extraction

class BaseSpacyNer(ABC):
    def __init__(
        self,
        arg: Any
    ):
        pass

    @abstractmethod
    def __call__(
        self,
        doc: Doc,
        ) -> list[Token]:
        pass


class DefaultNer(BaseSpacyNer):
    _key = 'default'
    
    def __call__(
        self,
        doc: Doc,
        ) -> list[Token]:
        """"No NER protocol, return a list of all tokens from doc"""
        with doc.retokenize() as retokenizer:
            for ent in doc.ents:
                if ent.label_ in {'CARDINAL', 'PERCENT'}:
                    retokenizer.merge(doc[ent.start: ent.end])

                if ent.label_ == 'MONEY':
                    temp = range(
                        ent.start, 
                        ent.end 
                    )
                    temp = [
                        i - 1 for i in temp 
                        if i > 0 
                        and  doc[i - 1].text in currency 
                        and doc[i].pos_ == 'NUM'
                    ]
                    for i in temp:
                        retokenizer.merge(doc[i: i + 2])

        output = [
            token for token in doc
            if (not token.is_stop)
            and (token.pos_ != 'PUNCT')
            and (token.dep_ != 'punct')
            and len(token) > 1
        ]

        return output


######################

### spacy text/lemma extractions
class BaseSpacyTokenExtraction(ABC):
    def __init__(
        self,
        arg: Any
    ):
        pass
    
    @abstractmethod
    def __call__(
        self,
        token_list: list[Token],
        ) -> list[str]: 
        pass


class TextExtraction(BaseSpacyTokenExtraction):
    _key = 'False'

    def __call__(
        self,
        token_list: list[Token],
        ) -> list[str]: 
        """Returns token text, minus stopwords"""
        output = [
            token.text for token in token_list
        ]
        return output


class LemmaExtraction(BaseSpacyTokenExtraction):
    _key = 'True'

    def __call__(
        self,
        token_list: list[Token],
        ) -> list[str]:
        """Returns token lemma, minus stopwords"""
        output = [
            token.lemma_ for token in token_list 
        ]
        return output


######################

### spacy lowercase
class BaseSpacyLowerCase(ABC):
    def __init__(
        self,
        arg: Any
    ):
        pass

    @abstractmethod
    def __call__(
        self,
        text_list: list[str],
        ) -> list[str]: 
        pass


class LowerCaseFalse(BaseSpacyLowerCase):
    _key = 'False'

    def __call__(
        self,
        text_list: list[str],
        ) -> list[str]: 
        return text_list


class LowerCaseTrue(BaseSpacyLowerCase):
    _key = 'True'

    def __call__(
        self,
        text_list: list[str],
        ) -> list[str]: 
        return [str.lower(el) for el in text_list]


######################

### fit/transform modifiers for sklearn

class BaseGensimModifier(ABC):
    def __init__(
        self,
        arg: Any
    ):
        pass

    @abstractmethod
    def __call__(
        self,
        output: csr_matrix
    ) -> csr_matrix: 
        pass
    

class DefaultOutput(BaseGensimModifier):
    _key = 'False'
    
    def __call__(
        self,
        output: csr_matrix
    ) -> csr_matrix:
        return output
    

class GensimCorpus(BaseGensimModifier):
    _key = 'True'
    
    def __call__(
        self,
        output: csr_matrix
    ) -> Sparse2Corpus:
        return Sparse2Corpus(output, documents_columns = False)

######################
### modifier class
class SpacyModifyPipeline:

    custom_tokenizer = SpacyModifyPipelineDescriptor(
        BaseSpacyModifier,
        ModifierInstanceArgHandler(CustomTokenizer),
        'Custom Tokenizer not recognised. Tokenizer must be a class that inherits CustomTokenizer.'
    )

    stop_words = SpacyModifyPipelineDescriptor(
        BaseSpacyStopWords,
        OptionalBoolCollectionArgHandler(), 
        'stop_words argument not recognised. Must be either None, a boolean or a set of strings.'
    )

    def __init__(
        self,
        spacy_model: Language, 
        custom_tokenizer: Optional[CustomTokenizer] = None,
        stop_words: Union[bool, Collection[str], None] = None
    ):
        self.spacy_model = spacy_model
        self.custom_tokenizer = custom_tokenizer
        self.stop_words = stop_words
        
        self.modify_language()

    def modify_language(self):
        modifier = ChainerCleanList(
            self.__dict__.get('_custom_tokenizer_modifier'),
            self.__dict__.get('_stop_words_modifier')
        )
        self.spacy_model = modifier(self.spacy_model)

    def __call__(
        self,
        documents
    ):
        return self.spacy_model(documents)


### spacy modifier class
class SpacyExtendedPipeline(SpacyModifyPipeline):

    preproc_function = SpacyNoModifierDescriptor()

    ner = SpacyExtendPipelineDescriptor(
        BaseSpacyNer,
        OptionalArgHandler(str),
        'NER argument not recognized.'
    )

    lemmatize = SpacyExtendPipelineDescriptor(
        BaseSpacyTokenExtraction,
        BasicArgHandler(bool),
        'lemmatize argument not recognized.'
    )

    lowercase = SpacyExtendPipelineDescriptor(
        BaseSpacyLowerCase,
        BasicArgHandler(bool),
        f'lowercase argument not recognized.'
    )

    def __init__(
        self,
        spacy_model: Language, 
        preproc_function: Optional[Callable[[str], str]] = None,
        lowercase: bool = True, 
        custom_tokenizer: Optional[CustomTokenizer] = None,
        stop_words: Union[bool, Collection[str], None] = None,
        ner: Optional[str] = None,
        lemmatize: bool = False
    ):
        super().__init__(
            spacy_model, 
            custom_tokenizer, 
            stop_words
        )
        self.preproc_function = preproc_function
        self.lowercase = lowercase
        self.ner = ner
        self.lemmatize = lemmatize
        
        self.modify_language()

    def construct_pipeline(self):
        self.pipeline = ChainerCleanList(
            self.__dict__.get('_preproc_function'),
            self.spacy_model,
            self.__dict__.get('_ner_modifier'),
            self.__dict__.get('_lemmatize_modifier'),
            self.__dict__.get('_lowercase_modifier')
        )

    def modify_language(self):
        super().modify_language()
        self.construct_pipeline()

    def __call__(
        self,
        documents: Series
    ):
        return documents.apply(self.pipeline)


### Spacy Extended pipeline with ngram processsing available (no custom tools implemented as yet)
class SpacyNgrams(SpacyExtendedPipeline):

    def __init__(
        self,
        spacy_model: Language, 
        preproc_function: Optional[Callable[[str], str]] = None,
        lowercase: bool = True, 
        custom_tokenizer: Optional[CustomTokenizer] = None,
        stop_words: Union[bool, Collection[str], None] = None,
        ner: Optional[str] = None,
        lemmatize: bool = False,
        ngrams: Optional[callable] = None
    ):
        super().__init__(
            spacy_model = spacy_model, 
            preproc_function = preproc_function,
            lowercase = lowercase, 
            custom_tokenizer = custom_tokenizer,
            stop_words = stop_words,
            ner = ner,
            lemmatize = lemmatize
        )
        self.ngrams = ngrams

    def __call__(self, documents: Series):
        output = super().__call__(documents)

        if self.ngrams:
            output = self.ngrams(output)

        return output


### Spacy to Gensim Bag of Words
class SpacyGensimBOW(SpacyNgrams):
    def __call__(self, documents: Series):
        output = super().__call__(documents)

        self.texts = output
        self.dictionary = Dictionary(output)

        return [self.dictionary.doc2bow(record) for record in output]


### Tfidf Vectorizer child with Spacy language processor as the analyzer function
class SpacyTfidf(TfidfVectorizer): 

    custom_tokenizer = SpacyModifyPipelineDescriptor(
        BaseSpacyModifier,
        ModifierInstanceArgHandler(CustomTokenizer),
        'Custom Tokenizer not recognised. Tokenizer must be a class that inherits CustomTokenizer.'
    )

    stop_words = SpacyModifyPipelineDescriptor(
        BaseSpacyStopWords,
        OptionalBoolCollectionArgHandler(), 
        'stop_words argument not recognised. Must be either None, a boolean or a set of strings.'
    )

    preproc_function = SpacyNoModifierDescriptor()

    ner = SpacyExtendPipelineDescriptor(
        BaseSpacyNer,
        OptionalArgHandler(str),
        'NER argument not recognized.'
    )

    lemmatize = SpacyExtendPipelineDescriptor(
        BaseSpacyTokenExtraction,
        BasicArgHandler(bool),
        'lemmatize argument not recognized.'
    )

    lowercase = SpacyExtendPipelineDescriptor(
        BaseSpacyLowerCase,
        BasicArgHandler(bool),
        f'lowercase argument not recognized.'
    )

    gensim_output = SpacyExtendPipelineDescriptor(
        BaseGensimModifier,
        BasicArgHandler(bool),
        f'Gensim modifier argument not recognized.'
    )

    def __init__(
        self,
        spacy_model: Language, 
        input: str = 'content',
        encoding: str = 'utf-8', 
        decode_error: str = 'strict',
        preproc_function: Optional[Callable[[str], str]] = None,
        lowercase: bool = True, 
        custom_tokenizer: Optional[CustomTokenizer] = None,
        stop_words: Union[bool, Collection[str], None] = None,
        ner: Optional[str] = None,
        lemmatize: bool = False,
        #ngram_range: tuple[int, int] = (1, 1), 
        max_df = 1.0, 
        min_df = 1,
        max_features = None,
        dtype = np.float64, 
        norm = 'l2', 
        use_idf = True, 
        smooth_idf = True, 
        sublinear_tf = False,
        gensim_output: bool = False
    ):
    
        self.spacy_model = spacy_model
        self.custom_tokenizer = custom_tokenizer
        self.stop_words = stop_words
        self.preproc_function = preproc_function
        self.lowercase = lowercase
        self.ner = ner
        self.lemmatize = lemmatize
        
        self.modify_language()
        
        # Initialize Tfidf Vectorizer
        super().__init__(
            input = input,
            encoding = encoding, 
            decode_error = decode_error,
            analyzer = self.analyzer,
            max_df = max_df, 
            min_df = min_df,
            max_features = max_features,
            dtype = dtype, 
            norm = norm, 
            use_idf = use_idf, 
            smooth_idf = smooth_idf, 
            sublinear_tf = sublinear_tf,
        )

        self.gensim_output = gensim_output 

    def construct_pipeline(self):
        self.analyzer = ChainerCleanList(
            self.__dict__.get('_preproc_function'),
            self.spacy_model,
            self.__dict__.get('_ner_modifier'),
            self.__dict__.get('_lemmatize_modifier'),
            self.__dict__.get('_lowercase_modifier')
        )

    def modify_language(self):
        modifier = ChainerCleanList(
            self.__dict__.get('_custom_tokenizer_modifier'),
            self.__dict__.get('_stop_words_modifier')
        )
        self.spacy_model = modifier(self.spacy_model)
        self.construct_pipeline()

    def transform(self, raw_documents):
        return self._gensim_output_modifier(super().transform(raw_documents))


    def fit_transform(self, raw_documents):
        super().fit(raw_documents)
        return self.transform(raw_documents)