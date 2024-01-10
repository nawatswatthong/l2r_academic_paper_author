"""
This is the template for implementing the rankers for your search engine.
You will be implementing WordCountCosineSimilarity, DirichletLM, TF-IDF, BM25, Pivoted Normalization,
and your own ranker.
"""
import numpy as np
from collections import Counter, defaultdict
from sentence_transformers import CrossEncoder
from indexing import InvertedIndex


class Ranker:
    """
    The ranker class is responsible for generating a list of documents for a given query, ordered by their scores
    using a particular relevance function (e.g., BM25).
    A Ranker can be configured with any RelevanceScorer.
    """
    def __init__(self, index: InvertedIndex, document_preprocessor, stopwords: set[str], 
                 scorer: 'RelevanceScorer', raw_text_dict: dict[int,str]=None) -> None:
        """
        Initializes the state of the Ranker object.

        NOTE: Previous homeworks had you passing the class of the scorer to this function.
            This has been changed as it created a lot of confusion.
            You should now pass an instantiated RelevanceScorer to this function.

        Args:
            index: An inverted index
            document_preprocessor: The DocumentPreprocessor to use for turning strings into tokens
            stopwords: The set of stopwords to use or None if no stopword filtering is to be done
            scorer: The RelevanceScorer object
            raw_text_dict: A dictionary mapping a document ID to the raw string of the document
        """
        self.index = index
        self.tokenize = document_preprocessor.tokenize
        self.scorer = scorer
        self.stopwords = stopwords
        self.raw_text_dict = raw_text_dict

    def query(self, query) -> list[tuple[int, float]]:
        """
        Searches the collection for relevant documents to the query and
        returns a list of documents ordered by their relevance (most relevant first).

        Args:
            query: The query to search for
            pseudofeedback_num_docs: If pseudo-feedback is requested, the number
                 of top-ranked documents to be used in the query,
            pseduofeedback_alpha: If pseudo-feedback is used, the alpha parameter for weighting
                how much to include of the original query in the updated query
            pseduofeedback_beta: If pseudo-feedback is used, the beta parameter for weighting
                how much to include of the relevant documents in the updated query

        Returns:
            A list containing tuples of the documents (ids) and their relevance scores

        NOTE: We are standardizing the query output of Ranker to match with L2RRanker.query and VectorRanker.query
        The query function should return a sorted list of tuples where each tuple has the first element as the document ID
        and the second element as the score of the document after the ranking process.
        """
        query_parts = self.tokenize(query)
        if self.stopwords != None:
            for i, qword in enumerate(query_parts):
                if qword in self.stopwords:
                    query_parts[i] = None
        query_word_counts = Counter(query_parts)

        doc_word_counts_dict = defaultdict(dict)
        for qword in query_word_counts:
            if qword != None and qword in self.index.index:
                for docid, freq in self.index.index[qword]:
                    doc_word_counts_dict[docid][qword] = freq

        results = []
        for docid in doc_word_counts_dict:
            score = self.scorer.score(docid, doc_word_counts_dict[docid], query_word_counts)
            if score != 0:
                results.append((docid, score))
        if len(results) == 0:
            return []
        
        results.sort(key=lambda x: x[1], reverse=True)
    
        return results

class RelevanceScorer:
    """
    This is the base interface for all the relevance scoring algorithm.
    It will take a document and attempt to assign a score to it.
    """
    def __init__(self, index: InvertedIndex, parameters) -> None:
        raise NotImplementedError

    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int]) -> float:
        """
        Returns a score for how relevance is the document for the provided query.

        Args:
            docid: The ID of the document
            doc_word_counts: A dictionary containing all words in the document and their frequencies.
                Words that have been filtered will be None.
            query_word_counts: A dictionary containing all words in the query and their frequencies.
                Words that have been filtered will be None.

        Returns:
            A score for how relevant the document is (Higher scores are more relevant.)

        TODO (HW4): Note that the `query_word_counts` is now a dictionary of words and their counts.
            This is changed from the previous homeworks.
        """
        raise NotImplementedError


class WordCountCosineSimilarity(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters={}) -> None:
        self.index = index
        self.parameters = parameters

    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int]) -> float:
        cos_sim_score = 0
        for qword in query_word_counts:
            if qword == None or query_word_counts[qword] == None or qword not in doc_word_counts:
                continue
            c_wd = doc_word_counts[qword]
            c_qd = query_word_counts[qword]
            cos_sim_score += c_qd * c_wd
        return cos_sim_score

class DirichletLM(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters={'mu': 2000}) -> None:
        self.index = index
        self.parameters = parameters

    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int]) -> float:
        c_count = self.index.get_statistics()['total_token_count']
        d_length = self.index.get_doc_metadata(docid)['length']
        mu = self.parameters['mu']
        dir_score = 0
        for qword in query_word_counts:
            if qword == None or query_word_counts[qword] == None or qword not in doc_word_counts:
                continue
            c_wq = query_word_counts[qword]
            c_wc = self.index.get_term_metadata(qword)['count']
            p_wc = c_wc / c_count
            c_wd = doc_word_counts[qword]
            dir_score += c_wq * np.log(1 + c_wd/(mu*p_wc))
        dir_score += len(query_word_counts)*np.log(mu/(d_length+mu))
        return dir_score

class BM25(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters={'b': 0.75, 'k1': 1.2, 'k3': 8}) -> None:
        self.index = index
        self.b = parameters['b']
        self.k1 = parameters['k1']
        self.k3 = parameters['k3']

    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int])-> float:
        b = self.b
        k1 = self.k1
        k3 = self.k3
        d_count = self.index.get_statistics()['number_of_documents']
        d_length = self.index.get_doc_metadata(docid)['length']
        avdl = self.index.get_statistics()['mean_document_length']
        bm25_score = 0
        for qword in query_word_counts:
            if qword == None or query_word_counts[qword] == None or qword not in doc_word_counts:
                continue
            c_wd = doc_word_counts[qword]
            f_wd = len(self.index.index[qword])
            c_wq = query_word_counts[qword]
            idf = np.log((d_count-f_wd+0.5)/(f_wd+0.5))
            tf = (k1+1)*c_wd/(k1*(1-b+b*d_length/avdl)+c_wd)
            qtf = (k3+1)*c_wq/(k3+c_wq)
            bm25_score += idf * tf * qtf
        return bm25_score

class PersonalizedBM25(RelevanceScorer):
    def __init__(self, index: InvertedIndex, relevant_doc_index: InvertedIndex,
                 parameters={'b': 0.75, 'k1': 1.2, 'k3': 8}) -> None:
        """
        Initializes Personalized BM25 scorer.

        Args:
            index: The inverted index used to use for computing most of BM25
            relevant_doc_index: The inverted index of only documents a user has rated as relevant,
                which is used when calculating the personalized part of BM25
            parameters: The dictionary containing the parameter values for BM25

        Returns:
            The Personalized BM25 score
        """
        self.index = index
        self.relevant_doc_index = relevant_doc_index
        self.b = parameters['b']
        self.k1 = parameters['k1']
        self.k3 = parameters['k3']

    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int]) -> float:
        b = self.b
        k1 = self.k1
        k3 = self.k3
        D_count = self.index.get_statistics()['number_of_documents']
        d_length = self.index.get_doc_metadata(docid)['length']
        avdl = self.index.get_statistics()['mean_document_length']
        R_count = self.relevant_doc_index.get_statistics()['number_of_documents']
        bm25_score = 0
        for qword in query_word_counts:
            if qword == None or query_word_counts[qword] == None or qword not in doc_word_counts:
                continue
            c_wd = doc_word_counts[qword]
            f_wd = self.index.get_term_metadata(qword)['document_count']
            c_wq = query_word_counts[qword]
            try:
                r = self.relevant_doc_index.get_term_metadata(qword)['document_count']
            except:
                r = 0
            idf = np.log((r+0.5)*(D_count-f_wd-R_count+r+0.5)/(f_wd-r+0.5)/(R_count-r+0.5))
            tf = (k1+1)*c_wd/(k1*(1-b+b*d_length/avdl)+c_wd)
            qtf = (k3+1)*c_wq/(k3+c_wq)
            bm25_score += idf * tf * qtf
        return bm25_score

class PivotedNormalization(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters={'b': 0.2}) -> None:
        self.index = index
        self.b = parameters['b']

    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int]) -> float:
        b = self.b
        d_count  = self.index.get_statistics()['number_of_documents']
        d_length = self.index.get_doc_metadata(docid)['length']
        avdl = self.index.get_statistics()['mean_document_length']
        pn_score = 0
        for qword in query_word_counts:
            if qword == None or query_word_counts[qword] == None or qword not in doc_word_counts:
                continue
            c_wd = doc_word_counts[qword]
            f_wd = len(self.index.index[qword])
            c_wq = query_word_counts[qword]
            idf = np.log((d_count+1)/f_wd)
            tf = (1+np.log(1+np.log(c_wd)))/(1-b+b*d_length/avdl)
            qtf = c_wq
            pn_score += idf * tf * qtf
        return pn_score

class TF_IDF(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters={}) -> None:
        self.index = index
        self.parameters = parameters

    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int]) -> float:
        d_count  = self.index.get_statistics()['number_of_documents']
        tfidf_score = 0
        for qword in query_word_counts:
            if qword == None or query_word_counts[qword] == None or qword not in doc_word_counts:
                continue
            c_wd = doc_word_counts[qword]
            f_wd = len(self.index.index[qword])
            tf = np.log(c_wd+1)
            idf = np.log(d_count/f_wd) + 1
            tfidf_score += idf * tf
        return tfidf_score

class CrossEncoderScorer:
    def __init__(self, raw_text_dict: dict[int, str],
                 cross_encoder_model_name: str = 'cross-encoder/msmarco-MiniLM-L6-en-de-v1') -> None:
        """
        Initializes a CrossEncoderScorer object.

        Args:
            raw_text_dict: A dictionary where the document id is mapped to a string with the first 500 words
                in the document
            cross_encoder_model_name: The name of a cross-encoder model

        NOTE 1: The CrossEncoderScorer class uses a pre-trained cross-encoder model
            from the Sentence Transformers package to score a given query-document pair.

        NOTE 2: This is not a RelevanceScorer object because the method signature for score() does not match,
            but it has the same intent, in practice.
        """
        self.model = CrossEncoder(cross_encoder_model_name, max_length=500)
        self.raw_text_dict = raw_text_dict

    def score(self, docid: int, query: str) -> float:
        """
        Gets the cross-encoder score for the given document.
        
        Args:
            docid: The id of the document
            query: The query in its original form (no stopword filtering/tokenization)

        Returns:
            The score returned by the cross-encoder model
        """
        if docid not in self.raw_text_dict.keys():
            return 0
        if query == None or query == "":
            return 0
        score = self.model.predict([[query, self.raw_text_dict[docid]]])[0]
        return score

