from sentence_transformers import SentenceTransformer
from numpy import ndarray
from ranker import Ranker
import numpy as np
import pickle
from numpy.linalg import norm
from collections import defaultdict


class AuthorRanker_1:
    def __init__(self, 
                 paper_ranker, 
                 docid_to_authorid, 
                 author_data, 
                 PAPER_L2R_RANK_RESULT_PATH) -> None:

        self.paper_ranker = paper_ranker
        self.docid_to_authorid = docid_to_authorid
        self.author_data = author_data
        with open(PAPER_L2R_RANK_RESULT_PATH, 'rb') as f:
            self.rank_result_dict = pickle.load(f)

    def query(self, query):
        author_to_score = defaultdict(float)
        if query in self.rank_result_dict:
            paper_rank_result = self.rank_result_dict[query]
        else:
            paper_rank_result = self.paper_ranker.query(query)
        for (docid, score) in paper_rank_result[:10000]:
            if docid not in self.docid_to_authorid:
                continue
            for authorid in self.docid_to_authorid[docid]:
                author_name = self.author_data[authorid]['name']
                if docid not in self.author_data[authorid]['docid']:
                    continue
                author_order = self.author_data[authorid]['docid'][docid]
                order_factor = 1/(author_order+2)
                author_to_score[author_name] += score*order_factor

        author_rank_result = []
        for author_name in author_to_score:
            score = author_to_score[author_name]
            if score != 0:
                author_rank_result.append((author_name, score))

        author_rank_result.sort(key=lambda x: x[1], reverse=True)

        return author_rank_result

        
class AuthorRanker_2:
    def __init__(self, 
                 paper_ranker, 
                 docid_to_authorid, 
                 docid_to_citation, 
                 authorid_to_author_name,
                 PAPER_L2R_RANK_RESULT_PATH) -> None:

        self.paper_ranker = paper_ranker
        self.docid_to_authorid = docid_to_authorid
        self.docid_to_citation = docid_to_citation
        self.authorid_to_author_name = authorid_to_author_name
        with open(PAPER_L2R_RANK_RESULT_PATH, 'rb') as f:
            self.rank_result_dict = pickle.load(f)

    def query(self, query):
        author_to_score = defaultdict(float)
        if query in self.rank_result_dict:
            paper_rank_result = self.rank_result_dict[query]
        else:
            paper_rank_result = self.paper_ranker.query(query)
        for (docid, score) in paper_rank_result[:10000]:
            if docid not in self.docid_to_authorid:
                continue
            for authorid in self.docid_to_authorid[docid]:
                author_name = self.authorid_to_author_name[authorid]
                if docid not in self.docid_to_citation:
                    continue
                citation_num = self.docid_to_citation[docid]
                author_to_score[author_name] += score*np.log(citation_num+1)

        author_rank_result = []
        for author_name in author_to_score:
            score = author_to_score[author_name]
            if score != 0:
                author_rank_result.append((author_name, score))

        author_rank_result.sort(key=lambda x: x[1], reverse=True)

        return author_rank_result

        
class AuthorRanker_3:
    def __init__(self, 
                 bi_encoder_model_name, 
                 encoded_docs, 
                 docid_to_row_dict, 
                 docid_to_authorid, 
                 authorid_to_author_name, 
                 paper_ranker,
                 PAPER_L2R_RANK_RESULT_PATH) -> None:

        self.model = SentenceTransformer(bi_encoder_model_name, device='cpu')
        self.encoded_docs = encoded_docs
        self.docid_to_row_dict = docid_to_row_dict
        self.docid_to_authorid = docid_to_authorid
        self.authorid_to_author_name = authorid_to_author_name
        self.paper_ranker = paper_ranker
        with open(PAPER_L2R_RANK_RESULT_PATH, 'rb') as f:
            self.rank_result_dict = pickle.load(f)

        
    def query(self, query):
        author_to_agg_vec = {}
        if query in self.rank_result_dict:
            paper_rank_result = self.rank_result_dict[query]
        else:
            paper_rank_result = self.paper_ranker.query(query)
        for (docid, score) in paper_rank_result[:10000]:
            if docid not in self.docid_to_authorid:
                continue
            for authorid in self.docid_to_authorid[docid]:
                author_name = self.authorid_to_author_name[authorid]
                if docid not in self.docid_to_row_dict:
                    continue
                title_vec = self.encoded_docs[self.docid_to_row_dict[docid]]
                if author_name not in author_to_agg_vec:
                    author_to_agg_vec[author_name] = title_vec*score
                else:
                    author_to_agg_vec[author_name] += title_vec*score
        

        query_vec = self.model.encode(query)
        
        author_rank_result = []
        for author_name in author_to_agg_vec:
            author_vec = author_to_agg_vec[author_name]
            score = np.dot(query_vec, author_vec)/(norm(query_vec)*norm(author_vec))
            if score != 0:
                author_rank_result.append((author_name, score))

        author_rank_result.sort(key=lambda x: x[1], reverse=True)
        
        return author_rank_result
        
        
class AuthorRanker_4:
    def __init__(self, 
                 bi_encoder_model_name, 
                 encoded_docs, 
                 docid_to_row_dict, 
                 docid_to_authorid, 
                 authorid_to_author_name, 
                 docid_to_citation) -> None:

        self.model = SentenceTransformer(bi_encoder_model_name, device='cpu')
        self.encoded_docs = encoded_docs
        self.docid_to_row_dict = docid_to_row_dict
        self.docid_to_authorid = docid_to_authorid
        self.authorid_to_author_name = authorid_to_author_name
        self.docid_to_citation = docid_to_citation

        
    def query(self, query):
        author_to_agg_vec = {}
        for docid in self.docid_to_citation:
            if docid not in self.docid_to_authorid:
                continue
            citation_num = self.docid_to_citation[docid]
            for authorid in self.docid_to_authorid[docid]:
                author_name = self.authorid_to_author_name[authorid]
                if docid not in self.docid_to_row_dict:
                    continue
                title_vec = self.encoded_docs[self.docid_to_row_dict[docid]]
                if author_name not in author_to_agg_vec:
                    author_to_agg_vec[author_name] = title_vec*np.log(citation_num+1)
                else:
                    author_to_agg_vec[author_name] += title_vec*np.log(citation_num+1)
        

        query_vec = self.model.encode(query)
        
        author_rank_result = []
        for author_name in author_to_agg_vec:
            author_vec = author_to_agg_vec[author_name]
            score = np.dot(query_vec, author_vec)/(norm(query_vec)*norm(author_vec))
            if score != 0:
                author_rank_result.append((author_name, score))

        author_rank_result.sort(key=lambda x: x[1], reverse=True)
        
        return author_rank_result