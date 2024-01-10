SCRACTCH_PATH = "/scratch/si650f23_class_root/si650f23_class/nawatsw/si650"
DATA_VERSION = "v12"
PAPER_DATA_PATH = f"{SCRACTCH_PATH}/dblp.{DATA_VERSION}.json"
PAPER_ABSTRACT_INDEX = f"{SCRACTCH_PATH}/paper{DATA_VERSION}_abstract_index"
PAPER_TITLE_INDEX = f"{SCRACTCH_PATH}/paper{DATA_VERSION}_title_index"
RECOG_CATEGORY_PATH = f'{SCRACTCH_PATH}/recognized_categories.pickle'
DOC_CATEGORY_INFO_PATH = f'{SCRACTCH_PATH}/doc_category_info.pickle'

DOCID_LIST_PATH = f'{SCRACTCH_PATH}/docid_list.pickle'
DOCID_TO_YEAR_RELEASE_PATH = f'{SCRACTCH_PATH}/docid_to_year_release.pickle'
DOCID_TO_AUTHORID_PATH = f'{SCRACTCH_PATH}/docid_to_authorid.pickle'
DOCID_TO_NETWORK_FEATURES_PATH = f'{SCRACTCH_PATH}/docid_to_network_features.pickle'
DOCID_TO_CITATION_PATH = f'{SCRACTCH_PATH}/docid_to_citation.pickle'
DOCID_TO_TITLE_VEC_PATH = f'{SCRACTCH_PATH}/docid_to_title_vec.pickle'
ENCODED_TITLE_ARRAY_PATH = f'{SCRACTCH_PATH}/encoded_title_array.pickle'
DOCID_TO_TITLE_VEC_ROWIDX_PATH = f'{SCRACTCH_PATH}/docid_to_title_vec_rowidx.pickle'

AUTHOR_COLLECTION_PATH = f'{SCRACTCH_PATH}/author_collection.pickle'
AUTHORID_TO_AUTHOR_NAME_PATH = f'{SCRACTCH_PATH}/authorid_to_author_name.pickle'
PAPER_NETWORK_METRICS_PATH = f'{SCRACTCH_PATH}/various_metrics.pickle'

BM25_RANKER_PATH = f'{SCRACTCH_PATH}/BM25Ranker.pickle'
L2R_RANKER_PATH = f'{SCRACTCH_PATH}/l2rRanker.pickle'
L2R_RANKER_FITTED_PATH = f'{SCRACTCH_PATH}/l2rRanker_fitted.pickle'

TRAIN_PAPER_DATA_PATH = 'datasets/train_paper_data.csv'
TEST_PAPER_DATA_PATH = 'datasets/test_paper_data.csv'

TRAIN_AUTHOR_DATA_PATH = 'datasets/train_author_data.csvs'
TEST_AUTHOR_DATA_PATH = 'datasets/test_author_data.csv'
TRAIN_TEST_AUTHOR_DATA_PATH = 'datasets/train_test_author_data.csv'

PAPER_BM25_RANK_RESULT_PATH = f'{SCRACTCH_PATH}/paper_bm25_rank_result.pickle'
PAPER_L2R_RANK_RESULT_PATH = f'{SCRACTCH_PATH}/paper_l2r_rank_result.pickle'
AUTHOR_1_RANK_RESULT_PATH = f'{SCRACTCH_PATH}/author_1_rank_result.pickle'
AUTHOR_2_RANK_RESULT_PATH = f'{SCRACTCH_PATH}/author_2_rank_result.pickle'
AUTHOR_3_RANK_RESULT_PATH = f'{SCRACTCH_PATH}/author_3_rank_result.pickle'
AUTHOR_4_RANK_RESULT_PATH = f'{SCRACTCH_PATH}/author_4_rank_result.pickle'

PAPER_EVAL_RESULT_PATH  = f'{SCRACTCH_PATH}/paper_eval_result.pickle'
AUTHOR_EVAL_RESULT_PATH  = f'{SCRACTCH_PATH}/author_eval_result.pickle'

STOPWORD_PATH = f"{SCRACTCH_PATH}/stopwords.txt"
BIENCODER_MODEL_NAME = 'sentence-transformers/msmarco-MiniLM-L12-cos-v5'
TOTAL_PAPER_COUNT = 4894081
CATEGORIES_COUNT_CUTOFF = 2000