# Learning-to-Rank Search Engine for Querying Academic Research Articles and Authors

Our project develops a sophisticated search engine for academic research, aimed at simplifying the discovery of research articles, advisors, and institutions aligned with specific research interests. Building on the initial motivation to bridge the gap in finding suitable academic collaborations, our system uses a rich dataset from the ArnetMiner Citation Network. We've integrated advanced 'Learning to Rank' algorithms, specifically the LGBMRanker, enhanced with TF_IDF related features, extracted citation network features, and cross encoder scores. This approach has led to the successful construction of a system that effectively ranks academic authors and research papers. We propose three distinct models for author ranking, evaluated using both MAP and NCDG metrics. The future vision of this project includes expanding the model to query institutions and research centers, offering a valuable tool for researchers seeking optimal alignment with their research interests.

### Files in codes folder

#### `paper_data_fetching.ipynb`
- Jupyter Notebook for fetching paper data from Google Scholar
#### `author_data_fetching.ipynb`
- Jupyter Notebook for fetching author data from Google Scholar

#### `paper_api.py`
- contains functions for fetching paper data from Google Scholar
#### `profile_api.py`
- contains functions for fetching author data from Google Scholar

#### `main_author.ipynb`
- Jupyter Notebook for author-related processes.

#### `main_paper.ipynb`
- Jupyter Notebook for paper-related processes.

#### `database_extraction.ipynb`
- Jupyter Notebook for preprocessing and indexing process.

#### `network_feature_extraction.ipynb`
- Jupyter Notebook for generating network features from database.

#### `constants.py`
- Contains constant values used across the system.

#### `document_preprocessor.py`
- Preprocesses documents for indexing.

#### `indexing.py`
- Handles the indexing of documents.

#### `l2r.py`
- Learning-to-rank algorithms for paper-level model.

#### `ranker.py`
- General ranker for various IR tasks.

#### `author_ranker.py` 
- Rankers for authors-level based on certain criteria.

#### `relevance.py`
- Assesses the relevance of documents.

### Usage Instructions
0. download dataset and edit data path
- download DBLP+Citation, ACM Citation network data version 12 from http://www.arnetminer.org/citation
- edit your data path at `SCRATCH_PATH` variable in `constants.py`

1. Preprocessing documents
- Run `database_extraction.ipynb` to preprocess and indexing the documents
- Run `network_feature_extraction.ipynb` to generate network feature

2. fetch data from Google Scholar using SERP API
- Edit the research_topics list in `paper_data_fetching.ipynb` and `author_data_fetching.ipynb` to include or exclude topics as needed.
- run `paper_data_fetching.ipynb` and `author_data_fetching.ipynb`

3. IR main notebooks
- run `main_paper.ipynb`, the notebook containing interactive cells to execute tasks related to papers
- run `main_author.ipynb`, the notebook containing interactive cells to execute specific tasks related to authors
