import nltk
import numpy as np
from nltk.corpus import genesis
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import pearsonr
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from gensim.models import LdaMulticore
import gensim.corpora as corpora
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import spacy
from textstat.textstat import textstatistics
import math
import re

def pre_process(sentence : str):
    Stopwords = list(set(nltk.corpus.stopwords.words('english')))
    words = word_tokenize(sentence)
    words = [word.lower() for word in words if word.isalpha() and word not in Stopwords] 
    return words

def pre_process_to_string(sentence : str):
    Stopwords = list(set(nltk.corpus.stopwords.words('english')))
    words = word_tokenize(sentence)
    words = [word.lower() for word in words if word.isalpha() and word not in Stopwords]
    return ' '.join(words)

def plot_difference_matplotlib(mdiff, title="", annotation=None):
    """Helper function to plot difference between models.

    Uses matplotlib as the backend."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(18, 14))
    data = ax.imshow(mdiff, cmap='RdBu_r', origin='lower')
    plt.title(title)
    plt.colorbar(data)

def display_matrix(matrix):
    n = 100
    colors = ["red", "orange", "yellow", "white", "green", "blue", "magenta", "purple" ,"black"]
    custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=256)
    plt.figure(figsize=(12, 10))  # Adjust the figure size for better visibility
    sns.heatmap(matrix, cmap=custom_cmap, square=True, cbar_kws={"shrink": .8, 'ticks': np.arange(0, 1.1, 0.1)},
                annot=False, fmt=".2f", linewidths=.5, vmin=0.05, vmax=1)

    # Set titles and labels
    plt.title('Vocabulary Overlap Matrix (100x100)', fontsize=20)
    plt.xlabel('Chapter Number', fontsize=15)
    plt.ylabel('Chapter Number', fontsize=15)

    # Adjust x and y ticks to show every 10th chapter for clarity
    plt.xticks(ticks=np.arange(0, n, step=10), labels=[str(i) for i in range(1, n+1, 10)], fontsize=10)
    plt.yticks(ticks=np.arange(0, n, step=10), labels=[str(i) for i in range(1, n+1, 10)], fontsize=10)

    # Show the heatmap
    plt.tight_layout()
    plt.show()
if __name__=='__main__':
    contents= []
    chapter_name = ""
    # creates 2d list that contains title of the chapter and text
    with open("data2.txt", "r") as data :
        for content in data.readlines() :
            _data = content.split("\t")
            if "Hell" in _data[0]  :
                chapter_name = "Hell "
            elif "PURGATORY" in _data[0]  :
                chapter_name = "PURGATORY "
            elif "PARADISE" in _data[0]  :
                chapter_name = "PARADISE "
            else :
                _data[0] = chapter_name + _data[0]
                contents.append(_data)

    count_chapters = len([i[0] for i in contents])
    barplot_data = {}
    # Task 1
    for content in contents : 
        vocabulary = pre_process(content[1].lower())
        
        uniq_token = set(vocabulary)
        content.append(len(uniq_token)/len(vocabulary))
        content.append(uniq_token)
        content.append(vocabulary)
    # for content in contents :

    print("Token count",sum([len(i[3]) for i in contents if "Hell" in i[0]]) / 34)
    print("Token count",sum([len(i[3]) for i in contents if "PURGATORY" in i[0]]) / 33)
    print("Token count",sum([len(i[3]) for i in contents if "PARADISE" in i[0]]) / 33)

    print("Token count",sum([len(i[4]) for i in contents if "Hell" in i[0]]) / 34)
    print("Token count",sum([len(i[4]) for i in contents if "PURGATORY" in i[0]]) / 33)
    print("Token count",sum([len(i[4]) for i in contents if "PARADISE" in i[0]]) / 33)
    
    # print(len([i[3] for i in contents]))
    # print([i[0] for i in contents])
    # fig = plt.figure(figsize = (10, 5))
    # plt.bar([i[0] for i in contents], [i[2] for i in contents])
    # plt.xticks(rotation=90)
    # plt.xlabel("Chapter")
    # plt.ylabel("Vocab/token ratio")
    # plt.show()

    # Task 2
    # ratio common vocab/ overall vocab

    # n_m_matrix = [[0]*count_chapters] * count_chapters
    # for index_n, n in enumerate(contents) :
        # for index_m, m in enumerate(contents) :
            # n_m_matrix[index_n][index_m] = len(list(n[3] & m[3])) / len(list(n[3] | m[3]))

    # print(n_m_matrix)

    # Task 3
    # LDA create 3 topics based on 8 words pair of chapters
    def apply_lda_to_chapters(chapters, num_topics=3, words_per_topic=8):
    # Create dictionary and corpus
        id2word = corpora.Dictionary(chapters)
        corpus = [id2word.doc2bow(chapter) for chapter in chapters]
        topic_words = []
        model_list = []
        # Train LDA model for each chapter
        for index, chapter_corpus in enumerate(corpus):
            print(index)
            lda_model = LdaMulticore([chapter_corpus], num_topics=num_topics, id2word=id2word,workers=4, passes=10, eval_every=None, batch=True)
            # Extract top words per topic
            model_list.append(lda_model)
        # n_m_matrix = [[0]*count_chapters] * count_chapters
        n_m_matrix = [[0]*5] * 5

        for x, model_x in enumerate(model_list) :
            for y, model_y in enumerate(model_list) :
                mdiff, annotation = model_x.diff(model_y, distance='jaccard', num_words=words_per_topic)
                _value = 0
                
        #         for i in mdiff :
        #             for j in i :
        #                 _value += j
        #         avg_value = _value / 9
        #         n_m_matrix[y][x] = avg_value
        # print(n_m_matrix)
        # display_matrix(n_m_matrix)
        return topic_words

    # Calculate similarity between two sets of topics using Jaccard similarity
    # def calculate_topic_similarity(topic_words):

    dataset = [i[4] for i in contents]
    # dataset = [i[4] for i in contents[0:5]]
    # print("test", apply_lda_to_chapters(dataset))
    # num_topics = 3
    
    # id2word = corpora.Dictionary(documents=dataset, prune_at=None)
    # corpus = [id2word.doc2bow(text) for text in dataset]

    # lda_fst = LdaMulticore(
    #         corpus=corpus, num_topics=num_topics, id2word=id2word,
    #         workers=4, eval_every=None, passes=10, batch=True,
    #     )
    
    # mdiff, annotation = lda_fst.diff(lda_fst, distance='jaccard', num_words=8)
    # print(mdiff)

    # Task 4
    # cosine similarity

    from empath import Empath
    lexicon = Empath()
    n_m_matrix_empath = [list(lexicon.analyze(i[1], normalize=True).values()) for i in contents]
    # display_matrix(cosine_similarity(n_m_matrix_empath, n_m_matrix_empath))

    # Task 5
    
    # identifing words from chapters
    df = pd.read_csv('EmoTag1200-scores.csv')

    chapters_tokens = [i[4] for i in contents]
    emotag_tokens = [pre_process_to_string(i) for i in df["name"]]
    df["name"] = emotag_tokens
    chapter_string = []

    for content in contents :
        chapter_string.append(pre_process_to_string(content[1].lower()))

    value_list = []
    for chapter, content in zip(chapter_string, contents):
        _list = []
        for emotag_token in emotag_tokens :
            if emotag_token in chapter :
                value = df.loc[df["name"] == emotag_token][["anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust"]]
                _list.append(value.values[0])
        content.append(sum(_list) / len(_list))

    # Calculating the cosine similarity

    # n_m_matrix_empath = [[0]*count_chapters] * count_chapters
    similarity_list = [i[5] for i in contents]
    # print(similarity_list)
    # display_matrix(cosine_similarity(
    #     similarity_list, 
    #     similarity_list
    #     ))
    
    # Task 6
    # Gunning fog formula, Fry readability graph, Forecast formula and Flesch score

    score_list =[]
    from scores import forecast_readability_score, analyze_text, gunning_fog_index, calculate_flesch_reading_ease, fry_readability_index
    for index, content in enumerate(contents) :
        score_vector = []
        text = content[1].lower()

        total_words, total_sentences, total_syllables = analyze_text(text)
        score_vector.append(gunning_fog_index(text))
        score_vector.append(calculate_flesch_reading_ease(text))
        score_vector.append(fry_readability_index(total_words, total_sentences, total_syllables))
        score_vector.append(forecast_readability_score(text))
        score_list.append(score_vector)

    norm_score_vector = []
    
    for vector in score_list :
        _norm_score_vector = []
        for index, value in enumerate(vector) :
            data = [i[index] for i in score_list]
            value = (value - np.min(data)) / (np.max(data) - np.min(data))
            _norm_score_vector.append(value)
        norm_score_vector.append(_norm_score_vector)

    # display_matrix(cosine_similarity(
    #     norm_score_vector, 
    #     norm_score_vector
    #     ))

    # Task 7
    # n_m_matrix_empath, similarity_list, norm_score_vector

    _empath = [tuple(pair) for sub in n_m_matrix_empath for pair in zip(sub, sub[1:])]
    _similarity = [tuple(pair) for sub in similarity_list for pair in zip(sub, sub[1:])]
    _score_vector = [tuple(pair) for sub in norm_score_vector for pair in zip(sub, sub[1:])]

    # Task 8

    # Task 9
    # import poesy

# create a Poem object by string
    # poem = poesy.Poem("""
    # When in the chronicle of wasted time
    # I see descriptions of the fairest wights,
    # And beauty making beautiful old rhyme
    # In praise of ladies dead and lovely knights,
    # Then, in the blazon of sweet beauty's best,
    # Of hand, of foot, of lip, of eye, of brow,
    # I see their antique pen would have express'd
    # Even such a beauty as you master now.
    # So all their praises are but prophecies
    # Of this our time, all you prefiguring;
    # And, for they look'd but with divining eyes,
    # They had not skill enough your worth to sing:
    # For we, which now behold these present days,
    # Had eyes to wonder, but lack tongues to praise.
    # """)
    
    # poem.summary()
    
    
    # for content in contents :
    #     print(content[1])
    #     poem = Poem(content[1])

    #     poem.summary()
 
