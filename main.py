import nltk
import numpy as np
from nltk.tokenize import word_tokenize
from scipy.stats import pearsonr
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from gensim.models import LdaMulticore
import gensim.corpora as corpora
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

# The normal style of preprocessing stopwords, tokenization, lowering of words and checking non-characters
# returns list of tokens
def pre_process(sentence : str):
    Stopwords = list(set(nltk.corpus.stopwords.words('english')))
    words = word_tokenize(sentence)
    words = [word.lower() for word in words if word.isalpha() and word not in Stopwords] 
    return words

# only difference to first one is returned value is joint to string
def pre_process_to_string(sentence : str):
    Stopwords = list(set(nltk.corpus.stopwords.words('english')))
    words = word_tokenize(sentence)
    words = [word.lower() for word in words if word.isalpha() and word not in Stopwords]
    return ' '.join(words)

# Plots the task 2
def plot_difference_matplotlib(mdiff, title="", annotation=None):
    """Helper function to plot difference between models.
    Uses matplotlib as the backend."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(18, 14))
    data = ax.imshow(mdiff, cmap='RdBu_r', origin='lower')
    plt.title(title)
    plt.colorbar(data)

# Plots all the nxn matrices.
def display_matrix(matrix):
    n = 100
    colors = ["red", "orange", "yellow", "white", "green", "blue", "magenta", "purple" ,"black"]
    custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=256)
    plt.figure(figsize=(12, 10))  # Adjust the figure size for better visibility
    sns.heatmap(matrix, cmap=custom_cmap, square=True, cbar_kws={"shrink": .8, 'ticks': np.arange(0, 1.1, 0.1)},
                annot=False, fmt=".2f", linewidths=.5, vmin=0.05, vmax=1 )

    # Set titles and labels
    plt.title('Similarity matrix (100x100)', fontsize=20)
    plt.xlabel('Chapter Number', fontsize=15)
    plt.ylabel('Chapter Number', fontsize=15)

    # Adjust x and y ticks to show every 10th chapter for clarity
    plt.xticks(ticks=np.arange(0, n, step=10), labels=[str(i) for i in range(1, n+1, 10)], fontsize=10)
    plt.yticks(ticks=np.arange(0, n, step=10), labels=[str(i) for i in range(1, n+1, 10)], fontsize=10)

    # Show the heatmap
    plt.tight_layout()
    plt.show()

def display_readability(matrix) :
    # Gunning fog formula, Flesch score, Fry readability graph, Forecast formula
    n = len(matrix) 
    x_axis_labels = ["Gunning", "Flesch", "Fry", "Forcast"]
    y_axis_labels = ["Gunning", "Flesch", "Fry", "Forcast"]
    colors = ["red", "orange", "yellow", "white", "green", "blue", "magenta", "purple","brown", "black"]
    custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=256)
    plt.figure(figsize=(12, 10))  # Adjust the figure size for better visibility
    sns.heatmap(matrix, cmap=custom_cmap, square=True, cbar_kws={"shrink": .8, 'ticks': np.arange(-1, 1.1, 0.1)},
                annot=True, fmt=".2f", linewidths=.5, vmin=-1, vmax=1, xticklabels=x_axis_labels, yticklabels=y_axis_labels)

    # Set titles and labels
    plt.title('Readability similarity', fontsize=20)
    plt.xlabel('Score', fontsize=15)
    plt.ylabel('Score', fontsize=15)

    # Show the heatmap
    plt.tight_layout()
    plt.show()

def display_indicators(matrix):
    n = len(matrix) 
    x_axis_labels = [str(i+1) for i in range(1, n+1, 1)]
    y_axis_labels = [str(i+1) for i in range(1, n+1, 1)]
    colors = ["red", "orange", "yellow", "white", "green", "blue", "magenta", "purple","brown", "black"]
    custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=256)
    plt.figure(figsize=(12, 10))  # Adjust the figure size for better visibility
    sns.heatmap(matrix, cmap=custom_cmap, square=True, cbar_kws={"shrink": .8, 'ticks': np.arange(0, 1.1, 0.1)},
                annot=False, fmt=".2f", linewidths=.5, vmin=0.05, vmax=1, xticklabels=x_axis_labels, yticklabels=y_axis_labels)

    # Set titles and labels
    plt.title('Indicator Matrix (100x100)', fontsize=20)
    plt.xlabel('Task number', fontsize=15)
    plt.ylabel('Task number', fontsize=15)

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
    print("First table results")
    print("Token count",sum([len(i[3]) for i in contents if "Hell" in i[0]]) / 34)
    print("Token count",sum([len(i[3]) for i in contents if "PURGATORY" in i[0]]) / 33)
    print("Token count",sum([len(i[3]) for i in contents if "PARADISE" in i[0]]) / 33)

    print("Token count",sum([len(i[4]) for i in contents if "Hell" in i[0]]) / 34)
    print("Token count",sum([len(i[4]) for i in contents if "PURGATORY" in i[0]]) / 33)
    print("Token count",sum([len(i[4]) for i in contents if "PARADISE" in i[0]]) / 33)
    
    fig = plt.figure(figsize = (10, 5))
    plt.bar([i[0] for i in contents], [i[2] for i in contents])
    plt.xticks(rotation=90)
    plt.xlabel("Chapter")
    plt.ylabel("Vocab/token ratio")
    plt.show()

    # Task 2
    # ratio common vocab/ overall vocab
    n_m_ratio_matrix = []
    for index_n, n in enumerate(contents) :
        m_ratio_matrix = []
        for index_m, m in enumerate(contents) :
            m_ratio_matrix.append(len(list(n[3] & m[3])) / len(list(n[3] | m[3])))
        n_m_ratio_matrix.append(m_ratio_matrix)
    
    display_matrix(n_m_ratio_matrix)

    # Task 4
    # Empath caterization 

    from empath import Empath

    lexicon = Empath()
    n_m_matrix_empath = [list(lexicon.analyze(i[1], normalize=True).values()) for i in contents]

    # Dispaly the heatmap
    display_matrix(cosine_similarity(n_m_matrix_empath, n_m_matrix_empath))
    
    # Saving the similarity for the task 7
    cosine_similarity_task_4 =cosine_similarity(
        n_m_matrix_empath, 
        n_m_matrix_empath
        )
    # Saves the cosine similarity of Empath
    np.savetxt("Empath_categorization_similarity.csv", cosine_similarity_task_4)
    # Task 5
    
    # Used emotag
    df = pd.read_csv('EmoTag1200-scores.csv')
    # preprocessing of emotags and setting the chapter to right form
    chapters_tokens = [i[4] for i in contents]
    emotag_tokens = [pre_process_to_string(i) for i in df["name"]]
    df["name"] = emotag_tokens
    chapter_string = []

    for content in contents :
        chapter_string.append(pre_process_to_string(content[1].lower()))

    # identifing matchin emotags from chapters
    value_list = []
    for chapter, content in zip(chapter_string, contents):
        _list = []
        for emotag_token in emotag_tokens :
            if emotag_token in chapter :
                value = df.loc[df["name"] == emotag_token][["anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust"]]
                _list.append(value.values[0])
        # calculating the average
        content.append(sum(_list) / len(_list))

    # Calculating the cosine similarity
    similarity_list = [i[5] for i in contents]

    # Saving the similarity for the task 7
    cosine_similarity_task_5 =cosine_similarity(
        similarity_list, 
        similarity_list
        )
    # Saves the cosine similarity of emotag
    np.savetxt("emotag_similarity.csv", cosine_similarity_task_5)
    display_matrix(cosine_similarity(
        similarity_list, 
        similarity_list
        ))
    
    # Task 6
    # Gunning fog formula, Flesch score, Fry readability graph, Forecast formula

    score_list =[]
    from scores import forecast_readability_score, analyze_text, gunning_fog_index, calculate_flesch_reading_ease, fry_readability_index
    for index, content in enumerate(contents) :
        score_vector = []
        text = content[1].lower()
        # calculating Gunning fog formula, Flesch score, Fry readability graph, Forecast formula for each chapter and putting them to list.
        total_words, total_sentences, total_syllables = analyze_text(text)
        score_vector.append(gunning_fog_index(text))
        score_vector.append(calculate_flesch_reading_ease(text))
        score_vector.append(fry_readability_index(total_words, total_sentences, total_syllables))
        score_vector.append(forecast_readability_score(text))
        score_list.append(score_vector)

    norm_score_vector = []
    # normalization of the previous scores
    for vector in score_list :
        _norm_score_vector = []
        for index, value in enumerate(vector) :
            data = [i[index] for i in score_list]
            value = (value - np.min(data)) / (np.max(data) - np.min(data))
            _norm_score_vector.append(value)
        norm_score_vector.append(_norm_score_vector)
    # saves the normalized vectors
    np.savetxt("norm_score_vector.csv", norm_score_vector)

    # Calculating pearson correlation between the scores and displaying it with heatmap
    readability_similarity = []
    for i in range(4) :
        _readability_similarity = []
        for j in range(4) :
            _readability_similarity.append(pearsonr([vector[i] for vector in score_list], [vector[j] for vector in score_list])[0])
        readability_similarity.append(_readability_similarity)
    display_readability(readability_similarity)

    display_matrix(cosine_similarity(
        norm_score_vector, 
        norm_score_vector
        ))
    cosine_similarity_task_6 =cosine_similarity(
        norm_score_vector, 
        norm_score_vector
        )
    # saves the similarity score
    np.savetxt("readability_similarity.csv", norm_score_vector)
    
    # Task 7
    def calculate_pearson_correlation(jaccard_matrix, cosine_matrix):
        """Calculate Pearson correlation coefficient between two similarity matrices."""
        # Generate vector pairs for both matrices
        jaccard_values = jaccard_matrix.flatten()  # Extracting values for comparison
        cosine_values = cosine_matrix.flatten()
        # Calculate the Pearson correlation
        pearson_coefficient, p_value = pearsonr(jaccard_values, cosine_values)
        
        return pearson_coefficient, p_value
    
    # loading the cosine similarities of the past tasks
    cosine_similarity_task_3 = np.loadtxt('topic_similarity_matrix.csv')
    cosine_similarity_tasks = []
    cosine_similarity_tasks.append(n_m_ratio_matrix)
    cosine_similarity_tasks.append(cosine_similarity_task_3)
    cosine_similarity_tasks.append(cosine_similarity_task_4)
    cosine_similarity_tasks.append(cosine_similarity_task_5)
    cosine_similarity_tasks.append(cosine_similarity_task_6)
    task_7 = []
    # the past tasks and calculating the pearson correlation.
    for index_i, i in enumerate(cosine_similarity_tasks) :
        _task_7 = []
        for index_j, j in enumerate(cosine_similarity_tasks) :
            print(index_i, index_j)
            _task_7.append(calculate_pearson_correlation(np.array(i), np.array(j))[0])
        task_7.append(_task_7)
    # displaying the indicator table
    display_indicators(task_7)
    # saving the indicator table
    np.savetxt("indicator_table.csv", task_7)

    # Task 9
    import poesy

    # problems with this library

    # create a Poem object by string
    poem = poesy.Poem("""
    When in the chronicle of wasted time
    I see descriptions of the fairest wights,
    And beauty making beautiful old rhyme
    In praise of ladies dead and lovely knights,
    Then, in the blazon of sweet beauty's best,
    Of hand, of foot, of lip, of eye, of brow,
    I see their antique pen would have express'd
    Even such a beauty as you master now.
    So all their praises are but prophecies
    Of this our time, all you prefiguring;
    And, for they look'd but with divining eyes,
    They had not skill enough your worth to sing:
    For we, which now behold these present days,
    Had eyes to wonder, but lack tongues to praise.
    """)
    
    # poem.summary()
