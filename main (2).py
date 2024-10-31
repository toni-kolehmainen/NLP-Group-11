import nltk
import re
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import gensim
from string import punctuation
from nltk import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from gensim.corpora import Dictionary
from nltk.tokenize import word_tokenize
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import pearsonr
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from empath import Empath
import matplotlib.pyplot as plt
from gensim.models import LdaMulticore
import gensim.corpora as corpora
from gensim.models import LdaModel
from sklearn.metrics import jaccard_score
from itertools import combinations
from collections import defaultdict
import spacy
import pandas as pd
import matplotlib.patches as patches

nlp = spacy.load("en_core_web_sm")

# Prints the whole book vocabulary (Task 1)
def book_vocabulary():
    with open("data.txt", "r", encoding='utf-8') as data:
        book_text = data.read()

    entire_book_tokens = word_tokenize(book_text)
    entire_book_vocab = set(entire_book_tokens)
    print(f"Overall Vocabulary Size of the Book: {len(entire_book_vocab)}")

def pre_process(sentence : str):
    Stopwords = list(set(nltk.corpus.stopwords.words('english')))
    words = word_tokenize(sentence)
    words = [word.lower() for word in words if word.isalpha() and word not in Stopwords] 
    return words

# Function to analyze the book by chapter
def analyze_book_by_chapter(chapters):
    # Split the book into chapters using the tab character as delimiter 
    vocab_sizes = []
    token_counts = []
    ratios = []
    chapterdata = []
    # Tokenize the entire book to get the overall vocabulary
    # Process each chapter
    chap_num = 0
    excluded_keywords = {"PARADISE", "Hell", "PURGATORY"}
    for chapter in chapters:

        data = chapter[0].split("\t")
        
        if data[0] in excluded_keywords:
            continue
        tokens = pre_process(data[1].lower())
        
        # Count total tokens in the chapter
        total_tokens = len(tokens)
        # Identify the unique vocabulary (set of unique tokens)
        vocab_size = len(set(tokens))
        
        # Calculate the ratio of vocabulary size to total number of tokens
        ratio = vocab_size / total_tokens if total_tokens != 0 else 0
        
        # Store the results
        vocab_sizes.append(vocab_size)
        token_counts.append(total_tokens)
        ratios.append(ratio)
        chapterdata.append(data)
        
        # Print chapter details
        # print(f"Chapter {chap_num+1}: Vocabulary Size = {vocab_size}, Total Tokens = {total_tokens}, Ratio = {ratio:.4f}")
        chap_num += 1
    return vocab_sizes, token_counts, ratios, chapterdata

def analyse_matrix(chapterdata):
    overlap_matrix = np.zeros((100, 100))
    for i in range(100):
        vocab1 = set(pre_process(chapterdata[i][1].lower()))
        for j in range(100):

            print("processing matrix [{},{}]".format(i,j))
            vocab2 = set(pre_process(chapterdata[j][1].lower()))

            common_vocab = vocab1.intersection(vocab2)

            overall_vocab = vocab1.union(vocab2)

            if overall_vocab:  # Prevent division by zero
                overlap_ratio = len(common_vocab) / len(overall_vocab)
            else:
                overlap_ratio = 0

            overlap_matrix[i, j] = overlap_ratio
    return overlap_matrix

    
    

def plot_ratios(ratios):
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(ratios) + 1), ratios, color='skyblue')
    plt.xlabel('Chapter')
    plt.ylabel('Vocabulary Size / Total Tokens')
    plt.title('Vocabulary Size / Total Tokens Ratio Per Chapter')
    plt.xticks(range(1, len(ratios) + 1), rotation=90)
    plt.yticks([i * 0.1 for i in range(11)])
    plt.ylim(0, 1)  # Set y-axis limits to improve readability
    plt.grid(axis='y')  # Add horizontal grid lines for clarity
    plt.show()

def display_matrix(matrix):
    n = 100
    colors = ["red", "orange", "yellow", "white", "green", "blue", "magenta", "purple","brown", "black"]
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

eng_stopwords = set(stopwords.words('english'))
tokenizer = RegexpTokenizer(r'\s+', gaps=True)
stemmer = PorterStemmer()
translate_tab = {ord(p): u" " for p in punctuation}

def text2tokens(raw_text):
    """Split the raw_text string into a list of stemmed tokens."""
    clean_text = raw_text.lower().translate(translate_tab)
    tokens = [token.strip() for token in tokenizer.tokenize(clean_text)]
    tokens = [token for token in tokens if token not in eng_stopwords]
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return [token for token in stemmed_tokens if len(token) > 2]  # skip short tokens

# For task 3
def gen_LDA_models(chapterdata):
    dataset = [text2tokens(chapter[1]) for chapter in chapterdata]
    dictionary = Dictionary(documents=dataset, prune_at=None)
    dictionary.filter_extremes(no_below=5, no_above=0.3, keep_n=None)
    dictionary.compactify()
    d2b_dataset = [dictionary.doc2bow(doc) for doc in dataset]  # convert list of tokens to bag of word representation

    num_topics = 10
    lda_fst = LdaMulticore(
            corpus=d2b_dataset, num_topics=num_topics, id2word=dictionary,
            workers=4, eval_every=None, passes=10, batch=True,
    )
    lda_snd = LdaMulticore(
        corpus=d2b_dataset, num_topics=num_topics, id2word=dictionary,
        workers=4, eval_every=None, passes=20, batch=True,
    )
    return lda_fst, lda_snd
# For task 3
def plot_difference_matplotlib(mdiff, title="", annotation=None):
    """Helper function to plot difference between models.

    Uses matplotlib as the backend."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(18, 14))
    data = ax.imshow(mdiff, cmap='RdBu_r', origin='lower')
    plt.title(title)
    plt.colorbar(data)
    plt.show()

# Testing
# Apply LDA to each chapter's text and extract topics

def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    words = gensim.utils.simple_preprocess(text, deacc=True)  # Tokenize and clean text
    return words

def apply_lda_to_chapters(chapters, num_topics=3, words_per_topic=8):
    # Process each chapter's text
    processed_texts = [preprocess_text(chapter[1]) for chapter in chapters]  # Preprocess raw text
    
    # Prepare dictionary and corpus
    id2word = corpora.Dictionary(processed_texts)  # Create a dictionary from tokenized words
    corpus = [id2word.doc2bow(text) for text in processed_texts]  # Create a BoW corpus for each chapter
    
    topic_words = []

    # Train LDA model for each chapter
    for chapter_corpus in corpus:
        lda_model = LdaModel([chapter_corpus], num_topics=num_topics, id2word=id2word, passes=15)
        # Extract top words per topic
        topics = lda_model.show_topics(num_topics=num_topics, num_words=words_per_topic, formatted=False)
        topic_words.append([[word for word, _ in topic] for _, topic in topics])
    
    return topic_words

# Calculate similarity between two sets of topics using Jaccard similarity
def calculate_topic_similarity(topic_words):
    num_chapters = len(topic_words)
    similarity_matrix = np.zeros((num_chapters, num_chapters))
    
    # Calculate similarity for each pair of chapters
    for i, j in combinations(range(num_chapters), 2):
        similarities = []
        for topic_i in topic_words[i]:
            for topic_j in topic_words[j]:
                # Calculate Jaccard similarity between two sets of words
                intersection = set(topic_i) & set(topic_j)
                union = set(topic_i) | set(topic_j)
                jaccard_sim = len(intersection) / len(union) if union else 0
                similarities.append(jaccard_sim)
        
        # Average similarity for chapter pair (i, j)
        avg_similarity = np.mean(similarities) if similarities else 0
        similarity_matrix[i, j] = similarity_matrix[j, i] = avg_similarity
    np.fill_diagonal(similarity_matrix, 1)
    
    return similarity_matrix
# For Task 8
def three_most_frequent_terms(chapterdata):
    combined_text = ''.join(chapter[1] for chapter in chapterdata)
    combined_text = re.sub(r'[^\w\s]', '', combined_text.lower())
    words = word_tokenize(combined_text)
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in stop_words]
    word_counts = Counter(filtered_words)
    top_three = word_counts.most_common(3)
    return top_three, word_counts

def get_context_tokens(chapterdata):
    # Define target words
    target_words = {"thou", "thy", "one"}

    # Initialize list to store token context information
    context_data = []

    # Process each chapter in chapterdata
    for i, chapter in enumerate(chapterdata):
        chapter_name = i  # Assuming chapter[0] is the chapter name
        chapter_text = chapter[1]
        
        # Parse text with spaCy
        doc = nlp(chapter_text)

        # Initialize a dictionary to track POS counts for each target word
        pos_count = {word: {'NOUN': 0, 'VERB': 0, 'ADJ': 0, 'ADV': 0, 'PRON': 0, 'PROPN': 0} for word in target_words}

        # Loop through tokens and check for target words
        for token in doc:
            if token.text.lower() in target_words:
                # Extract tokens within a 2-token window
                context_tokens = doc[max(token.i - 2, 0): token.i + 3]
                
                # Count POS tags for the target word
                for t in context_tokens:
                    if t != token and t.pos_ in pos_count[token.text.lower()]:
                        pos_count[token.text.lower()][t.pos_] += 1
        
        # Store the counts as percentage
        for word in target_words:
            total_count = sum(pos_count[word].values())
            if total_count > 0:
                # Calculate percentages
                pos_percentage = {pos: (count / total_count) * 100 for pos, count in pos_count[word].items()}
            else:
                # If no counts, set all to 0%
                pos_percentage = {pos: 0 for pos in pos_count[word]}
            
            context_data.append({
                'chapter': chapter_name,
                'target_word': word,
                'percentages': pos_percentage
            })
    
    return context_data

def plot_stacked_pos_distribution(context_data):
    # Prepare DataFrame for plotting
    pos_data = []
    
    # Extract data into a DataFrame
    for item in context_data:
        chapter = item['chapter']
        target_word = item['target_word']
        percentages = item['percentages']
        
        pos_data.append({
            'chapter': chapter,
            'target_word': target_word,
            'NOUN': percentages['NOUN'],
            'VERB': percentages['VERB'],
            'ADJ': percentages['ADJ'],
            'ADV': percentages['ADV'],
            'PRON': percentages['PRON'],
            'PROPN': percentages['PROPN']
        })
    
    pos_df = pd.DataFrame(pos_data)

    # Filter to keep only the relevant POS columns
    pos_columns = ['NOUN', 'VERB', 'ADJ', 'ADV', 'PRON', 'PROPN']
    filtered_df = pos_df[pos_columns]

    # Create a new DataFrame to plot, keeping all rows intact
    plot_data = filtered_df.copy()
    plot_data['Index'] = np.arange(len(plot_data))  # Create a unique index for each row

    # Set up colors for the bars
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    # Create a stacked bar plot
    fig, ax = plt.subplots(figsize=(15, 8))

    # Create an array for the bottom positions of each bar
    bottoms = np.zeros(len(plot_data))

    # Create bars for each POS tag
    for pos_tag, color in zip(pos_columns, colors):
        ax.bar(plot_data['Index'], plot_data[pos_tag], bottom=bottoms, label=pos_tag, color=color)
        bottoms += plot_data[pos_tag]

    # Set x-axis labels to represent each chapter and word
    chapter_labels = []
    for chapter in range(1,101):  # Assuming 100 chapters
        chapter_labels.extend([f"{chapter} thou", f"{chapter} thy", f"{chapter} one"])

    # Assign custom x-tick labels, displaying only for the second occurrence of each group of three
    ax.set_xticks(plot_data['Index'])
    ax.set_xticklabels([f"{i//3}" if i % 3 == 1 else "" for i in range(len(plot_data))], rotation=90)

    # Add titles and labels
    ax.set_title('POS Value Distribution by Chapter (thou, thy, one)', fontsize=16)
    ax.set_xlabel('Chapter, Word(thou, thy, one)', fontsize=14)
    ax.set_ylabel('Percentage', fontsize=14)
    ax.legend(title='POS Tags', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Draw rectangles around every three bars for outlining
    for i in range(0, len(plot_data), 3):
        rect = patches.Rectangle((i - 0.5, 0), 3, 100, linewidth=1, edgecolor='black', facecolor='none')
        ax.add_patch(rect)

    # Show grid and plot
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

if __name__=='__main__':
    # Task 1
    contents= []

    book_vocabulary()
    # creates 2d list that contains title of the chapter and text
    with open("data.txt", "r", encoding='utf-8') as data:
        for content in data.readlines() :
            contents.append(content.split("\n"))

            
    # Analyze the book by chapters
    vocab_sizes, token_counts, ratios, chapterdata = analyze_book_by_chapter(contents)
    # Plot the vocabulary size / total tokens ratio for each chapter
    plot_ratios(ratios)

    # Task 2
    # Run these for the matrix file (takes about 1 min to run)
    # matrix = analyse_matrix(chapterdata)
    # np.savetxt("chapter_matrix.csv", matrix)

    loaded_matrix = np.loadtxt('chapter_matrix.csv')
    display_matrix(loaded_matrix)

    # Task 3
    # Run these for topic similarity matrix (takes about 5 seconds)
    # topics = apply_lda_to_chapters(chapterdata)
    # matrix = calculate_topic_similarity(topics)
    # np.savetxt("topic_similarity_matrix3.csv", matrix)
    
    topic_matrix = np.loadtxt('topic_similarity_matrix.csv')
    display_matrix(topic_matrix)

    #Task 8 
    three_terms, word_counts = three_most_frequent_terms(chapterdata)
    print(three_terms)
    context_data = get_context_tokens(chapterdata)
    #plot_pos_distribution(context_data,"thou")
    #plot_pos_distribution(context_data,"thy")
    #plot_pos_distribution(context_data,"one")
    plot_stacked_pos_distribution(context_data)
