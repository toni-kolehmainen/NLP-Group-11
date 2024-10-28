import nltk
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from nltk.corpus import genesis
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import pearsonr
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from empath import Empath
import matplotlib.pyplot as plt
from gensim.models import LdaMulticore
import gensim.corpora as corpora

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
    