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
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
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
import csv

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

            # Optional: print the overlap ratio for each comparison
            # print(f"Overlap between canto {i} and {j}: {overlap_ratio}")
    return overlap_matrix

def analyse_matrix_with_cosine(chapterdata):
    # Prepare data for vectorization
    chapter_texts = [pre_process(chapter[1].lower()) for chapter in chapterdata]
    chapter_texts = [" ".join(text) for text in chapter_texts]  # Join tokens back into strings

    # Vectorize to get term frequencies
    vectorizer = CountVectorizer()
    term_matrix = vectorizer.fit_transform(chapter_texts)

    # Calculate cosine similarity matrix
    cosine_sim_matrix = cosine_similarity(term_matrix)

    return cosine_sim_matrix

def analyze_within_book_overlap(overlap_matrix, start_index, end_index):
    """
    Calculates the average vocabulary overlap for a single book by analyzing the overlap matrix.

    Parameters:
        overlap_matrix (np.array): 100x100 matrix of vocabulary overlaps for all cantos.
        start_index (int): The starting index of the book within the matrix.
        num_chapters (int): The number of chapters (cantos) in the book.

    Returns:
        float: The average vocabulary overlap within the book.
    """
    # Extract the sub-matrix corresponding to the book
    book_matrix = overlap_matrix[start_index:end_index, start_index:end_index]
    
    # Calculate the average of off-diagonal elements in the sub-matrix
    # Mask the diagonal to exclude self-overlap (which is always 1)
    masked_matrix = np.tril(book_matrix, -1) + np.triu(book_matrix, 1)
    
    # Calculate the mean of non-zero elements
    num_elements = (end_index - start_index) * ((end_index-start_index) - 1)  # Number of off-diagonal elements in a symmetric matrix
    avg_overlap = np.sum(masked_matrix) / num_elements if num_elements != 0 else 0

    return avg_overlap    
    

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

def display_matrix(matrix, name):
    n = 100
    colors = ["red", "orange", "yellow", "white", "green", "blue", "magenta", "purple","brown", "black"]
    custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=256)
    plt.figure(figsize=(12, 10))  # Adjust the figure size for better visibility
    sns.heatmap(matrix, cmap=custom_cmap, square=True, cbar_kws={"shrink": .8, 'ticks': np.arange(0, 1.1, 0.1)},
                annot=False, fmt=".2f", linewidths=.5, vmin=0.05, vmax=1)

    # Set titles and labels
    plt.title((f'{name} (100x100)'), fontsize=20)
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
    vocab = list(id2word.values())
    corpus = [id2word.doc2bow(text) for text in processed_texts]  # Create a BoW corpus for each chapter
    
    
    topic_words = []
    # Train LDA model for each chapter
    for chapter_corpus in corpus:
        lda_model = LdaModel([chapter_corpus], num_topics=num_topics, id2word=id2word, passes=15)
        # Extract top words per topic
        topics = lda_model.show_topics(num_topics=num_topics, num_words=words_per_topic, formatted=False)
        topic_words.append([[word for word, _ in topic] for _, topic in topics])
    
    return topic_words, vocab

# topic vectors for cosine similarity
def create_topic_vectors(topic_words, vocab):
    vectors = []
    vocab_dict = {word: idx for idx, word in enumerate(vocab)}
    
    for topics in topic_words:
        vector = np.zeros(len(vocab_dict))  # Create a vector for the entire vocabulary
        for topic in topics:
            for word in topic:
                if word in vocab_dict:
                    vector[vocab_dict[word]] += 1  # Increase frequency count
        vectors.append(vector)
    
    return np.array(vectors)


# Calculate similarity between two sets of topics using Jaccard similarity
def calculate_jaccard_topic_similarity(topic_words):
    num_chapters = len(topic_words)
    similarity_matrix = np.zeros((num_chapters, num_chapters))
    
    for i, j in combinations(range(num_chapters), 2):
        similarities = []
        for topic_i in topic_words[i]:  # Iterate over topics in chapter i
            for topic_j in topic_words[j]:  # Iterate over topics in chapter j
                # Calculate Jaccard similarity between two sets of words
                intersection = set(topic_i) & set(topic_j)
                union = set(topic_i) | set(topic_j)
                jaccard_sim = len(intersection) / len(union) if union else 0
                similarities.append(jaccard_sim)

        # Average similarity for chapter pair (i, j)
        avg_similarity = np.mean(similarities) if similarities else 0
        similarity_matrix[i, j] = similarity_matrix[j, i] = avg_similarity

    # Fill diagonal with 1 (similarity of a chapter with itself)
    np.fill_diagonal(similarity_matrix, 1)
    
    return similarity_matrix

def combine_similarity_matrices(similarity_matrix_cosine, similarity_matrix_jaccard, alpha=0.5):
    # Ensure the matrices are of the same shape
    assert similarity_matrix_cosine.shape == similarity_matrix_jaccard.shape, "Matrices must have the same dimensions"
    
    # Combine using a weighted average
    combined_similarity = alpha * similarity_matrix_cosine + (1 - alpha) * similarity_matrix_jaccard
    return combined_similarity
# For Task 7
def calculate_pearson_correlation(jaccard_matrix, cosine_matrix):
    """Calculate Pearson correlation coefficient between two similarity matrices."""
    # Generate vector pairs for both matrices

    jaccard_values = jaccard_matrix.flatten()  # Extracting values for comparison
    cosine_values = cosine_matrix.flatten()

    # Calculate the Pearson correlation
    pearson_coefficient, p_value = pearsonr(jaccard_values, cosine_values)
    
    return pearson_coefficient, p_value
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

def get_context_tokens(chapterdata,emotag1200):
    # Define target words
    target_words = {"thou", "thy", "one"}

    # Initialize list to store token context information
    context_data = []
    emotion_context_data = []

    overall_pos_count = {word: {'NOUN': 0, 'VERB': 0, 'ADJ': 0, 'ADV': 0, 'PRON': 0, 'PROPN': 0} for word in target_words}
    overall_emotion_count = {word: {'anger': 0, 'anticipation': 0, 'disgust': 0, 'fear': 0, 'joy': 0, 'sadness': 0, 'surprise': 0, 'trust': 0} for word in target_words}
    # Process each chapter in chapterdata
    for i, chapter in enumerate(chapterdata):
        chapter_name = i  # Assuming chapter[0] is the chapter name
        chapter_text = chapter[1]
        
        # Parse text with spaCy
        doc = nlp(chapter_text)

        # Initialize a dictionary to track POS counts for each target word
        pos_count = {word: {'NOUN': 0, 'VERB': 0, 'ADJ': 0, 'ADV': 0, 'PRON': 0, 'PROPN': 0} for word in target_words}
        emotion_counts = {word: {'anger': 0, 'anticipation': 0, 'disgust': 0, 'fear': 0, 'joy': 0, 'sadness': 0, 'surprise': 0, 'trust': 0} for word in target_words}
        # Loop through tokens and check for target words
        for token in doc:
            if token.text.lower() in target_words:
                # Extract tokens within a 2-token window
                context_tokens = doc[max(token.i - 2, 0): token.i + 3]
                

                context_tokens_before = [t.text.lower() for t in doc[max(token.i - 2, 0): token.i]]
                context_tokens_after = [t.text.lower() for t in doc[token.i + 1: token.i + 3]]
                
                # Combine tokens only within these context ranges
                combined_context_before = " ".join(context_tokens_before)
                combined_context_after = " ".join(context_tokens_after)

                if combined_context_before in emotag1200:
                    for emotion, score in emotag1200[combined_context_before].items():
                        emotion_counts[token.text.lower()][emotion] += float(score)

                if combined_context_after in emotag1200:
                    for emotion, score in emotag1200[combined_context_after].items():
                        emotion_counts[token.text.lower()][emotion] += float(score)

                # Count POS tags for the target word
                for t in context_tokens:
                    if t != token and t.pos_ in pos_count[token.text.lower()]:
                        pos_count[token.text.lower()][t.pos_] += 1
                    if t.text.lower() in emotag1200:
                        for emotion, score in emotag1200[t.text.lower()].items():
                            emotion_counts[token.text.lower()][emotion] += float(score)
        
        for word in target_words:
            for pos, count in pos_count[word].items():
                overall_pos_count[word][pos] += count
            for emotion, count in emotion_counts[word].items():
                overall_emotion_count[word][emotion] += count
        # Store the raw counts for each chapter
        for word in target_words:
            context_data.append({
                'chapter': chapter_name,
                'target_word': word,
                'NOUN': pos_count[word]['NOUN'],
                'VERB': pos_count[word]['VERB'],
                'ADJ': pos_count[word]['ADJ'],
                'ADV': pos_count[word]['ADV'],
                'PRON': pos_count[word]['PRON'],
                'PROPN': pos_count[word]['PROPN']
            })
            emotion_context_data.append({
                'chapter': chapter_name,
                'target_word': word,
                'anger': emotion_counts[word]['anger'],
                'anticipation': emotion_counts[word]['anticipation'],
                'disgust': emotion_counts[word]['disgust'],
                'fear': emotion_counts[word]['fear'],
                'joy': emotion_counts[word]['joy'],
                'sadness': emotion_counts[word]['sadness'],
                'surprise': emotion_counts[word]['surprise'],
                'trust': emotion_counts[word]['trust']
            })

    print("Overall POS Counts Summary:")
    for word, pos_counts in overall_pos_count.items():
        print(f"Word '{word}': {pos_counts}")
    print("\nOverall Emotion Counts Summary:")
    for word, emotion_counts in overall_emotion_count.items():
        formatted_counts = {emotion: round(count, 2) for emotion, count in emotion_counts.items()}
        print(f"Word '{word}': {formatted_counts}")
    return context_data, emotion_context_data

def load_emotag_data(file_path):
    """Load EmoTag1200 data from CSV into a dictionary with emotion scores."""
    emotag_data = {}
    df = pd.read_csv(file_path, header=0)
    
    for _, row in df.iterrows():
        word = row[2].lower()  # The word associated with the emoji (index 2)
        emotag_data[word] = {
            "anger": float(row['anger']),
            "anticipation": float(row['anticipation']),
            "disgust": float(row['disgust']),
            "fear": float(row['fear']),
            "joy": float(row['joy']),
            "sadness": float(row['sadness']),
            "surprise": float(row['surprise']),
            "trust": float(row['trust'])
        }
    
    return emotag_data


def plot_total_pos_tags_by_word(csv_file_path):
    # Read the CSV into a DataFrame
    df = pd.read_csv(csv_file_path)
    
    # List of POS tags in the data
    pos_tags = ['NOUN', 'VERB', 'ADJ', 'ADV', 'PRON', 'PROPN']
    
    # Get unique target words
    target_words = df['target_word'].unique()

    # Calculate the total POS counts for each word
    total_pos_counts = df.groupby('target_word')[pos_tags].sum()

    # Create a bar plot for each word
    for word in target_words:
        # Get the total counts for the current word
        word_totals = total_pos_counts.loc[word]

        # Create a figure for the word
        plt.figure(figsize=(8, 5))
        
        # Plot the total POS counts as a bar chart
        word_totals.plot(kind='bar', color='skyblue')
        
        # Adding labels and title
        plt.xlabel('POS Tags')
        plt.ylabel('Total Count')
        plt.title(f'Total POS Tag Counts for "{word.capitalize()}"')
        
        # Show the plot for each word
        plt.tight_layout()
        plt.show()


#For task 9



if __name__=='__main__':
    # Task 1
    contents= []
    # Prints vocabulary of the whole book
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
    #matrix = analyse_matrix(chapterdata)
    #np.savetxt("vocabulary_matrix.csv", matrix)

    #matrix = analyse_matrix_with_cosine(chapterdata)
    #np.savetxt("vocabulary_cosine_matrix.csv", matrix)

    # Loads the data from file"
    loaded_matrix = np.loadtxt('vocabulary_matrix.csv')
    # Calculates Book overlap
    inferno_overlap = analyze_within_book_overlap(loaded_matrix,0,33)
    purgatory_overlap = analyze_within_book_overlap(loaded_matrix,34,67)
    paradise_overlap = analyze_within_book_overlap(loaded_matrix,67,100)
    print("Inferno book average vocabulary overlap : {}\n".format(inferno_overlap),
          "Purgatory book average vocabulary overlap : {}\n".format(purgatory_overlap),
          "Paradise book average vocabulary overlap : {}\n".format(paradise_overlap)
        )
    display_matrix(loaded_matrix, "Vocabulary overlap matrix")
    # Vocabulary overlap using cosine
    voc_cosine_matrix = np.loadtxt("vocabulary_cosine_matrix.csv")
    display_matrix(voc_cosine_matrix, "Cosine Vocabulary overlap matrix")
    # Calculates Pearson corr and p-value of the two vocabulary overlap matrixes
    pearson_coeff, p_value = calculate_pearson_correlation(loaded_matrix,voc_cosine_matrix)
    print(f"Pearson correlation between Vocabulary overlap measures {pearson_coeff} with p-value of {p_value}")


    # Task 3
    # Run these for topic similarity matrix (takes about 5 seconds)
    #topics, vocab = apply_lda_to_chapters(chapterdata)
    
    # Calculate cosine and jaccard topic similarity
    #topic_vectors = create_topic_vectors(topics, vocab)
    #similarity_matrix = cosine_similarity(topic_vectors)
    #np.savetxt("topic_cosine_similarity_matrix.csv", similarity_matrix)
    #Calculate jaccard similarity
    #matrix = calculate_jaccard_topic_similarity(topics)
    #np.savetxt("topic_jaccard_similarity_matrix.csv", matrix)

    topic_c_matrix = np.loadtxt("topic_cosine_similarity_matrix.csv")
    display_matrix(topic_c_matrix, "Topic Cosine similarity matrix")
    
    
    topic_j_matrix = np.loadtxt('topic_jaccard_similarity_matrix.csv')
    display_matrix(topic_j_matrix, "Topic Jaccard similarity matrix")

    # Combination matrix
    #combination_matrix = combine_similarity_matrices(topic_c_matrix, topic_j_matrix, 0.5)
    #np.savetxt("topic_similarity_combination_matrix.csv", combination_matrix)

    comb_matrix = np.loadtxt("topic_similarity_combination_matrix.csv")
    display_matrix(comb_matrix, "Topic similarity Combination matrix")

    pearson_coeff, p_value = calculate_pearson_correlation(topic_j_matrix,topic_c_matrix)

    print(f"Pearson correlation between Jaccard and Cosine topic similarity {pearson_coeff} with p-value of {p_value}")
    #np.savetxt("topic_similarity_pearson_matrix.csv", pearson_matrix)

    #comb_matrix = np.loadtxt("topic_similarity_pearson_matrix.csv")
    #display_matrix(comb_matrix, "Topic similarity pearson matrix")

    #Task 8 

    three_terms, word_counts = three_most_frequent_terms(chapterdata)
    # prints the three terms and the times they appear
    print(three_terms)
    data = load_emotag_data("EmoTag1200-scores.csv")
    # gives context_data for POS tags and emotion_context_data for EmoTag 
    #context_data, emotion_context_data = get_context_tokens(chapterdata, data)
    # Saves the data into files
    '''
    with open("term_pos_tag_data.csv", mode="w", newline='', encoding='utf-8') as csv_file:
        pos_fieldnames = ['chapter', 'target_word', 'NOUN', 'VERB', 'ADJ', 'ADV', 'PRON', 'PROPN']
        writer = csv.DictWriter(csv_file, fieldnames=pos_fieldnames)
        
        # Write header
        writer.writeheader()
        
        # Write rows
        for data in context_data:
            writer.writerow(data)

    with open("term_emotag_data.csv", mode="w", newline='', encoding='utf-8') as csv_file:
        emotion_fieldnames = ['chapter', 'target_word', 'anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust']
        writer = csv.DictWriter(csv_file, fieldnames=emotion_fieldnames)
        
        # Write header
        writer.writeheader()
        
        # Write rows
        for data in emotion_context_data:
            writer.writerow(data)
    '''
    plot_total_pos_tags_by_word("term_pos_tag_data.csv")
    
    
