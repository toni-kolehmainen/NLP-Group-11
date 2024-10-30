import re

def count_sentences(text):
    sentences = re.split(r'[.!?]', text)
    return len([s for s in sentences if s.strip()])


def count_complex_words(text):
    words = re.findall(r'\b\w+\b', text)
    complex_words = [word for word in words if syllable_count(word) >= 3]
    return len(complex_words)

def syllable_count(word):
    word = word.lower()
    vowels = "aeiouy"
    count = 0
    if word[0] in vowels:
        count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1
    if word.endswith("e"):
        count -= 1
    if count == 0:
        count += 1
    return count

def gunning_fog_index(text):
    total_words = count_words(text)
    total_sentences = count_sentences(text)
    complex_words = count_complex_words(text)

    if total_words == 0 or total_sentences == 0:
        return None  # Avoid division by zero

    average_sentence_length = total_words / total_sentences
    percent_complex_words = (complex_words / total_words) * 100

    fog_index = 0.4 * (average_sentence_length + percent_complex_words)
    return fog_index


def analyze_text(text):
    words = text.split()
    total_words = len(words)
    total_sentences = count_sentences(text)
    total_syllables = sum(count_syllables(word) for word in words)
    return total_words, total_sentences, total_syllables

def fry_readability_index(total_words, total_sentences, total_syllables):
    words_per_sentence = total_words / total_sentences
    syllables_per_word = total_syllables / total_words
    fry_index = 0.39 * (words_per_sentence + 11.8 * syllables_per_word) - 15.59
    return fry_index


def count_single_syllable_words(text):
    words = re.findall(r'\b\w+\b', text)
    single_syllable_words = [word for word in words if syllable_count(word) == 1]
    return len(single_syllable_words)

def forecast_readability_score(text):
    single_syllable_words = count_single_syllable_words(text)
    forecast_score = 20 - (single_syllable_words / 15)
    return forecast_score

def count_words(text):
    words = re.findall(r'\b\w+\b', text)
    return len(words)

def count_syllables(word):
    word = word.lower()
    vowels = "aeiouy"
    count = 0
    if word[0] in vowels:
        count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1
    if word.endswith("e"):
        count -= 1
    if count == 0:
        count += 1
    return count

def calculate_flesch_reading_ease(text):
    total_words = count_words(text)
    total_sentences = len(re.findall(r'[.!?]', text))
    total_syllables = sum(count_syllables(word) for word in re.findall(r'\b\w+\b', text))

    flesch_reading_ease = 206.835 - 1.015 * (total_words / total_sentences) - 84.6 * (total_syllables / total_words)
    return flesch_reading_ease
