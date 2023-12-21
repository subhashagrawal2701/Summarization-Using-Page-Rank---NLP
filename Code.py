import nltk
# nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import string
from nltk.stem import WordNetLemmatizer
import math
from nltk.corpus import wordnet
import networkx as nx


# # Read input file and store it as a string in text
# with open('input.txt', 'r') as f:
#     text = f.read()
# Read input file and store it as a string in text
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Tokenize the text into sentences and store them as a list in sentences
sentences = sent_tokenize(text)

# Create a list to store the preprocessed sentences
preprocessed_sentences = []

for sentence in sentences:
    # REMOVE PUNCTUATION
    # Create a translation table to remove punctuation
    table = str.maketrans('', '', string.punctuation)
    no_punct_sentence = sentence.translate(table)

    # CONVERT TO LOWERCASE
    no_punct_sentence = no_punct_sentence.lower()

    # TOKENIZE INTO WORDS
    words = word_tokenize(no_punct_sentence)

    # REMOVE STOPWORDS
    eng_stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in eng_stop_words]

    # LEMMATIZE WORDS
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]

    # Append the preprocessed sentence as a space-separated string to the list
    preprocessed_sentences.append(' '.join(lemmatized_words))

# Tokenize words in the entire document
all_words = [word for sentence in preprocessed_sentences for word in sentence.split()]

print(all_words)

# Calculate the IDF for each word
idf_values = {}
total_sentences = len(preprocessed_sentences)

for word in set(all_words):
    sentences_with_word = sum(1 for sentence in preprocessed_sentences if word in sentence)
    idf = math.log(total_sentences / (1 + sentences_with_word))
    idf_values[word] = idf
    print(word, idf_values[word])

# Create a TF-IDF matrix as a list of dictionaries
tfidf_matrix = []

for sentence in preprocessed_sentences:
    tfidf_dict = {}
    words_in_sentence = sentence.split()
    total_words_in_sentence = len(words_in_sentence)
    for word in words_in_sentence:
        # Calculate TF (Term Frequency) for each word in the sentence
        tf = words_in_sentence.count(word) / total_words_in_sentence
        # Calculate TF-IDF
        tfidf = tf * idf_values[word]
        tfidf_dict[word] = tfidf
    tfidf_matrix.append(tfidf_dict)

# Print the TF-IDF matrix
for i, sentence in enumerate(sentences):
    print(f"Sentence {i + 1}:")
    for word, tfidf_value in tfidf_matrix[i].items():
        print(f"Word '{word}': TF-IDF = {tfidf_value:.4f}")

# Compute cosine similarity between TF-IDF vectors of sentences
# def cosine_similarity(vector1, vector2):
#     dot_product = sum(vector1[word] * vector2[word] for word in vector1 if word in vector2)
#     norm_vector1 = math.sqrt(sum(value ** 2 for value in vector1.values()))
#     norm_vector2 = math.sqrt(sum(value ** 2 for value in vector2.values()))
#     similarity = dot_product / (norm_vector1 * norm_vector2)
#     return similarity

def cosine_similarity(vector1, vector2):
    dot_product = sum(vector1[word] * vector2[word] for word in vector1 if word in vector2)
    norm_vector1 = math.sqrt(sum(value ** 2 for value in vector1.values()))
    norm_vector2 = math.sqrt(sum(value ** 2 for value in vector2.values()))

    # Check for zero division
    if norm_vector1 > 0 and norm_vector2 > 0:
        similarity = dot_product / (norm_vector1 * norm_vector2)
    else:
        similarity = 0  # Set similarity to zero if either vector has zero magnitude

    return similarity



# Create an undirected weighted graph
G = nx.Graph()
num_sentences = len(sentences)

for i in range(num_sentences):
    G.add_node(i)  # Add nodes representing sentences
    for j in range(i + 1, num_sentences):
        similarity = cosine_similarity(tfidf_matrix[i], tfidf_matrix[j])
        G.add_edge(i, j, weight=similarity)  # Add edges with cosine similarity as weight

# Compute PageRank
pagerank = nx.pagerank(G)

# Display the top-n nodes (sentences) with the highest PageRank values as the summary
n = 3  # Number of top sentences to display
top_sentences = sorted(pagerank, key=pagerank.get, reverse=True)[:n]
summary = [sentences[idx] for idx in sorted(top_sentences)]

# Display the summary
print("\nSummary:")
for sentence in sentences:
    if sentence in summary:
        print(sentence)

# Write the summary to an output file
with open('Summary_MMR.txt', 'w') as output_file:
    output_file.write("Summary:\n")
    for sentence in sentences:
        if sentence in summary:
            output_file.write(sentence + '\n')

print("\n\nSummary has been written to 'Summary_MMR.txt'.")
