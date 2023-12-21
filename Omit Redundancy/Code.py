import nltk
import random
import string
import math
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
import networkx as nx

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Read input file and store it as a string in text
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
all_words = list(set([word for sentence in preprocessed_sentences for word in sentence.split()]))  # Unique words

# Calculate the IDF for each word
idf_values = {}
total_sentences = len(preprocessed_sentences)

for word in all_words:
    sentences_with_word = sum(1 for sentence in preprocessed_sentences if word in sentence)
    idf = math.log(total_sentences / (1 + sentences_with_word))
    idf_values[word] = idf

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

choice=input("Enter 1 for MMR and 2 for Clustering")
# Print the TF-IDF matrix
# for i, sentence in enumerate(sentences):
#     print(f"Sentence {i + 1}:")
#     for word, tfidf_value in tfidf_matrix[i].items():
#         print(f"Word '{word}': TF-IDF = {tfidf_value:.4f}")


if choice=='1' :

    # Compute cosine similarity between TF-IDF vectors of sentences
    # def cosine_similarity(vector1, vector2):
    #     dot_product = sum(vector1[word] * vector2[word] for word in vector1 if word in vector2)
    #     norm_vector1 = math.sqrt(sum(value ** 2 for value in vector1.values()))
    #     norm_vector2 = math.sqrt(sum(value ** 2 for value in vector2.values()))
    #     similarity = dot_product / (norm_vector1 * norm_vector2)
    #     return similarity

    def cosine_similarity(vector1, vector2):
        common_words = set(vector1.keys()) & set(vector2.keys())

        if not common_words:
            return 0.0  # Return zero similarity if there are no common words

        dot_product = sum(vector1[word] * vector2[word] for word in common_words)
        norm_vector1 = math.sqrt(sum(value ** 2 for value in vector1.values()))
        norm_vector2 = math.sqrt(sum(value ** 2 for value in vector2.values()))

        if norm_vector1 == 0 or norm_vector2 == 0:
            return 0.0  # Return zero similarity if one or both vectors have zero magnitude

        similarity = dot_product / (norm_vector1 * norm_vector2)
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


    # Find the sentence with the highest PageRank score
    max_pagerank_sentence = max(pagerank, key=pagerank.get)
    max_pagerank_score = pagerank[max_pagerank_sentence]

    def mmr(pagerank, sentences, lambda_param):
        
        if not sentences:
            return None  # Return None if there are no sentences left to select

        max_mmr_score = -float('inf')
        selected_sentence = None

        for i, sentence in enumerate(sentences):
            mmr_score = lambda_param * pagerank[i] - (1 - lambda_param) * max([pagerank[j] for j in range(len(sentences)) if j != i], default=0)

            if mmr_score > max_mmr_score:
                max_mmr_score = mmr_score
                selected_sentence = sentence

        return selected_sentence


    no_of_sentence = 2 
    selected = []

    for i in range(no_of_sentence):
        next_sentence = mmr(pagerank, sentences, 0.5)  # lambda_param =0.5
        selected.append(next_sentence)
        # delete the selected sentence from sentences
        if next_sentence in sentences:
            index = sentences.index(next_sentence)
            del sentences[index]

    with open('Summary_SentenceGraph.txt', 'w') as f:

        # Print the selected sentences
        f.write("Selected Sentences:\n")
        print("Selected Sentences:")
        for sentence in selected:
            f.write(sentence)
            f.write("\n")
            print(sentence)


elif choice=='2':


    #---------------CLUSTERING----------------  
    # Step 1: Choose the number of clusters (K)
    K = 3 

    # Step 2: Initialize centroids randomly
    initial_centroid_indices = random.sample(range(len(preprocessed_sentences)), K)
    centroids = [np.array([tfidf_matrix[i].get(word, 0) for word in all_words]) for i in initial_centroid_indices]  # Convert to NumPy array
    # print(centroids)

    # Step 3: Define a function to calculate cosine similarity
    def cosine_similarity(vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        return dot_product / (norm_vec1 * norm_vec2)

    # Step 4 and 5: K-Means Loop
    max_iterations = 100  # Adjust this based on convergence
    for iteration in range(max_iterations):
        # Initialize clusters as defaultdicts of lists
        clusters = defaultdict(list)

        # Assign each sentence to the nearest centroid
        for i, sentence_vector in enumerate(tfidf_matrix):
            sentence_vec_array = np.array([sentence_vector.get(word, 0) for word in all_words])  # Convert to NumPy array
            similarities = [cosine_similarity(sentence_vec_array, centroid) for centroid in centroids]
            nearest_centroid_index = np.argmax(similarities)
            clusters[nearest_centroid_index].append(i)

        # Update centroids
        new_centroids = []
        for cluster_indices in clusters.values():
            cluster_vectors = [np.array([tfidf_matrix[i].get(word, 0) for word in all_words]) for i in cluster_indices]  # Convert to NumPy array
            mean_vector = np.mean(cluster_vectors, axis=0)
            new_centroids.append(mean_vector)
        new_centroids = np.array(new_centroids)

        # Check for convergence - checks if all elements in centroids and new_centroids are approximately equal
        if np.all(np.isclose(centroids, new_centroids)):
            break

        centroids = new_centroids
        
    with open('Summary_SentenceGraph.txt', 'w') as f:

        # Print the cluster assignments for each sentence
        
        f.write("Sentences in Each Cluster\n")
        for cluster_id, sentence_indices in clusters.items():
            f.write(f"Cluster {cluster_id + 1} Sentences:\n")
            for sentence_index in sentence_indices:
                f.write(sentences[sentence_index])
                f.write("\n")
        f.write("\n\n\n\n")


        # Print the mean centroid for each cluster
        for cluster_id, centroid in enumerate(centroids):
            f.write(f"Cluster {cluster_id + 1} Mean Centroid:\n")  # Print cluster identifier
            for term, score in sorted(zip(all_words, centroid), key=lambda x: x[1], reverse=True):
                # Sort the terms in the centroid by TF-IDF score in descending order and print each term with its score
                f.write(f"{term}: {score}\n")
                # f.write("\n")
            f.write("\n")  # Print an empty line to separate clusters



        f.write("------------------------------TASK2------------------------------------------------------\n")

        # Helper function to get bigrams from a sentence
        def get_bigrams(sentence):
            words = sentence.split()
            return [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]

        def process_clusters(clusters, sentences, tfidf_matrix):
            final_sentences = []  # Initialize a list to store the final processed sentences

            # Iterate through each cluster of sentences
            for cluster_indices in clusters.values():
                if not cluster_indices:
                    continue  # Skip empty clusters

                # Step 1a: Identify the sentence closest to the cluster centroid
                centroid_indices = [i for i in cluster_indices]  # Get the indices of sentences in this cluster
                centroid_values = [np.array([tfidf_matrix[i].get(word, 0) for word in all_words]) for i in centroid_indices]
                # Calculate the centroid of this cluster's TF-IDF vectors
                centroid = np.mean(centroid_values, axis=0)

                # Calculate cosine similarity for each sentence in the cluster with the centroid
                similarities = [cosine_similarity(np.array([tfidf_matrix[i].get(word, 0) for word in all_words]), centroid) for i in cluster_indices]

                # Find the index of the sentence with the highest cosine similarity to the centroid
                closest_sentence_index = cluster_indices[np.argmax(similarities)]

                S1 = sentences[closest_sentence_index]  # The sentence closest to the centroid

                # Step 1b: Find a sentence with at least 3 common bigrams with S1
                S2 = None  # Initialize S2 to None
                S1_bigrams = set(get_bigrams(S1))  # Get the bigrams in S1

                for candidate_index in cluster_indices:
                    if candidate_index == closest_sentence_index:
                        continue  # Skip the sentence that is already S1
                    candidate_sentence = sentences[candidate_index]
                    candidate_bigrams = set(get_bigrams(candidate_sentence))  # Get the bigrams in the candidate sentence
                    common_bigrams = S1_bigrams.intersection(candidate_bigrams)  # Find common bigrams
                    if len(common_bigrams) >= 3:  # Check if there are at least 3 common bigrams
                        S2 = candidate_sentence
                        break

                # Step 1c: If no S2 exists, save only S1 and return
                if S2 is None:
                    final_sentences.append(S1)
                    continue

                # Step 1d: Construct a sentence graph G using S1 and S2
                G = nx.DiGraph()
                start_node = 'start'
                end_node = 'end'
                G.add_node(start_node)
                G.add_node(end_node)

                for sentence in [S1, S2]:
                    bigrams = get_bigrams(sentence)
                    for i, bigram in enumerate(bigrams):
                        if bigram not in G:
                            G.add_node(bigram)
                        if i == 0:
                            G.add_edge(start_node, bigram)
                        if i == len(bigrams) - 1:
                            G.add_edge(bigram, end_node)
                        if i > 0:
                            G.add_edge(bigrams[i - 1], bigram)

                # Step 2: Generate a sentence by traversing the graph randomly
                path = nx.shortest_path(G, source=start_node, target=end_node)
                generated_sentence = ' '.join(path[1:-1])  # Exclude start and end nodes
                final_sentences.append(generated_sentence)  # Append the generated sentence to the final list

            return final_sentences  # Return the list of processed sentences


        # Process clusters and get the final sentences
        final_sentences = process_clusters(clusters, sentences, tfidf_matrix)

        # Print the final sentences
        for i, sentence in enumerate(final_sentences):
            f.write(f"Cluster {i + 1} Sentence: {sentence}\n")


        f.write("\n\n")
        f.write("--------------------------------TASK3------------------------------------------\n")

        # Process clusters and get the final sentences
        final_sentences = process_clusters(clusters, sentences, tfidf_matrix)
        # Calculate cluster_order for each cluster
        cluster_order = {}

        # Iterate through each cluster and assign the index of the first sentence (S1) in the cluster as its order
        for cluster_id, sentence_indices in clusters.items():
            closest_sentence_index = sentence_indices[0]  # S1 index
            cluster_order[cluster_id] = closest_sentence_index

        # Sort clusters based on cluster_order
        sorted_clusters = sorted(cluster_order.keys(), key=lambda x: cluster_order[x])

        # Arrange sentences based on sorted_clusters
        ordered_sentences = []

        # Iterate through the sorted cluster IDs to arrange sentences
        for cluster_id in sorted_clusters:
            ordered_sentences.append(final_sentences[cluster_id])

        # Print the final ordered sentences
        for i, sentence in enumerate(ordered_sentences):
            # Print each sentence along with its cluster ID, order, and content
            f.write(f"Cluster {sorted_clusters[i] + 1} Sentence (Order: {cluster_order[sorted_clusters[i]]}): {sentence}\n")
