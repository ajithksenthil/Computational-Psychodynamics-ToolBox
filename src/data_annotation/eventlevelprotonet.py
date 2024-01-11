import torch
from torch import nn
from scipy.spatial.distance import cdist
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from scipy.spatial.distance import cdist
import random
from sklearn.metrics.pairwise import euclidean_distances
import gensim.downloader as api
from gensim.models.word2vec import Word2Vec


import transformers
from transformers import AutoTokenizer, AutoModel

# Load pre-trained model tokenizer, you can replace 'bert-base-uncased' with the specific model you want to use
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Load pre-trained model (weights)
model = AutoModel.from_pretrained('bert-base-uncased')


class ProtoNet(nn.Module):
    def __init__(self):
        super(ProtoNet, self).__init__()

    def forward(self, x, prototypes):
        # Calculate Euclidean distance between each x and the prototypes
        dists = torch.cdist(x, prototypes)
        return dists

def compute_prototypes(support_set):
    """
    Computes the prototype for each class in the support set.
    """
    prototypes = {}
    for label, vectors in support_set.items():
        prototypes[label] = torch.mean(vectors, dim=0)
    return prototypes

def create_episode(word_vectors, n_support, n_query):
    """
    Create a support and query set for an episode.
    """
    support_set = {}
    query_set = {}
    for label, vectors in word_vectors.items():
        # Randomly sample words for support and query set
        random.shuffle(vectors)
        support_set[label] = vectors[:n_support]
        query_set[label] = vectors[n_support:n_support + n_query]
    return support_set, query_set



def get_contextual_embedding(event_sentence):
    """
    Generate a contextual embedding for a sentence.
    """
    inputs = tokenizer(event_sentence, return_tensors="pt")
    outputs = model(**inputs)

    # Use the embedding of the [CLS] token (first token) as the representation
    return outputs.last_hidden_state[:, 0, :].detach()

def event_to_sentence(event):
    """
    Convert an event representation to a sentence-like structure.
    """
    return f"The {event['Subject']} is {event['Action']} the {event['Object']} in a {event['Environment']}."

def create_event_representations(word_clouds):
    """
    Create event representations and labels from word clouds. TODO modify word clouds to be event clouds and change this method to fit them
    """
    events = []
    for label, words in word_clouds.items():
        for word in words:
            # Create an event representation
            event = {"Subject": "subject", "Action": word, "Object": "object", "Environment": "environment"}
            event_sentence = event_to_sentence(event)
            embedding = get_contextual_embedding(event_sentence)
            events.append((embedding, label))
    return events

# Set up word clouds
B_word_cloud = ["Communicate", "Organize", "Teach", "Control", "Direct", "Start", "Brag", "Motivate", "Inspire",
                "Empower", "Lead", "Facilitate", "Guide", "Educate", "Mentor", "Encourage", "Advise", "Counsel",
                "Advocate", "Promote", "Endorse", "Support"]
C_word_cloud = ["Absorb", "Acquire", "Assemble", "Digest", "Enroll", "Gather", "Ingest", "Investigate", "Learn",
                "Peruse", "Receive", "Review", "Seek", "Study", "Take in", "Explore", "Respect", "Understand",
                "Analyze", "Comprehend", "Examine", "Scrutinize"]
P_word_cloud = ["Act", "Animate", "Cavort", "Compete", "Engage", "Entertain", "Frolic", "Gamble", "Game", "Jest",
                "Joke", "Leap", "Perform", "Prance", "Recreate", "Sport", "Toy", "Work", "Do", "Explore", "Show",
                "Teach", "Amuse", "Divert", "Enjoy", "Entertain"]
S_word_cloud = ["Catnap", "Doze", "Dream", "Hibernate", "Nap", "Nod", "Ponder", "Repose", "Rest", "Slumber", "Snooze",
                "Sustain", "Think", "Unwind", "Conserve", "Organize", "Introspect", "Process", "Preserve", "Meditate",
                "Reflect", "Relax", "Rejuvenate"]

b_word_cloud_lower = [word.lower() for word in B_word_cloud]
c_word_cloud_lower = [word.lower() for word in C_word_cloud]
p_word_cloud_lower = [word.lower() for word in P_word_cloud]
s_word_cloud_lower = [word.lower() for word in S_word_cloud]


# Prepare event representations
word_clouds = {
    'B': b_word_cloud_lower,
    'C': c_word_cloud_lower,
    'P': p_word_cloud_lower,
    'S': s_word_cloud_lower
}
event_representations = create_event_representations(word_clouds)


# Split event representations into training and validation sets
# TODO we need a function to split the data
# train_events, val_events = split_data(event_representations, train_ratio=0.8)

# Training loop
num_episodes = 5
n_support = 5
n_query = 5

# Initialize ProtoNet
model = ProtoNet()
optimizer = optim.Adam(model.parameters(), lr=0.001)
# Event-Level Training Loop
for episode in range(num_episodes):
    support_set, query_set = create_episode(event_representations, n_support, n_query)

    # Computing prototypes
    prototypes = {}
    for label, events in support_set.items():
        embeddings = torch.stack([event[0] for event in events])
        prototypes[label] = torch.mean(embeddings, dim=0)
    prototype_tensor = torch.stack(list(prototypes.values()))

    # Preparing query samples and labels
    query_samples = torch.stack([event[0] for event in query_set])
    query_labels = torch.tensor([list(prototypes.keys()).index(event[1]) for event in query_set])

    # Forward pass and compute loss
    dists = model(query_samples, prototype_tensor)
    loss = F.cross_entropy(-dists, query_labels)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Episode {episode + 1}, Loss: {loss.item()}")

    # Validation step
    model.eval()
    with torch.no_grad():
        val_support_set, val_query_set = create_episode(val_events, n_support_val, n_query_val)
        val_prototypes = compute_prototypes(val_support_set)
        val_prototype_tensor = torch.stack(list(val_prototypes.values()))

        val_dists = model(torch.stack([x[0] for x in val_query_set]), val_prototype_tensor)
        val_labels = torch.tensor([x[1] for x in val_query_set])
        val_loss = F.cross_entropy(-val_dists, val_labels)
        print(f"Validation Loss in Episode {episode + 1}: {val_loss.item()}")

    model.train()




def classify_new_words(model, new_words, word2vec_model, prototypes):
    """
    Classify new words using the trained Prototypical Networks.
    
    :param model: Trained Prototypical Network model.
    :param new_words: List of new words to classify.
    :param word2vec_model: Pre-trained Word2Vec model used for embeddings.
    :param prototypes: Learned prototypes for each class.
    :return: Predicted class labels for the new words.
    """
    new_word_embeddings = [word2vec_model.wv[word.lower()] for word in new_words if word.lower() in word2vec_model.wv]
    new_word_embeddings_tensor = torch.tensor(new_word_embeddings, dtype=torch.float32)
    
    # Compute distances to prototypes
    dists = model(new_word_embeddings_tensor, prototypes)
    
    # Classify based on the shortest distance to prototypes
    predicted_classes = torch.argmin(dists, dim=1)
    
    return predicted_classes.numpy()


def test(): 
    # Example usage
    new_words = ['Word1', 'Word2', 'Word3']
    predicted_labels = classify_new_words(model, new_words, model, prototype_tensor)
