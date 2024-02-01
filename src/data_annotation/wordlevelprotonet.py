import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from scipy.spatial.distance import cdist
import numpy as np
import random
from sklearn.metrics.pairwise import euclidean_distances
import gensim.downloader as api
from gensim.models.word2vec import Word2Vec

# Load a pre-trained Word2Vec model
word2vec_model = api.load('word2vec-google-news-300')

# Get the embedding size
embedding_size = word2vec_model.vector_size
print("Embedding size:", embedding_size)



class ProtoNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ProtoNet, self).__init__()
        # Define a learnable transformation
        self.transform = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU()
        )

    def forward(self, x, prototypes):
        # Apply the transformation to both x and prototypes
        transformed_x = self.transform(x)
        transformed_prototypes = [self.transform(proto) for proto in prototypes]
        transformed_prototypes = torch.stack(transformed_prototypes)
        
        # Calculate Euclidean distance between transformed x and the transformed prototypes
        dists = torch.cdist(transformed_x, transformed_prototypes)
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

# Function to get vectors for a list of words using a pre-trained model
def get_vectors(word_list, model):
    vectors = []
    for word in word_list:
        try:
            vector = model[word]
            vectors.append(vector)
        except KeyError:
            # Word is not in the model's vocabulary
            continue
    return vectors






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

# Use the function to get vectors for your word clouds
B_vectors = get_vectors(b_word_cloud_lower, word2vec_model)
C_vectors = get_vectors(c_word_cloud_lower, word2vec_model)
P_vectors = get_vectors(p_word_cloud_lower, word2vec_model)
S_vectors = get_vectors(s_word_cloud_lower, word2vec_model)


# training loop 
word_vectors = {'B': B_vectors, 'C': C_vectors, 'P': P_vectors, 'S': S_vectors}
n_classes = len(word_vectors)
n_support = 5  # Number of examples per class in support set
n_query = 5    # Number of examples per class in query set
num_episodes = 5


# Assuming you have validation data prepared similarly to training data
val_B_vectors = B_vectors
val_C_vectors = C_vectors
val_P_vectors = P_vectors
val_S_vectors = S_vectors
val_word_vectors = {'B': val_B_vectors, 'C': val_C_vectors, 'P': val_P_vectors, 'S': val_S_vectors}
val_word_tensors = {label: torch.tensor(vectors, dtype=torch.float32) for label, vectors in val_word_vectors.items()}

# Define the number of support and query samples for validation
n_support_val = 5
n_query_val = 5
# Assuming word_vectors is a dict with class labels as keys and Word2Vec vectors as values
# Convert word vectors to PyTorch tensors
word_tensors = {label: torch.tensor(vectors, dtype=torch.float32) for label, vectors in word_vectors.items()}


# Assuming the Word2Vec embeddings are 300-dimensional
input_size = 300
# Choose a hidden size for the transformation
hidden_size = 256

# Initialize ProtoNet
model = ProtoNet(input_size, hidden_size)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for episode in range(num_episodes):
    support_set, query_set = create_episode(word_tensors, n_support, n_query)

    prototypes = compute_prototypes(support_set)
    prototype_tensor = torch.stack(list(prototypes.values()))

    # Prepare query samples and labels for classification
    query_samples, query_labels = [], []
    for label, vectors in query_set.items():
        query_samples.extend(vectors)
        query_labels.extend([label] * len(vectors))
    query_samples = torch.stack(query_samples)
    query_labels = torch.tensor([list(prototypes.keys()).index(label) for label in query_labels], dtype=torch.long)

    # Forward pass and compute loss
    dists = model(query_samples, prototype_tensor)
    loss = F.cross_entropy(-dists, query_labels)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Episode {episode + 1}, Loss: {loss.item()}")

    # Validation phase
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        val_support_set, val_query_set = create_episode(val_word_tensors, n_support_val, n_query_val)
        val_prototypes = compute_prototypes(val_support_set)
        val_prototype_tensor = torch.stack(list(val_prototypes.values()))

        # Prepare validation query samples and labels
        val_query_samples, val_query_labels = [], []
        for label, vectors in val_query_set.items():
            val_query_samples.extend(vectors)
            val_query_labels.extend([label] * len(vectors))
        val_query_samples = torch.stack(val_query_samples)
        val_query_labels = torch.tensor([list(val_prototypes.keys()).index(label) for label in val_query_labels], dtype=torch.long)

        # Forward pass for validation
        val_dists = model(val_query_samples, val_prototype_tensor)
        val_loss = F.cross_entropy(-val_dists, val_query_labels)

        print(f"Validation Loss in Episode {episode + 1}: {val_loss.item()}")

    model.train()  # Set the model back to training mode



def classify_new_words(new_words, word2vec_model, prototype_tensor):
    """
    Classify new words using the trained Prototypical Networks.

    :param new_words: List of new words to classify.
    :param word2vec_model: Pre-trained Word2Vec model used for embeddings.
    :param prototype_tensor: Tensor of learned prototypes for each class.
    :return: Predicted class labels for the new words.
    """
    # Correctly accessing word vectors from the KeyedVectors object
    new_word_embeddings = [word2vec_model[word.lower()] for word in new_words if word.lower() in word2vec_model]
    if not new_word_embeddings:
        print("None of the new words were found in the model's vocabulary.")
        return []
    new_word_embeddings_tensor = torch.tensor(new_word_embeddings, dtype=torch.float32)
    
    # Compute distances to prototypes
    dists = model(new_word_embeddings_tensor, prototype_tensor)
    
    # Classify based on the shortest distance to prototypes
    predicted_classes = torch.argmin(dists, dim=1)
    
    return predicted_classes.numpy()





def test(): 
    # Example usage
    new_words = ['jump', 'mystify', 'look', 'consider', 'instruct', 'plan']
    predicted_labels = classify_new_words(new_words, word2vec_model, prototype_tensor)
    print(predicted_labels)


test()