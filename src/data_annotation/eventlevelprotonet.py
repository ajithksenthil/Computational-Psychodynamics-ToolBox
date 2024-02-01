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
input_size = 768  # For bert-base-uncased
hidden_size = 512  # Example hidden size, adjust as needed

proto_net_BC = ProtoNet(input_size, hidden_size, dropout_rate=0.5)
proto_net_PS = ProtoNet(input_size, hidden_size, dropout_rate=0.5)

optimizer_BC = optim.Adam(proto_net_BC.parameters(), lr=0.001)
optimizer_PS = optim.Adam(proto_net_PS.parameters(), lr=0.001)

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




def classify_new_events(new_events, proto_net_BC, proto_net_PS, prototype_tensor_BC, prototype_tensor_PS):
    new_event_embeddings = [get_contextual_embedding(event) for event in new_events]
    new_event_embeddings_tensor = torch.stack(new_event_embeddings)
    
    dists_BC = proto_net_BC(new_event_embeddings_tensor, prototype_tensor_BC)
    dists_PS = proto_net_PS(new_event_embeddings_tensor, prototype_tensor_PS)
    
    predicted_classes_BC = torch.argmin(dists_BC, dim=1)
    predicted_classes_PS = torch.argmin(dists_PS, dim=1)
    
    # Combine BC and PS predictions here
    # This requires custom logic based on your classification scheme
    
    return predicted_classes_BC.numpy(), predicted_classes_PS.numpy()

def test(): 
    # Adjust this function to create event sentences and classify them using the updated `classify_new_events`
    new_events = ['Event sentence 1', 'Event sentence 2', 'Event sentence 3']
    predicted_labels_BC, predicted_labels_PS = classify_new_events(new_events, proto_net_BC, proto_net_PS, prototype_tensor_BC, prototype_tensor_PS)
    print("Predicted labels for BC:", predicted_labels_BC)
    print("Predicted labels for PS:", predicted_labels_PS)

