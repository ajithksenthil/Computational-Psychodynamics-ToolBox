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
    def __init__(self, input_size, hidden_size, dropout_rate=0.5):
        super(ProtoNet, self).__init__()
        # Define a learnable transformation
        # This can be a single layer or a sequence of layers
        self.transform = nn.Sequential(
            nn.Linear(input_size, hidden_size),  # First transformation layer
            nn.ReLU(),  # Non-linearity
            nn.Dropout(dropout_rate),  # Dropout for regularization
            # Add more layers as needed, depending on the complexity of your task
            nn.Linear(hidden_size, hidden_size // 2),  # Additional transformation layer
            nn.ReLU(),  # Non-linearity
        )

    def forward(self, x, prototypes):
        # Apply the transformation to the input embeddings x
        transformed_x = self.transform(x)

        # Apply the same transformation to each prototype
        # Ensure that prototypes are properly unsqueezed if they lack a batch dimension
        transformed_prototypes = torch.stack([self.transform(proto.unsqueeze(0)).squeeze(0) for proto in prototypes])
        
        # Calculate Euclidean distance between the transformed x and the transformed prototypes
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

def create_episode(event_data, n_support, n_query):
    """
    Create a support and query set for an episode from event embeddings or representations.
    
    :param event_data: Dictionary of class labels to lists of event embeddings.
    :param n_support: Number of examples per class in the support set.
    :param n_query: Number of examples per class in the query set.
    :return: Two dictionaries representing the support and query sets.
    """
    support_set = {}
    query_set = {}
    for label, embeddings in event_data.items():
        # Ensure embeddings are shuffled before sampling
        random.shuffle(embeddings)
        support_set[label] = embeddings[:n_support]
        query_set[label] = embeddings[n_support:n_support + n_query]
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

def prepare_query_samples_and_labels(query_set):
    """
    Prepare query samples and labels for classification from the query set.
    
    :param query_set: Dictionary of class labels to lists of tuples (embedding, label).
    :return: A tuple of (query_samples_tensor, query_labels_tensor).
    """
    query_samples = []
    query_labels = []
    label_to_index = {label: idx for idx, label in enumerate(query_set.keys())}  # Map labels to indices
    
    for label, events in query_set.items():
        for event in events:
            query_samples.append(event[0])  # Assuming event[0] is the embedding
            query_labels.append(label_to_index[label])  # Convert label to index
    
    # Convert lists to tensors
    query_samples_tensor = torch.stack(query_samples)
    query_labels_tensor = torch.tensor(query_labels, dtype=torch.long)
    
    return query_samples_tensor, query_labels_tensor


def train_proto_net(proto_net, optimizer, query_samples, query_labels, prototype_tensor):
    """
    Train the Prototypical Network for one episode.
    
    :param proto_net: The Prototypical Network model to be trained.
    :param optimizer: The optimizer for updating the model's parameters.
    :param query_samples: Tensor of query samples for the current episode.
    :param query_labels: Tensor of labels for the query samples.
    :param prototype_tensor: Tensor of prototypes for the current episode.
    """
    proto_net.train()  # Ensure the network is in training mode
    
    # Forward pass: Compute the distances from query samples to prototypes
    dists = proto_net(query_samples, prototype_tensor)
    
    # Compute the loss: Negative distance because we're using the CrossEntropy loss,
    # which expects logits (higher values for more likely classes). Since distance implies
    # the opposite (lower is better), we negate the distances.
    loss = F.cross_entropy(-dists, query_labels)
    
    # Backward pass: compute gradient of the loss with respect to model parameters
    optimizer.zero_grad()  # Reset gradients; they accumulate by default
    loss.backward()
    
    # Perform a single optimization step (parameter update)
    optimizer.step()

    # Optionally return the loss for logging or any other monitoring
    return loss.item()


def validate_proto_net(proto_net, val_events, task_label):
    """
    Validate the Prototypical Network on a validation dataset.
    
    :param proto_net: The Prototypical Network model to be validated.
    :param val_events: The validation dataset.
    :param task_label: A string indicating the task ('BC' or 'PS') to filter the relevant events.
    """
    proto_net.eval()  # Set the network to evaluation mode
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    # Filter validation events for the current task
    task_events = {k: v for k, v in val_events.items() if k in task_label}
    
    # Prepare support and query sets for validation
    # Assuming create_episode and compute_prototypes are defined similarly to the training phase
    support_set, query_set = create_episode(task_events, n_support_val, n_query_val)
    prototypes = compute_prototypes(support_set)
    prototype_tensor = torch.stack(list(prototypes.values()))
    
    query_samples, query_labels = prepare_query_samples_and_labels(query_set)
    
    with torch.no_grad():
        # Forward pass to compute distances
        dists = proto_net(query_samples, prototype_tensor)
        
        # Compute the loss
        loss = F.cross_entropy(-dists, query_labels)
        total_loss += loss.item() * query_samples.size(0)  # Multiply by batch size for accurate mean calculation
        
        # Compute accuracy
        _, predicted = torch.min(dists, 1)
        total_correct += (predicted == query_labels).sum().item()
        total_samples += query_labels.size(0)
    
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples * 100
    
    print(f"Validation Results - Task {task_label}: Average Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")


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
dropout_rate = 0.5  # Common starting point, adjust as needed

proto_net_BC = ProtoNet(input_size, hidden_size, dropout_rate=0.5)
proto_net_PS = ProtoNet(input_size, hidden_size, dropout_rate=0.5)

optimizer_BC = optim.Adam(proto_net_BC.parameters(), lr=0.001)
optimizer_PS = optim.Adam(proto_net_PS.parameters(), lr=0.001)

# Event-Level Training Loop
for episode in range(num_episodes):
    # Assuming event_representations is structured similarly to word_vectors
    # and split into training (train_events) and validation (val_events) datasets
    
    # Create episodes for BC and PS training
    support_set_BC, query_set_BC = create_episode({key: val for key, val in train_events.items() if key in ['B', 'C']}, n_support, n_query)
    support_set_PS, query_set_PS = create_episode({key: val for key, val in train_events.items() if key in ['P', 'S']}, n_support, n_query)
    
    # Compute prototypes for BC and PS
    prototypes_BC = compute_prototypes(support_set_BC)
    prototypes_PS = compute_prototypes(support_set_PS)
    
    prototype_tensor_BC = torch.stack(list(prototypes_BC.values()))
    prototype_tensor_PS = torch.stack(list(prototypes_PS.values()))

    # Training for BC
    query_samples_BC, query_labels_BC = prepare_query_samples_and_labels(query_set_BC)
    train_proto_net(proto_net_BC, optimizer_BC, query_samples_BC, query_labels_BC, prototype_tensor_BC)
    
    # Training for PS
    query_samples_PS, query_labels_PS = prepare_query_samples_and_labels(query_set_PS)
    train_proto_net(proto_net_PS, optimizer_PS, query_samples_PS, query_labels_PS, prototype_tensor_PS)

    # Validation for BC and PS
    validate_proto_net(proto_net_BC, val_events, 'BC')
    validate_proto_net(proto_net_PS, val_events, 'PS')




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

