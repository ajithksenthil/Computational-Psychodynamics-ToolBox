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
    

# class ProtoNet(nn.Module):
#     def __init__(self, input_size, hidden_size, dropout_rate=0.5):
#         super(ProtoNet, self).__init__()
#         # Define a more complex learnable transformation
#         self.transform = nn.Sequential(
#             nn.Linear(input_size, hidden_size),
#             nn.ReLU(),
#             nn.Dropout(dropout_rate),  # Add dropout after the activation
#             nn.Linear(hidden_size, hidden_size),  # Additional linear layer
#             nn.ReLU()  # Activation function for the additional layer
#             # Can add more layers or dropout as needed
#         )

#     def forward(self, x, prototypes):
#         # Apply the transformation to both x and prototypes
#         transformed_x = self.transform(x)
#         # Transform prototypes using a loop or another method that ensures
#         # each prototype is individually processed by the transform
#         transformed_prototypes = torch.stack([self.transform(proto.unsqueeze(0)).squeeze(0) for proto in prototypes])
        
#         # Calculate Euclidean distance between transformed x and the transformed prototypes
#         dists = torch.cdist(transformed_x, transformed_prototypes)
#         return dists


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

def prepare_query_samples_and_labels(query_set, prototypes=None):
    query_samples = []
    query_labels = []
    label_to_index = {label: idx for idx, label in enumerate(query_set.keys())}
    
    for label, vectors in query_set.items():
        query_samples.extend(vectors)
        query_labels.extend([label_to_index[label]] * len(vectors))
    
    query_samples_tensor = torch.stack(query_samples)
    query_labels_tensor = torch.tensor(query_labels, dtype=torch.long)
    
    # If prototypes are used for something specific here, add that logic
    
    return query_samples_tensor, query_labels_tensor

def validate_network(proto_net, optimizer, word_tensors_val, n_support_val, n_query_val):
    proto_net.eval()  # Set the network to evaluation mode
    val_loss = 0.0
    val_accuracy = 0.0
    total_queries = 0
    
    # Create a validation episode
    support_set_val, query_set_val = create_episode(word_tensors_val, n_support_val, n_query_val)
    
    # Compute prototypes for the validation support set
    prototypes_val = compute_prototypes(support_set_val)
    prototype_tensor_val = torch.stack(list(prototypes_val.values()))

    # Prepare query samples and labels for validation
    query_samples_val, query_labels_val = prepare_query_samples_and_labels(query_set_val, prototypes_val)

    # Forward pass for validation
    with torch.no_grad():
        dists_val = proto_net(query_samples_val, prototype_tensor_val)
        val_loss = F.cross_entropy(-dists_val, query_labels_val)

        # Calculate accuracy
        _, predictions = torch.min(dists_val, 1)
        correct = (predictions == query_labels_val).sum().item()
        total_queries += query_labels_val.size(0)
        val_accuracy = correct / total_queries

    print(f"Validation Loss: {val_loss.item()}, Validation Accuracy: {val_accuracy * 100:.2f}%")
    proto_net.train()  # Set the model back to training mode
    return val_loss, val_accuracy



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
num_episodes = 15


# Assuming you have validation data prepared similarly to training data
val_B_vectors = B_vectors
val_C_vectors = C_vectors
val_P_vectors = P_vectors
val_S_vectors = S_vectors

# Assuming you have separate validation data prepared similarly for BC and PS
val_word_tensors_BC = {'B': torch.tensor(val_B_vectors, dtype=torch.float32), 'C': torch.tensor(val_C_vectors, dtype=torch.float32)}
val_word_tensors_PS = {'P': torch.tensor(val_P_vectors, dtype=torch.float32), 'S': torch.tensor(val_S_vectors, dtype=torch.float32)}

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
# model = ProtoNet(input_size, hidden_size)

# Initialize two ProtoNets: one for B vs. C, and one for P vs. S
proto_net_BC = ProtoNet(input_size=embedding_size, hidden_size=256)
proto_net_PS = ProtoNet(input_size=embedding_size, hidden_size=256)

optimizer_BC = optim.Adam(proto_net_BC.parameters(), lr=0.001)
optimizer_PS = optim.Adam(proto_net_PS.parameters(), lr=0.001)



for episode in range(num_episodes):
    # Split word_tensors for BC and PS networks
    word_tensors_BC = {'B': torch.tensor(B_vectors, dtype=torch.float32), 'C': torch.tensor(C_vectors, dtype=torch.float32)}
    word_tensors_PS = {'P': torch.tensor(P_vectors, dtype=torch.float32), 'S': torch.tensor(S_vectors, dtype=torch.float32)}

    # Create episodes for BC and PS networks
    support_set_BC, query_set_BC = create_episode(word_tensors_BC, n_support, n_query)
    support_set_PS, query_set_PS = create_episode(word_tensors_PS, n_support, n_query)

    # Compute prototypes for BC and PS networks
    prototypes_BC = compute_prototypes(support_set_BC)
    prototypes_PS = compute_prototypes(support_set_PS)

    prototype_tensor_BC = torch.stack(list(prototypes_BC.values()))
    prototype_tensor_PS = torch.stack(list(prototypes_PS.values()))

    # Prepare query samples and labels for BC classification
    query_samples_BC, query_labels_BC = prepare_query_samples_and_labels(query_set_BC, prototypes_BC)
    query_samples_PS, query_labels_PS = prepare_query_samples_and_labels(query_set_PS, prototypes_PS)

    # Forward pass and compute loss for BC network
    dists_BC = proto_net_BC(query_samples_BC, prototype_tensor_BC)
    loss_BC = F.cross_entropy(-dists_BC, query_labels_BC)

    # Backward and optimize for BC network
    optimizer_BC.zero_grad()
    loss_BC.backward()
    optimizer_BC.step()

    # Forward pass and compute loss for PS network
    dists_PS = proto_net_PS(query_samples_PS, prototype_tensor_PS)
    loss_PS = F.cross_entropy(-dists_PS, query_labels_PS)

    # Backward and optimize for PS network
    optimizer_PS.zero_grad()
    loss_PS.backward()
    optimizer_PS.step()

    print(f"Episode {episode + 1}, Loss for BC Network: {loss_BC.item()}, Loss for PS Network: {loss_PS.item()}")

    # Validation for BC network
    print("Validating BC Network:")
    validate_network(proto_net_BC, optimizer_BC, val_word_tensors_BC, n_support_val, n_query_val)

    # Validation for PS network
    print("Validating PS Network:")
    validate_network(proto_net_PS, optimizer_PS, val_word_tensors_PS, n_support_val, n_query_val)




def classify_new_words(new_words, word2vec_model, proto_net_BC, proto_net_PS, prototype_tensor_BC, prototype_tensor_PS):
    new_word_embeddings = [word2vec_model[word.lower()] for word in new_words if word.lower() in word2vec_model]
    if not new_word_embeddings:
        print("None of the new words were found in the model's vocabulary.")
        return []
    new_word_embeddings_tensor = torch.tensor(new_word_embeddings, dtype=torch.float32)
    
    # Compute distances to prototypes for both networks
    dists_BC = proto_net_BC(new_word_embeddings_tensor, prototype_tensor_BC)
    predicted_classes_BC = torch.argmin(dists_BC, dim=1)
    
    dists_PS = proto_net_PS(new_word_embeddings_tensor, prototype_tensor_PS)
    predicted_classes_PS = torch.argmin(dists_PS, dim=1)
    
    # Combine the outputs from both networks to form the final classification
    # This step requires a custom logic based on how you decide to combine the binary classifications
    # For simplification, let's just return the outputs for now
    return predicted_classes_BC.numpy(), predicted_classes_PS.numpy()




def test():
    # Example usage
    new_words = ['jump', 'mystify', 'look', 'consider', 'instruct', 'plan']
    
    # Assuming prototype_tensor_BC and prototype_tensor_PS are already defined
    predicted_labels_BC, predicted_labels_PS = classify_new_words(new_words, word2vec_model, proto_net_BC, proto_net_PS, prototype_tensor_BC, prototype_tensor_PS)
    
    # Assuming you have a mechanism to map the numerical predictions back to class labels
    # For simplification, the following just prints the predictions
    print("Predicted labels for B vs. C:", predicted_labels_BC)
    print("Predicted labels for P vs. S:", predicted_labels_PS)
    # You would include here the logic to combine these predictions into your final BP, BS, CP, CS classifications

test()
