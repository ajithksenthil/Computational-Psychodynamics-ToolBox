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
from sklearn.model_selection import train_test_split

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
    
    :param support_set: Dictionary of class labels to lists of event embeddings (tensors).
    :return: A dictionary of prototypes for each class.
    """
    prototypes = {}
    for label, vectors in support_set.items():
        # Directly convert list of vectors to a tensor
        vectors_tensor = torch.stack(vectors)
        
        # Compute the mean across the 0th dimension (across all vectors for the class)
        prototypes[label] = torch.mean(vectors_tensor, dim=0)
    return prototypes



def create_episode(event_data, n_support, n_query):
    """
    Create a support and query set for an episode from event embeddings or representations.
    
    :param event_data: Dictionary of class labels to lists of event embeddings.
    :param n_support: Number of examples per class in the support set.
    :param n_query: Number of examples per class in the query set.
    :return: Two dictionaries representing the support and query sets.
    """
    print("event_data length: ", len(event_data))
    print("n_support: ", n_support)
    print("n_query: ", n_query)
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


def create_event_embeddings(event_descriptions):
    event_embeddings = []
    for description in event_descriptions:
        embedding = get_contextual_embedding(description)
        event_embeddings.append(embedding)
    return event_embeddings


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
    # Ensure dists is correctly shaped
    dists = proto_net(query_samples, prototype_tensor).squeeze(-1)  # Adjust based on the actual extra dimension

    # This step assumes dists is [num_classes, batch_size] and needs to be transposed
    dists = dists.transpose(0, 1)
    # Compute the loss: Negative distance because we're using the CrossEntropy loss,
    # which expects logits (higher values for more likely classes). Since distance implies
    # the opposite (lower is better), we negate the distances.
    print(f"Query samples size: {query_samples.size()}")
    print(f"Query labels size: {query_labels.size()}")
    print(f"Dists size: {dists.size()}")
    loss = F.cross_entropy(-dists, query_labels)
    
    # Backward pass: compute gradient of the loss with respect to model parameters
    optimizer.zero_grad()  # Reset gradients; they accumulate by default
    loss.backward()
    
    # Perform a single optimization step (parameter update)
    optimizer.step()

    # Optionally return the loss for logging or any other monitoring
    return loss.item()


def validate_proto_net(proto_net, val_events, task_label, n_support_val, n_query_val):
    """
    Validate the Prototypical Network on a validation dataset.
    
    :param proto_net: The Prototypical Network model to be validated.
    :param val_events: The validation dataset.
    :param task_label: A string indicating the task ('BC' or 'PS') to filter the relevant events.
    :param n_support_val: Number of support examples per class in the validation set.
    :param n_query_val: Number of query examples per class in the validation set.
    """
    proto_net.eval()  # Set the network to evaluation mode
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    # Filter validation events for the current task...
    # Rest of the function remains the same...

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
    # print(f"Query set for {task_label}: {query_set}")
    query_samples, query_labels = prepare_query_samples_and_labels(query_set)
    
    with torch.no_grad():
        # Forward pass to compute distances
        dists = proto_net(query_samples, prototype_tensor)
        dists = dists.squeeze(-1)  # Remove the last dimension if it is of size 1
        print(f"Query samples size: {query_samples.size()}")
        print(f"Query labels size: {query_labels.size()}")
        print(f"Dists size: {dists.size()}")
        
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



# Information Animals
# Event descriptions for Blast
B_event_descriptions = [
    "A manager conducts a training session in the company's conference room.",
    "A teacher explains mathematical concepts to a high school class during an algebra lesson.",
    "A software developer demonstrates the new application features during a team meeting in the development lab.",
    "A tour guide describes historical landmarks to tourists on a city walking tour.",
    "A parent teaches road safety rules to their child at a busy intersection.",
    "A conference speaker presents research findings at an international science conference.",
    "A fitness coach instructs proper exercise techniques in a group workout class at the gym.",
    "An author reads excerpts from their latest book at a bookstore signing event.",
    "A project leader outlines the project phases to the project team in an online meeting.",
    "A chef demonstrates a new cooking technique to apprentices in the restaurant kitchen."
]

# Energy Animals
# Event descriptions for Consume
C_event_descriptions = [
    "A college student researches various sources for a term paper on the university's online library database.",
    "An amateur astronomer observes planetary movements through a telescope in their backyard observatory.",
    "A culinary enthusiast experiments with new recipes in their home kitchen.",
    "A technology buff explores the latest gadgets at a technology expo.",
    "A fashion blogger scours fashion magazines and websites for the upcoming trends.",
    "An entrepreneur attends workshops on innovative business models at a startup conference.",
    "A nature photographer captures rare wildlife in remote natural habitats.",
    "A language learner practices conversational skills with a native speaker via an online platform.",
    "A history teacher visits historical sites during a sabbatical in Europe.",
    "A music student learns new compositions by watching tutorials on a digital music platform."
]
# Event descriptions for Play
P_event_descriptions = [
    "A group of engineers collaborates on designing a new prototype in an open-concept innovation lab.",
    "Children participate in interactive science experiments at a community science fair.",
    "A team of marketers brainstorms creative campaign ideas during a creative workshop session.",
    "A family engages in a board game night in their living room.",
    "Software developers host a hackathon at a tech co-working space.",
    "A group of tourists explores a city's landmarks on a guided walking tour.",
    "A sports team practices team strategies on the field during an evening session.",
    "A band jams new song ideas in a home studio.",
    "Co-workers participate in a team-building exercise at a corporate retreat.",
    "Friends create a collaborative art project at a local community center."
]

# Event descriptions for Sleep
S_event_descriptions = [
    "A novelist reflects on plot developments in their quiet home office.",
    "A researcher analyzes experimental data alone in the laboratory after hours.",
    "A musician contemplates new melodies in a secluded studio space.",
    "A psychologist reflects on patient interactions in a personal journal at their home office.",
    "A student reviews lecture notes in a quiet corner of the library.",
    "An artist ponders themes for a new series in their personal studio surrounded by canvases.",
    "A strategic planner evaluates future scenarios in a silent, contemplative office setting.",
    "A programmer thinks through code architecture at their desk during the late-night hours.",
    "A philosopher meditates on ethical dilemmas in a peaceful garden.",
    "A teacher reflects on teaching methods in the solitude of their classroom after school."
]



# Information Animals
B_embeddings = create_event_embeddings(B_event_descriptions)
C_embeddings = create_event_embeddings(C_event_descriptions)

# Energy Animals
P_embeddings = create_event_embeddings(P_event_descriptions)
S_embeddings = create_event_embeddings(S_event_descriptions)

# Split into training and validation sets
B_train_embeddings, B_val_embeddings = train_test_split(B_embeddings, test_size=0.2, random_state=42)
C_train_embeddings, C_val_embeddings = train_test_split(C_embeddings, test_size=0.2, random_state=42)

P_train_embeddings, P_val_embeddings = train_test_split(P_embeddings, test_size=0.2, random_state=42)
S_train_embeddings, S_val_embeddings = train_test_split(S_embeddings, test_size=0.2, random_state=42)


# Combine the training and validation sets for each class into dictionaries for easy access
train_events = {'B': B_train_embeddings, 'C': C_train_embeddings, 'P': P_train_embeddings, 'S': S_train_embeddings}
val_events = {'B': B_val_embeddings, 'C': C_val_embeddings, 'P': P_val_embeddings, 'S': S_val_embeddings}

# Split event representations into training and validation sets
# TODO we need a function to split the data
# train_events, val_events = split_data(event_representations, train_ratio=0.8)

# Training loop
num_episodes = 15
n_support = 5
n_query = 5
# Number of examples per class in the support set for validation
n_support_val = 1

# Number of examples per class in the query set for validation
n_query_val = 1

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
    validate_proto_net(proto_net_BC, val_events, 'BC', n_support_val, n_query_val)
    validate_proto_net(proto_net_PS, val_events, 'PS', n_support_val, n_query_val)




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





"""
B: Blast Event Representations

Subject: A manager
Action: Conducts
Object(s): A training session
Environment: In the company's conference room
Event Representation: "A manager conducts a training session in the company's conference room."

Subject: A teacher
Action: Explains
Object(s): Mathematical concepts
Environment: To a high school class during algebra lesson
Event Representation: "A teacher explains mathematical concepts to a high school class during an algebra lesson."

Subject: A software developer
Action: Demonstrates
Object(s): The new application features
Environment: During a team meeting in the development lab
Event Representation: "A software developer demonstrates the new application features during a team meeting in the development lab."

Subject: A tour guide
Action: Describes
Object(s): Historical landmarks
Environment: To tourists on a city walking tour
Event Representation: "A tour guide describes historical landmarks to tourists on a city walking tour."

Subject: A parent
Action: Teaches
Object(s): Road safety rules
Environment: To their child at a busy intersection
Event Representation: "A parent teaches road safety rules to their child at a busy intersection."

Subject: A conference speaker
Action: Presents
Object(s): Research findings
Environment: At an international science conference
Event Representation: "A conference speaker presents research findings at an international science conference."

Subject: A fitness coach
Action: Instructs
Object(s): Proper exercise techniques
Environment: In a group workout class at the gym
Event Representation: "A fitness coach instructs proper exercise techniques in a group workout class at the gym."

Subject: An author
Action: Reads
Object(s): Excerpts from their latest book
Environment: At a bookstore signing event
Event Representation: "An author reads excerpts from their latest book at a bookstore signing event."

Subject: A project leader
Action: Outlines
Object(s): The project phases
Environment: To the project team in an online meeting
Event Representation: "A project leader outlines the project phases to the project team in an online meeting."

Subject: A chef
Action: Demonstrates
Object(s): A new cooking technique
Environment: To apprentices in the restaurant kitchen
Event Representation: "A chef demonstrates a new cooking technique to apprentices in the restaurant kitchen."

"""

"""
S: Sleep Event Representations

Subject: A novelist
Action: Reflects on
Object(s): Plot developments
Environment: In their quiet home office
Event Representation: "A novelist reflects on plot developments in their quiet home office."

Subject: A researcher
Action: Analyzes
Object(s): Experimental data
Environment: Alone in the laboratory after hours
Event Representation: "A researcher analyzes experimental data alone in the laboratory after hours."

Subject: A musician
Action: Contemplates
Object(s): New melodies
Environment: In a secluded studio space
Event Representation: "A musician contemplates new melodies in a secluded studio space."

Subject: A psychologist
Action: Reflects on
Object(s): Patient interactions
Environment: In a personal journal at their home office
Event Representation: "A psychologist reflects on patient interactions in a personal journal at their home office."

Subject: A student
Action: Reviews
Object(s): Lecture notes
Environment: In a quiet corner of the library
Event Representation: "A student reviews lecture notes in a quiet corner of the library."

Subject: An artist
Action: Ponders
Object(s): Themes for a new series
Environment: In their personal studio surrounded by canvases
Event Representation: "An artist ponders themes for a new series in their personal studio surrounded by canvases."

Subject: A strategic planner
Action: Evaluates
Object(s): Future scenarios
Environment: In a silent, contemplative office setting
Event Representation: "A strategic planner evaluates future scenarios in a silent, contemplative office setting."

Subject: A programmer
Action: Thinks through
Object(s): Code architecture
Environment: At their desk during the late-night hours
Event Representation: "A programmer thinks through code architecture at their desk during the late-night hours."

Subject: A philosopher
Action: Meditates on
Object(s): Ethical dilemmas
Environment: In a peaceful garden
Event Representation: "A philosopher meditates on ethical dilemmas in a peaceful garden."

Subject: A teacher
Action: Reflects on
Object(s): Teaching methods
Environment: In the solitude of their classroom after school
Event Representation: "A teacher reflects on teaching methods in the solitude of their classroom after school."

"""

"""
C: Consume Event Representations

Subject: A college student
Action: Researches
Object(s): Various sources for a term paper
Environment: On the university's online library database
Event Representation: "A college student researches various sources for a term paper on the university's online library database."

Subject: An amateur astronomer
Action: Observes
Object(s): Planetary movements
Environment: Through a telescope in their backyard observatory
Event Representation: "An amateur astronomer observes planetary movements through a telescope in their backyard observatory."

Subject: A culinary enthusiast
Action: Experiments with
Object(s): New recipes
Environment: In their home kitchen
Event Representation: "A culinary enthusiast experiments with new recipes in their home kitchen."

Subject: A technology buff
Action: Explores
Object(s): The latest gadgets
Environment: At a technology expo
Event Representation: "A technology buff explores the latest gadgets at a technology expo."

Subject: A fashion blogger
Action: Scours
Object(s): Fashion magazines and websites
Environment: For the upcoming trends
Event Representation: "A fashion blogger scours fashion magazines and websites for the upcoming trends."

Subject: An entrepreneur
Action: Attends
Object(s): Workshops on innovative business models
Environment: At a startup conference
Event Representation: "An entrepreneur attends workshops on innovative business models at a startup conference."

Subject: A nature photographer
Action: Captures
Object(s): Rare wildlife
Environment: In remote natural habitats
Event Representation: "A nature photographer captures rare wildlife in remote natural habitats."

Subject: A language learner
Action: Practices
Object(s): Conversational skills
Environment: With a native speaker via an online platform
Event Representation: "A language learner practices conversational skills with a native speaker via an online platform."

Subject: A history teacher
Action: Visits
Object(s): Historical sites
Environment: During a sabbatical in Europe
Event Representation: "A history teacher visits historical sites during a sabbatical in Europe."

Subject: A music student
Action: Learns
Object(s): New compositions
Environment: By watching tutorials on a digital music platform
Event Representation: "A music student learns new compositions by watching tutorials on a digital music platform."

"""

"""
P: Play Event Representations

Subject: A group of engineers
Action: Collaborate on
Object(s): Designing a new prototype
Environment: In an open-concept innovation lab
Event Representation: "A group of engineers collaborates on designing a new prototype in an open-concept innovation lab."

Subject: Children
Action: Participate in
Object(s): Interactive science experiments
Environment: At a community science fair
Event Representation: "Children participate in interactive science experiments at a community science fair."

Subject: A team of marketers
Action: Brainstorms
Object(s): Creative campaign ideas
Environment: During a creative workshop session
Event Representation: "A team of marketers brainstorms creative campaign ideas during a creative workshop session."

Subject: A family
Action: Engages in
Object(s): A board game night
Environment: In their living room
Event Representation: "A family engages in a board game night in their living room."

Subject: Software developers
Action: Host
Object(s): A hackathon
Environment: At a tech co-working space
Event Representation: "Software developers host a hackathon at a tech co-working space."

Subject: A group of tourists
Action: Explore
Object(s): A city's landmarks
Environment: On a guided walking tour
Event Representation: "A group of tourists explores a city's landmarks on a guided walking tour."

Subject: A sports team
Action: Practices
Object(s): Team strategies
Environment: On the field during an evening session
Event Representation: "A sports team practices team strategies on the field during an evening session."

Subject: A band
Action: Jams
Object(s): New song ideas
Environment: In a home studio
Event Representation: "A band jams new song ideas in a home studio."

Subject: Co-workers
Action: Participate in
Object(s): A team-building exercise
Environment: At a corporate retreat
Event Representation: "Co-workers participate in a team-building exercise at a corporate retreat."

Subject: Friends
Action: Create
Object(s): A collaborative art project
Environment: At a local community center
Event Representation: "Friends create a collaborative art project at a local community center."

"""