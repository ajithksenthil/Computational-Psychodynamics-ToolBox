import numpy as np
import openai
# import openai_secret_manager
import tkinter as tk
from tkinter import *
from tkinter import ttk
from io import BytesIO
import PIL.Image
from tkinter import *
from PIL import ImageTk, Image
import requests
# import spacy
import re
# import torch
# import en_core_web_lg
from openai import OpenAI

from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Access the API key
api_key = os.getenv("OPENAI_API_KEY")
print(api_key)
openai.api_key = api_key
client = OpenAI()
import gensim
import gensim.downloader as api
from gensim.models.word2vec import Word2Vec

# Download the "text8" dataset
dataset = api.load("text8")

# Extract the data and create a word2vec model
# data = [dataset[0]]  # The data is stored in the first element of the dataset tuple
model = Word2Vec(dataset)

# model = api.load("book-corpus-large-cased")
# model = api.load("text8")

# model = gensim.models.Word2Vec.load("word2vec.model")
# nlp = en_core_web_lg.load()

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# global vars
first_time = True
current_state = {}
story = ""
behavior_states = {}

# Define the word clouds for each binary pair
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
# Convert the word clouds into word vectors using a semantic network
# B_vectors = [model.wv[word] for word in B_word_cloud if model.wv.get_vector(word) is not None]
# C_vectors = [model.wv[word] for word in C_word_cloud if model.wv.get_vector(word) is not None]
# P_vectors = [model.wv[word] for word in P_word_cloud if model.wv.get_vector(word) is not None]
# S_vectors = [model.wv[word] for word in S_word_cloud if model.wv.get_vector(word) is not None]

# B_vectors = [model.wv[word] for word in b_word_cloud_lower if model.wv.get_vector(word) is not None]
# C_vectors = [model.wv[word] for word in c_word_cloud_lower if model.wv.get_vector(word) is not None]
# P_vectors = [model.wv[word] for word in p_word_cloud_lower if model.wv.get_vector(word) is not None]
# S_vectors = [model.wv[word] for word in s_word_cloud_lower if model.wv.get_vector(word) is not None]

B_vectors = []
for word in b_word_cloud_lower:
    try:
        vector = model.wv[word]
    except KeyError:
        # Word is not in the vocabulary, skip it
        continue
    B_vectors.append(vector)

C_vectors = []
for word in c_word_cloud_lower:
    try:
        vector = model.wv[word]
    except KeyError:
        # Word is not in the vocabulary, skip it
        continue
    C_vectors.append(vector)

P_vectors = []
for word in p_word_cloud_lower:
    try:
        vector = model.wv[word]
    except KeyError:
        # Word is not in the vocabulary, skip it
        continue
    P_vectors.append(vector)

S_vectors = []
for word in s_word_cloud_lower:
    try:
        vector = model.wv[word]
    except KeyError:
        # Word is not in the vocabulary, skip it
        continue
    S_vectors.append(vector)

# need to fill with optimized weights
weights = {}  # contains the weights for each word in the lattice provided by the initial word clouds

# get the action events

# list of dictionaries, each dictionary is an labeled action event with the action, subject, objects, and a key for the animals
# actionevents = []

characters = []


# Find the Animal scores after training with self supervision
# this is not used for initial prototype, it is for the optimized word clouds once trained with labeled data
def findWeightedAnimalScore(actionevents, lattices):
    for ae in actionevents:
        action = ae["action"]
        subject = ae["subject"]
        objects = ae["objects"]
        animals = []

        # should be 4 lattices one for each animal
        for lattice_index in range(len(lattices)):
            lattice = lattices[lattice_index]
            animal = []
            # lattice = nlp(lattice)
            # action = nlp(action)
            # number of words in each word cloud

            # how should we add the weights and initialize them randomly probably
            for word_index in range(len(lattice)):
                word = lattice[word_index]
                if word.has_vector() and action.has_vector():
                    animal.append(weights[lattice_index][word_index] * action.similarity(word))
            animal_score = np.mean(animal)
            animals.append(animal_score)
        ae["animals"] = animals


def gpt3(stext):
    response = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": stext}
        ]
    )
    # Assuming the response format, adapt accordingly if different.
    return response.choices[0].message.content


def parse_output(output):
    print("parse output", output)
    tuples_list = []
    lines = output.split('\n')
    print("lines", lines)
    for line in lines:
        match = re.search(r"Action: (.+); Subject: (.+); Object: (.+)", line)
        print("match", match)
        if match:
            action = match.group(1)
            subject = match.group(2)
            object = match.group(3)
            tuples_list.append((action, subject, object))
    return tuples_list
 


def determine_next_action(story, characters, current_state, behavior_states):
    characters = characters
    # Use GPT to generate a list of likely next events based on the context of the story
    likely_events, actions, eventlistresp, subjectList = generate_likely_events(story, characters)
    print("likely_events", likely_events)
    print("for real actions", actions)
    # event_tuples_list = parse_output(likely_events)
    # event_tuples_list = likely_events
    eventlist = eventlistresp.split(",")
    # print("event_tuples_list", event_tuples_list)
    # actions = [x[0] for x in event_tuples_list]

    print("eventlist", eventlist)

    # Use the hidden Markov model to determine the likely next behavior state
    next_behavior_state = determine_next_behavior_state(current_state, likely_events, characters, behavior_states)

    # Use the semantic network to find the specific action that is most similar to the next behavior state
    next_action, actionindex = find_most_similar_action(next_behavior_state, characters, actions, subjectList)

    return next_action, actionindex, eventlist


# Use GPT to generate a list of likely next events based on the context of the story
def generate_likely_events(story, characters):
    characters = characters
    # Set up the GPT model
    storyInput = story
    query = f"Write a set of possible next events for a story, in which each event is a potential branch that could happen independently of the others. The events should not be sequential, but rather parallel paths that the following story could potentially take: {storyInput} with these characters: {characters}"
    response = gpt3(query)
    # print(response)
    query2 = f"For each of the following events, break it down into its component parts into a list and format them into a list in python: the action, the subject performing the action, and the object(s) affected by the action: {response}"
    response2 = gpt3(query2)
    print("response2", response2)

    queryEventList = f"Take this list of events {response} and format it to be one python string where each event is separated by commas like so 'event,event,event'"
    eventListResponse = gpt3(queryEventList)

    query3 = f"Using the provided list of events, please extract the single verb that represents the actions in each event and format them in the same order into a list in python like so 'verb, verb, verb, ...': {response2}"
    response3 = gpt3(query3)
    print("response3", type(response3))
    actions = response3.split(", ")
    actions = [s.strip() for s in actions]
    # actions = [i.splitlines()[0] for i in actions]
    print("actions 226", type(actions), actions)
    event_list = response2

    querySubj = f"Using the provided list of events, please extract the subject performing the action in each event and format them in the same order into a list in python like so 'subject, subject, subject, ...' here is the list of events: {response2}"
    subjectlist = gpt3(querySubj)
    subjectlist = subjectlist.strip().split(", ")
    print("subjectlist", subjectlist)

    return event_list, actions, eventListResponse, subjectlist


# Use the hidden Markov model for each character to determine the likely next behavior state
def determine_next_behavior_state(current_state, events, characters, behavior_states):
    next_behavior_states = {}
    print("characters", characters)

    for character in characters:
        # Get the current behavior state for the character
        current_behavior_state = current_state[character]

        # Use the hidden Markov model for the character to determine the likely next behavior state
        print("transition_probabilities", transition_probabilities)
        next_behavior_state = np.random.choice(behavior_states[character], p=transition_probabilities[character][getBehaviorIndex(current_behavior_state)])

        # Store the likely next behavior state for the character
        next_behavior_states[character] = next_behavior_state
        current_state[character] = next_behavior_state  # updating current states for each character
    print("next_behavior_states", next_behavior_states)

    return next_behavior_states


def getSubjIndex(characters, subject):
  for i in range(len(characters)):
    if characters[i] in subject:
        print(f"{characters[i]} is a substring of {subject} and matches the index {i}")
        return i
    else:
        print(f"{characters[i]} is not a substring of {subject} or not match the index {i}")



# Use the semantic network to find the specific action that is most similar to the next behavior state
# actions = [set of likely next events as action words]
def find_most_similar_action(next_behavior_state, characters, actions, subjectlist):
    # Implement the logic to use the semantic network to find the specific action that is most similar to the next behavior state
    # remove later
    next_behavior_state_one_character = next_behavior_state[characters[0]]
    # Initialize a dictionary to store the similarity scores for each action
    similarity_scores = {}
    print("actions", actions)
    for actionindex in range(len(actions)): 
        action = actions[actionindex]
        try:
            next_behavior_state_one_character = next_behavior_state[characters[getSubjIndex(characters, subjectlist[actionindex])]]
        except:
            print("subject did not match")
            next_behavior_state_one_character = next_behavior_state[characters[0]]

        print("action", action)
        infosimilarity = 0
        energysimilarity = 0
        try:
            action_vector = model.wv[action]
            action_vector = action_vector.reshape(1, -1)
            print("found action vector")
        except KeyError:
            # handle the exception (e.g. skip this iteration of the loop)
            print("did not find action vector")
            continue

        if next_behavior_state_one_character == "BS":
            infosimilarity = cosine_similarity(action_vector, B_vectors)
            energysimilarity = cosine_similarity(action_vector, S_vectors)
            similarity_scores[action] = np.mean(infosimilarity) + np.mean(energysimilarity)

        elif next_behavior_state_one_character == "BP":
            infosimilarity = cosine_similarity(action_vector, B_vectors)
            energysimilarity = cosine_similarity(action_vector, P_vectors)
            similarity_scores[action] = np.mean(infosimilarity) + np.mean(energysimilarity)

        elif next_behavior_state_one_character == "CS":
            infosimilarity = cosine_similarity(action_vector, C_vectors)
            energysimilarity = cosine_similarity(action_vector, S_vectors)
            similarity_scores[action] = np.mean(infosimilarity) + np.mean(energysimilarity)

        elif next_behavior_state_one_character == "CP":
            infosimilarity = cosine_similarity(action_vector, C_vectors)
            energysimilarity = cosine_similarity(action_vector, P_vectors)
            similarity_scores[action] = np.mean(infosimilarity) + np.mean(energysimilarity)

    # Find the action with the highest similarity score
    print("similarity scores", similarity_scores)

    # use try and except statement and assign a random similarity score matrix the correct size.
    try: 
        next_action = max(similarity_scores, key=similarity_scores.get)
        action_index = actions.index(next_action)
    except:
        next_action = random.choice(actions)
        action_index = actions.index(next_action)

    print(next_action)
    return next_action, action_index


from hmmlearn import hmm


# train HMM
# modify this to use the predefined markov models
# Collect data on the behavior states of each character over time
# character_data = {
#   'Alice': ['BP', 'BS', 'CP', 'BP', 'BS', 'CP', 'BP'],
#   'Bob': ['BS', 'BP', 'CP', 'BS', 'BP', 'CP', 'BS']
# }

# # Preprocess the data to encode the behavior states as integers
# encoder = {}
# character_data_encoded = {}
# for character, data in character_data.items():
#   # Create an encoder to map the behavior states to integers
#   encoder[character] = {state: i for i, state in enumerate(set(data))}

#   # Encode the behavior states as integers
#   character_data_encoded[character] = [encoder[character][state] for state in data]

# # Define the set of possible behavior states and transition probabilities for each character
# # behavior_states = {
# #   'Alice': ['BP', 'BS', 'CP', 'CS'],
# #   'Bob': ['BP', 'BS', 'CP', 'CS']
# # }
# transition_probabilities = {
#   'Alice': np.array([[0.8, 0.1, 0.1, 0.0], [0.1, 0.8, 0.1, 0.0], [0.0, 0.1, 0.8, 0.1], [0.0, 0.0, 0.1, 0.9]]),
#   'Bob': np.array([[0.8, 0.1, 0.1, 0.0], [0.1, 0.8, 0.1, 0.0], [0.0, 0.1, 0.8, 0.1], [0.0, 0.0, 0.1, 0.9]])
# }

def initTransitionProbs(characters):
    global transition_probabilities
    transition_probabilities = {character: [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]] for character in characters}


# Train the hidden Markov model for each character
# character_models = {}
# for character, data in character_data_encoded.items():
#   # Create the hidden Markov model
#   model = hmm.MultinomialHMM(n_components=len(behavior_states[character]))

#   # Fit the model to the data using an iterative optimization algorithm
#   model.fit(np.array([data]).T, transition_probabilities=transition_probabilities[character])

#   # Store the trained model
#   character_models[character] = model

# hmm script portion end

# assign personalities to characters


personality_list = ["bpsc", "bpcs", "cbsp", "cpsb", "sbcp", "sbpc", "pcbs", "pcsb", "scbp", "scpb", "cbps", "csbp",
                    "bpsc", "bpsc", "spbc", "spcb"]
transition_matrices = {personality: [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]] for personality in
                       personality_list}

behavior_states_list = ['BS', 'BP', 'CS', 'CP']

# change 0.45 to 0.5

for personality in personality_list:
    # Create a matrix of zeros
    matrix = np.zeros((4, 4))

    # Fill in the matrix based on the personality
    if personality == "bpsc": # 1 demo
        matrix[0][0] = 0
        matrix[0][1] = 1
        matrix[0][2] = 0
        matrix[0][3] = 0
        matrix[1][0] = 0
        matrix[1][1] = 1
        matrix[1][2] = 0
        matrix[1][3] = 0
        matrix[2][0] = 0
        matrix[2][1] = 1
        matrix[2][2] = 0
        matrix[2][3] = 0
        matrix[3][0] = 0
        matrix[3][1] = 1
        matrix[3][2] = 0
        matrix[3][3] = 0
    elif personality == "sbpc": # 2 demo
        matrix[0][0] = 1
        matrix[0][1] = 0
        matrix[0][2] = 0
        matrix[0][3] = 0
        matrix[1][0] = 1
        matrix[1][1] = 0
        matrix[1][2] = 0
        matrix[1][3] = 0
        matrix[2][0] = 1
        matrix[2][1] = 0
        matrix[2][2] = 0
        matrix[2][3] = 0
        matrix[3][0] = 1
        matrix[3][1] = 0
        matrix[3][2] = 0
        matrix[3][3] = 0
    elif personality == "cpsb": # 3 demo
        matrix[0][0] = 0
        matrix[0][1] = 0
        matrix[0][2] = 0
        matrix[0][3] = 1
        matrix[1][0] = 0
        matrix[1][1] = 0
        matrix[1][2] = 0
        matrix[1][3] = 1
        matrix[2][0] = 0
        matrix[2][1] = 0
        matrix[2][2] = 0
        matrix[2][3] = 1
        matrix[3][0] = 0
        matrix[3][1] = 0
        matrix[3][2] = 0
        matrix[3][3] = 1
    elif personality == "csbp": # 4 demo
        matrix[0][0] = 0
        matrix[0][1] = 0
        matrix[0][2] = 1
        matrix[0][3] = 0
        matrix[1][0] = 0
        matrix[1][1] = 0
        matrix[1][2] = 1
        matrix[1][3] = 0
        matrix[2][0] = 0
        matrix[2][1] = 0
        matrix[2][2] = 1
        matrix[2][3] = 0
        matrix[3][0] = 0
        matrix[3][1] = 0
        matrix[3][2] = 1
        matrix[3][3] = 0
    elif personality == "bpsc": # 1
        matrix[0][0] = 0.25
        matrix[0][1] = 0.5
        matrix[0][2] = 0.15
        matrix[0][3] = 0.1
        matrix[1][0] = 0.25
        matrix[1][1] = 0.5
        matrix[1][2] = 0.15
        matrix[1][3] = 0.1
        matrix[2][0] = 0.25
        matrix[2][1] = 0.5
        matrix[2][2] = 0.15
        matrix[2][3] = 0.1
        matrix[3][0] = 0.25
        matrix[3][1] = 0.5
        matrix[3][2] = 0.15
        matrix[3][3] = 0.1
    elif personality == "bpcs": # 2
        matrix[0][0] = 0.25
        matrix[0][1] = 0.5
        matrix[0][2] = 0.1
        matrix[0][3] = 0.15
        matrix[1][0] = 0.25
        matrix[1][1] = 0.5
        matrix[1][2] = 0.1
        matrix[1][3] = 0.15
        matrix[2][0] = 0.25
        matrix[2][1] = 0.5
        matrix[2][2] = 0.1
        matrix[2][3] = 0.15
        matrix[3][0] = 0.25
        matrix[3][1] = 0.5
        matrix[3][2] = 0.1
        matrix[3][3] = 0.15
    elif personality == "bpcs": #3
        matrix[0][0] = 0.15
        matrix[0][1] = 0.5
        matrix[0][2] = 0.1
        matrix[0][3] = 0.25
        matrix[1][0] = 0.15
        matrix[1][1] = 0.5
        matrix[1][2] = 0.1
        matrix[1][3] = 0.25
        matrix[2][0] = 0.15
        matrix[2][1] = 0.5
        matrix[2][2] = 0.1
        matrix[2][3] = 0.25
        matrix[3][0] = 0.15
        matrix[3][1] = 0.5
        matrix[3][2] = 0.1
        matrix[3][3] = 0.25
    elif personality == "bpcs": #4
        matrix[0][0] = 0.15
        matrix[0][1] = 0.5
        matrix[0][2] = 0.25
        matrix[0][3] = 0.1
        matrix[1][0] = 0.15
        matrix[1][1] = 0.5
        matrix[1][2] = 0.25
        matrix[1][3] = 0.1
        matrix[2][0] = 0.15
        matrix[2][1] = 0.5
        matrix[2][2] = 0.25
        matrix[2][3] = 0.1
        matrix[3][0] = 0.15
        matrix[3][1] = 0.5
        matrix[3][2] = 0.25
        matrix[3][3] = 0.1
    elif personality == "bscp": #5
        matrix[0][0] = 0.5
        matrix[0][1] = 0.15
        matrix[0][2] = 0.25
        matrix[0][3] = 0.1
        matrix[1][0] = 0.5
        matrix[1][1] = 0.15
        matrix[1][2] = 0.25
        matrix[1][3] = 0.1
        matrix[2][0] = 0.5
        matrix[2][1] = 0.15
        matrix[2][2] = 0.25
        matrix[2][3] = 0.1
        matrix[3][0] = 0.5
        matrix[3][1] = 0.15
        matrix[3][2] = 0.25
        matrix[3][3] = 0.1
    elif personality == "bspc": #6
        matrix[0][0] = 0.5
        matrix[0][1] = 0.25
        matrix[0][2] = 0.15
        matrix[0][3] = 0.1
        matrix[1][0] = 0.5
        matrix[1][1] = 0.25
        matrix[1][2] = 0.15
        matrix[1][3] = 0.1
        matrix[2][0] = 0.5
        matrix[2][1] = 0.25
        matrix[2][2] = 0.15
        matrix[2][3] = 0.1
        matrix[3][0] = 0.5
        matrix[3][1] = 0.25
        matrix[3][2] = 0.15
        matrix[3][3] = 0.1
    elif personality == "bspc": #7
        matrix[0][0] = 0.5
        matrix[0][1] = 0.25
        matrix[0][2] = 0.15
        matrix[0][3] = 0.1
        matrix[1][0] = 0.5
        matrix[1][1] = 0.25
        matrix[1][2] = 0.15
        matrix[1][3] = 0.1
        matrix[2][0] = 0.5
        matrix[2][1] = 0.25
        matrix[2][2] = 0.15
        matrix[2][3] = 0.1
        matrix[3][0] = 0.5
        matrix[3][1] = 0.25
        matrix[3][2] = 0.15
        matrix[3][3] = 0.1
    elif personality == "bspc": #8
        matrix[0][0] = 0.5
        matrix[0][1] = 0.1
        matrix[0][2] = 0.25
        matrix[0][3] = 0.15
        matrix[1][0] = 0.5
        matrix[1][1] = 0.1
        matrix[1][2] = 0.25
        matrix[1][3] = 0.15
        matrix[2][0] = 0.5
        matrix[2][1] = 0.1
        matrix[2][2] = 0.25
        matrix[2][3] = 0.15
        matrix[3][0] = 0.5
        matrix[3][1] = 0.1
        matrix[3][2] = 0.25
        matrix[3][3] = 0.15
    elif personality == "cspb": #9
        matrix[0][0] = 0.15
        matrix[0][1] = 0.1
        matrix[0][2] = 0.25
        matrix[0][3] = 0.5
        matrix[1][0] = 0.15
        matrix[1][1] = 0.1
        matrix[1][2] = 0.25
        matrix[1][3] = 0.5
        matrix[2][0] = 0.15
        matrix[2][1] = 0.1
        matrix[2][2] = 0.25
        matrix[2][3] = 0.5
        matrix[3][0] = 0.15
        matrix[3][1] = 0.1
        matrix[3][2] = 0.25
        matrix[3][3] = 0.5
    elif personality == "cspb": #10
        matrix[0][0] = 0.1
        matrix[0][1] = 0.15
        matrix[0][2] = 0.25
        matrix[0][3] = 0.5
        matrix[1][0] = 0.1
        matrix[1][1] = 0.15
        matrix[1][2] = 0.25
        matrix[1][3] = 0.5
        matrix[2][0] = 0.1
        matrix[2][1] = 0.15
        matrix[2][2] = 0.25
        matrix[2][3] = 0.5
        matrix[3][0] = 0.1
        matrix[3][1] = 0.15
        matrix[3][2] = 0.25
        matrix[3][3] = 0.5
    elif personality == "csbp": #11
        matrix[0][0] = 0.1
        matrix[0][1] = 0.25
        matrix[0][2] = 0.15
        matrix[0][3] = 0.5
        matrix[1][0] = 0.1
        matrix[1][1] = 0.25
        matrix[1][2] = 0.15
        matrix[1][3] = 0.5
        matrix[2][0] = 0.1
        matrix[2][1] = 0.25
        matrix[2][2] = 0.15
        matrix[2][3] = 0.5
        matrix[3][0] = 0.1
        matrix[3][1] = 0.25
        matrix[3][2] = 0.15
        matrix[3][3] = 0.5
    elif personality == "cpsb": #12
        matrix[0][0] = 0.25
        matrix[0][1] = 0.15
        matrix[0][2] = 0.1
        matrix[0][3] = 0.5
        matrix[1][0] = 0.25
        matrix[1][1] = 0.15
        matrix[1][2] = 0.1
        matrix[1][3] = 0.5
        matrix[2][0] = 0.25
        matrix[2][1] = 0.15
        matrix[2][2] = 0.1
        matrix[2][3] = 0.5
        matrix[3][0] = 0.25
        matrix[3][1] = 0.15
        matrix[3][2] = 0.1
        matrix[3][3] = 0.5
    elif personality == "cpbs": #13
        matrix[0][0] = 0.25
        matrix[0][1] = 0.15
        matrix[0][2] = 0.5
        matrix[0][3] = 0.1
        matrix[1][0] = 0.25
        matrix[1][1] = 0.15
        matrix[1][2] = 0.5
        matrix[1][3] = 0.1
        matrix[2][0] = 0.25
        matrix[2][1] = 0.15
        matrix[2][2] = 0.5
        matrix[2][3] = 0.1
        matrix[3][0] = 0.25
        matrix[3][1] = 0.15
        matrix[3][2] = 0.5
        matrix[3][3] = 0.1
    elif personality == "scpb": #14
        matrix[0][0] = 0.15
        matrix[0][1] = 0.25
        matrix[0][2] = 0.5
        matrix[0][3] = 0.1
        matrix[1][0] = 0.15
        matrix[1][1] = 0.25
        matrix[1][2] = 0.5
        matrix[1][3] = 0.1
        matrix[2][0] = 0.15
        matrix[2][1] = 0.25
        matrix[2][2] = 0.5
        matrix[2][3] = 0.1
        matrix[3][0] = 0.15
        matrix[3][1] = 0.25
        matrix[3][2] = 0.5
        matrix[3][3] = 0.1
    elif personality == "scpb": #15
        matrix[0][0] = 0.1
        matrix[0][1] = 0.15
        matrix[0][2] = 0.5
        matrix[0][3] = 0.25
        matrix[1][0] = 0.1
        matrix[1][1] = 0.15
        matrix[1][2] = 0.5
        matrix[1][3] = 0.25
        matrix[2][0] = 0.1
        matrix[2][1] = 0.15
        matrix[2][2] = 0.5
        matrix[2][3] = 0.25
        matrix[3][0] = 0.1
        matrix[3][1] = 0.15
        matrix[3][2] = 0.5
        matrix[3][3] = 0.25
    elif personality == "scpb": #16
        matrix[0][0] = 0.15
        matrix[0][1] = 0.1
        matrix[0][2] = 0.5
        matrix[0][3] = 0.25
        matrix[1][0] = 0.15
        matrix[1][1] = 0.1
        matrix[1][2] = 0.5
        matrix[1][3] = 0.25
        matrix[2][0] = 0.15
        matrix[2][1] = 0.1
        matrix[2][2] = 0.5
        matrix[2][3] = 0.25
        matrix[3][0] = 0.15
        matrix[3][1] = 0.2
        matrix[3][2] = 0.5
        matrix[3][3] = 0.25
    elif personality == "bpsc": #17
        matrix[0][0] = 0.1
        matrix[0][1] = 0.5
        matrix[0][2] = 0.15
        matrix[0][3] = 0.25
        matrix[1][0] = 0.1
        matrix[1][1] = 0.5
        matrix[1][2] = 0.15
        matrix[1][3] = 0.25
        matrix[2][0] = 0.1
        matrix[2][1] = 0.5
        matrix[2][2] = 0.15
        matrix[2][3] = 0.25
        matrix[3][0] = 0.1
        matrix[3][1] = 0.5
        matrix[3][2] = 0.15
        matrix[3][3] = 0.25
    elif personality == "spcb": #18
        matrix[0][0] = 0.1
        matrix[0][1] = 0.5
        matrix[0][2] = 0.25
        matrix[0][3] = 0.15
        matrix[1][0] = 0.1
        matrix[1][1] = 0.5
        matrix[1][2] = 0.25
        matrix[1][3] = 0.15
        matrix[2][0] = 0.1
        matrix[2][1] = 0.5
        matrix[2][2] = 0.25
        matrix[2][3] = 0.15
        matrix[3][0] = 0.1
        matrix[3][1] = 0.5
        matrix[3][2] = 0.25
        matrix[3][3] = 0.15
    elif personality == "bscp": #19
        matrix[0][0] = 0.5
        matrix[0][1] = 0.15
        matrix[0][2] = 0.1
        matrix[0][3] = 0.25
        matrix[1][0] = 0.5
        matrix[1][1] = 0.15
        matrix[1][2] = 0.1
        matrix[1][3] = 0.25
        matrix[2][0] = 0.5
        matrix[2][1] = 0.15
        matrix[2][2] = 0.1
        matrix[2][3] = 0.25
        matrix[3][0] = 0.5
        matrix[3][1] = 0.15
        matrix[3][2] = 0.1
        matrix[3][3] = 0.25
    elif personality == "bscp": #20
        matrix[0][0] = 0.5
        matrix[0][1] = 0.25
        matrix[0][2] = 0.1
        matrix[0][3] = 0.15
        matrix[1][0] = 0.5
        matrix[1][1] = 0.25
        matrix[1][2] = 0.1
        matrix[1][3] = 0.15
        matrix[2][0] = 0.5
        matrix[2][1] = 0.25
        matrix[2][2] = 0.1
        matrix[2][3] = 0.15
        matrix[3][0] = 0.5
        matrix[3][1] = 0.25
        matrix[3][2] = 0.1
        matrix[3][3] = 0.15
    elif personality == "csbp": #21
        matrix[0][0] = 0.15
        matrix[0][1] = 0.25
        matrix[0][2] = 0.1
        matrix[0][3] = 0.5
        matrix[1][0] = 0.15
        matrix[1][1] = 0.25
        matrix[1][2] = 0.1
        matrix[1][3] = 0.5 
        matrix[2][0] = 0.15
        matrix[2][1] = 0.25
        matrix[2][2] = 0.1
        matrix[2][3] = 0.5
        matrix[3][0] = 0.15
        matrix[3][1] = 0.25
        matrix[3][2] = 0.1
        matrix[3][3] = 0.5
    elif personality == "csbp": #22
        matrix[0][0] = 0.25
        matrix[0][1] = 0.1
        matrix[0][2] = 0.15
        matrix[0][3] = 0.5
        matrix[1][0] = 0.25
        matrix[1][1] = 0.1
        matrix[1][2] = 0.15
        matrix[1][3] = 0.5 
        matrix[2][0] = 0.25
        matrix[2][1] = 0.1
        matrix[2][2] = 0.15
        matrix[2][3] = 0.5
        matrix[3][0] = 0.25
        matrix[3][1] = 0.1
        matrix[3][2] = 0.15
        matrix[3][3] = 0.5
    elif personality == "csbp": #23
        matrix[0][0] = 0.1
        matrix[0][1] = 0.25
        matrix[0][2] = 0.5
        matrix[0][3] = 0.15
        matrix[1][0] = 0.1
        matrix[1][1] = 0.25
        matrix[1][2] = 0.5
        matrix[1][3] = 0.15
        matrix[2][0] = 0.1
        matrix[2][1] = 0.25
        matrix[2][2] = 0.5
        matrix[2][3] = 0.15
        matrix[3][0] = 0.1
        matrix[3][1] = 0.25
        matrix[3][2] = 0.5
        matrix[3][3] = 0.15
    elif personality == "csbp": #24
        matrix[0][0] = 0.25
        matrix[0][1] = 0.1
        matrix[0][2] = 0.5
        matrix[0][3] = 0.15
        matrix[1][0] = 0.25
        matrix[1][1] = 0.1
        matrix[1][2] = 0.5
        matrix[1][3] = 0.15
        matrix[2][0] = 0.25
        matrix[2][1] = 0.1
        matrix[2][2] = 0.5
        matrix[2][3] = 0.15
        matrix[3][0] = 0.25
        matrix[3][1] = 0.1
        matrix[3][2] = 0.5
        matrix[3][3] = 0.15
    transition_matrices[personality] = matrix


def assignPersonalitytoCharacter(character, personality):
    transition_probabilities[character] = transition_matrices[personality]


def update_story_data(action, actionindex, eventlist, story, characters):
  try:
    event = eventlist[actionindex]
  except:
    # did not find the correct action index, need to fix this bug to keep indexes between events and actions consistent
    event = eventlist[len(eventlist) - 1]
  queryAction = f"Take this event: {event} and write the next event that can be appended to continue this story: {story} with these characters: {characters}"
  output = gpt3(queryAction)
  return output


def getBehaviorIndex(behaviorstate):
    if str.upper(behaviorstate) == 'BS':
        return 0
    elif str.upper(behaviorstate) == 'BP':
        return 1
    elif str.upper(behaviorstate) == 'CS':
        return 2
    elif str.upper(behaviorstate) == 'CP':
        return 3


def genSceneImage(actionevent, characters):
    queryScene = f"Write a concise prompt for the scene of {actionevent} with characters {characters} for DALL-E in less than 25 words"
    imageprompt = gpt3(queryScene)
    return imageprompt





def genImagefromScene(imageprompt):

    # # Define the story
    # story = "Once upon a time, there was a young girl named Alice who went on a journey through a magical land. She met many interesting characters, such as a rabbit in a waistcoat and a caterpillar smoking a hookah. Along the way, she faced many challenges, but she always found a way to overcome them with her courage and determination."

    # Use the OpenAI API to generate the image
    prompt = imageprompt
    # prompt = imageprompt
    response = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1024x1024",
        quality="standard",
        n=1,
    )


    # Get the image data from the URL
    image_url = response.data[0].url

    # Download the image from the URL
    response = requests.get(image_url)
    img_data = PIL.Image.open(BytesIO(response.content))

    return img_data

   



from tkinter import messagebox



personality_list = ["bpsc", "bpcs", "cbsp", "cpsb", "sbcp", "sbpc", "pcbs", "pcsb", "scbp", "scpb", "cbps", "csbp",
                    "bpsc", "bpsc", "spbc", "spcb"]
story_image = True  # Just a placeholder, we need to build the image using PIL

import random


def main():
    current_state = {}
    behavior_states = {}
    characters = []
    story = ""
    story_image = ""  # supposed to be a png file from DallE
    setting = ""
    genre = ""
    worldbuilding = {"setting": setting, "genre": genre}
    

  


if __name__ == "__main__":
    main()



