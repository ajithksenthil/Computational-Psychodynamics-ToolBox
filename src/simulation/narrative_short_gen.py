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
import spacy
import re
# import torch
# import en_core_web_lg
import sys
import os

# Add the parent directory to sys.path to access modules in sibling directories
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from data_annotation.wordlevelprotnet import classify_new_words, proto_net_BC, proto_net_PS, prototype_tensor_BC, prototype_tensor_PS, word2vec_model
from openai import OpenAI

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Access the API key
api_key = os.getenv("OPENAI_API_KEY")
print(api_key)
openai.api_key = api_key
client = OpenAI()


# global vars
first_time = True
current_state = {}
story = ""
behavior_states = {}



characters = []



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



# Use the prototype network to find the specific action that is most similar to the next behavior state
# actions = [set of likely next events as action words]
def find_most_similar_action(next_behavior_state, characters, actions, subjectlist, word2vec_model, proto_net_BC, proto_net_PS, prototype_tensor_BC, prototype_tensor_PS):
    # Classify actions using the prototype networks
    predicted_classes_BC, predicted_classes_PS = classify_new_words(actions, word2vec_model, proto_net_BC, proto_net_PS, prototype_tensor_BC, prototype_tensor_PS)
    
    # Combine BC and PS predictions to classify actions into BP, BS, CP, CS
    action_classes = []
    for bc, ps in zip(predicted_classes_BC, predicted_classes_PS):
        if bc == 0 and ps == 0:
            action_classes.append('BP')
        elif bc == 0 and ps == 1:
            action_classes.append('BS')
        elif bc == 1 and ps == 0:
            action_classes.append('CP')
        else:  # bc == 1 and ps == 1
            action_classes.append('CS')
    
    # Determine the most similar action based on the desired next_behavior_state
    similarity_scores = {action: (1 if action_class == next_behavior_state else 0) for action, action_class in zip(actions, action_classes)}
    
    # Select the action(s) with the highest similarity score
    max_score_actions = [action for action, score in similarity_scores.items() if score == max(similarity_scores.values())]
    
    # If there are multiple actions with the highest score, select one at random
    next_action = random.choice(max_score_actions) if max_score_actions else random.choice(actions)
    action_index = actions.index(next_action)

    print(f"Selected action: {next_action}")
    return next_action, action_index


def initTransitionProbs(characters):
    global transition_probabilities
    transition_probabilities = {character: [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]] for character in characters}



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
    if personality == "bpsc": # 1
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
    elif personality == "scbp": #15
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
    elif personality == "cbsp": #22
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
    eventrepresentation = eventlist[actionindex]
  except:
    # did not find the correct action index, need to fix this bug to keep indexes between events and actions consistent
    eventrepresentation = eventlist[len(eventlist) - 1]
  queryAction = f"Take this event representation: {eventrepresentation} and write the next event that can be appended to continue this story: {story} with these characters: {characters}"
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

   


personality_list = ["bpsc", "bpcs", "cbsp", "cpsb", "sbcp", "sbpc", "pcbs", "pcsb", "scbp", "scpb", "cbps", "csbp",
                    "bpsc", "bpsc", "spbc", "spcb"]
story_image = True  # Just a placeholder, we need to build the image using PIL


   
import random


def main():
    current_state = {}
    behavior_states = {}
    characters = []
    global story
    story = ""
    story_image = ""  # supposed to be a png file from DallE
    setting = ""
    genre = ""
    worldbuilding = {"setting": setting, "genre": genre}

  


if __name__ == "__main__":
    main()



