import numpy as np
import openai
import sqlite3
import re
# get the list of actionevents

def gpt3(stext):
    openai.api_key = "your openai key"
    response = openai.Completion.create(
        #        engine="davinci-instruct-beta",
        engine="text-davinci-003",
        prompt=stext,
        temperature=0.1,
        max_tokens=1000,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    content = response.choices[0].text.split('.')
    # print(content)
    return response.choices[0].text


def split_into_chunks(text, max_chunk_size=1000):
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    chunks = []
    current_chunk = ''

    for sentence in sentences:
        if len(current_chunk) + len(sentence) < max_chunk_size:
            current_chunk += sentence + ' '
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + ' '

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def genEvents(story):
    # generate a list of events
    # split the story into chunks that are 1000 characters or less
    chunks = split_into_chunks(story)

    
    # take each chunk and generate a list of events
    responselist = []
    for chunk in enumerate(chunks):
        queryEventSplit = f"Split the following story into events: {story} as a list of events of the form ['event1', 'event2', 'event3', ...] where each event is a string"
        response = gpt3(queryEventSplit)
        responselist.append(response)
    
    print(responselist)
    return responselist


def labelAnimal(actionevent):
    # take this event and label it as BS, BP, CS, or CP prompting gpt
    queryLabelInformation = f"Classify the behavior of the subject described in the following event description {actionevent} using the objective personality system binary token: Blast versus Consume. The binary token is defined as follows: Blast is communicating an organized pattern/data in terms of tribe reasons/values. Blast can manifest as Teaching, controlling, directing, getting started, bragging, explaining. Consume is the opposite of Blast, Consume is gathering patterns/data of personal logical significance/personal value resonance. Consume can manifest as learning, taking in information, respecting information. Determine if the event is more similar to Blast or Consume and return the string 'Blast' if Blast is more similar and 'Consume' if Consume is more similar."
    responseInfo = gpt3(queryLabelInformation)
    # Use list comprehension to keep only alphabetic characters
    filtered_responseInfoChars = [char for char in str(responseInfo) if char.isalpha()]
    # Join the characters back into a string
    filtered_responseInfo = "".join(filtered_responseInfoChars)

    queryLabelEnergy = f"Classify the behavior of the subject described in the following event description {actionevent} using the objective personality system binary token: Play versus Sleep. The binary token is defined as follows: Play is exploring data/patterns in terms of tribe reasons/values. Play can manifest as expending energy, work, doing, showing off. Sleep is the opposite of Play, Sleep is preserves energy, processing the known information and organizing it for the self. Sleep can manifest as preserving energy, processing, introspection. Determine if the event is more similar to Play or Sleep and return the string 'Play' if Play is more similar and 'Sleep' if Sleep is more similar."
    responseEnergy = gpt3(queryLabelEnergy)
    # Use list comprehension to keep only alphabetic characters
    filtered_responseEnergyChars = [char for char in str(responseEnergy) if char.isalpha()]
    # Join the characters back into a string
    filtered_responseEnergy = "".join(filtered_responseEnergyChars)
    
    return (actionevent, filtered_responseInfo, filtered_responseEnergy)


def process_story(story):
    narrative_events = genEvents(story=story)
    labels = []
    for actionevent in narrative_events:
        label = str(labelAnimal(actionevent))
        label = label.replace('\n', '')
        labels.append(label)
    
    return labels


def main():
    # Connect to the input and output databases
    input_db = sqlite3.connect("input.db")
    output_db = sqlite3.connect("output.db")

    # Create a table for storing the output data
    output_db.execute("CREATE TABLE IF NOT EXISTS labels (file_name TEXT, label TEXT)")

    # Read text files from the input database
    input_cursor = input_db.cursor()
    input_cursor.execute("SELECT file_name, content FROM text_files")
    files = input_cursor.fetchall()

    # Process each text file and store the labels in the output database
    for file_name, content in files:
        story_labels = process_story(content)
        for label in story_labels:
            output_db.execute("INSERT INTO labels (file_name, label) VALUES (?, ?)", (file_name, label))

    # Commit the changes and close  the connections
    output_db.commit()
    input_db.close()
    output_db.close()

    


if __name__ == "__main__":
    main()
