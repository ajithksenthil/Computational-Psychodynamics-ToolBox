from wordlevelprotonet import prototypes, model

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
