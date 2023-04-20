import numpy as np
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from getData import findUserById

# Încărcarea modelului SpaCy
nlp = spacy.load('en_core_web_sm')

# Definirea unei funcții pentru a preprocesa textul întrebărilor și răspunsurilor
def preprocess_text(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return ' '.join(tokens)

# Definirea întrebărilor și răspunsurilor
questions = ['What is your name?', 'What is the capital of France?', 'What is the meaning of life?']
answers = ['My name is AI', 'The capital of France is Paris', 'The meaning of life is subjective']

# Preprocesarea întrebărilor și răspunsurilor
preprocessed_questions = [preprocess_text(q) for q in questions]
preprocessed_answers = [preprocess_text(a) for a in answers]

# Definirea vectorului TF-IDF pentru întrebări
vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(preprocessed_questions)

# Definirea modelului de clasificare a întrebărilor
clf = cosine_similarity(question_vectors)

# Definirea unei funcții pentru a răspunde la întrebări
def answer_question(question):
    preprocessed_question = preprocess_text(question)
    question_vector = vectorizer.transform([preprocessed_question]).toarray()
    question_vector = (question_vector > 0).astype(int) # Transformați vectorul într-un obiect numpy.ndarray cu elemente de tipul int sau bool
    similarities = clf[question_vector]
    best_match_index = np.argmax(similarities)
    return answers[best_match_index]


# Verifica daca utilizatorul are abonamentul valid
def check_user_expired(user):
    is_user_expired = user["userExpired"]
    if is_user_expired:
        print("You need to update your subscription")
    else:
        print("Welcome " + user["firstName"] + " " + user["lastName"])



# Exemplu de întrebare și răspuns
question = 'What is your name?'
answer = answer_question(question)
print(answer)

user = findUserById("c4a760a8-dbcf-4e92-9c8d-5d9b2a6c793e")
check_user_expired(user)
print(user)
