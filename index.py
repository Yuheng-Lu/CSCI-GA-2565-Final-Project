import nltk
import string
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk.stem.porter import PorterStemmer
import torch
from torch.nn.utils.rnn import pad_sequence
import streamlit as st
from openai import OpenAI
import os
from dotenv import load_dotenv

nltk.download("punkt")
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def clean_text(text):
    """Remove punctuation from text and lower its case."""
    translator = str.maketrans("", "", string.punctuation + "“”’…—")
    return text.lower().translate(translator)


def process_text(text):
    """Process text to remove stopwords and apply stemming."""
    words = text.split()
    processed_words = [stemmer.stem(word) for word in words if word not in stop_words]
    return " ".join(processed_words)


def preprocess_input(user_review, word_to_index, device="cpu"):
    sentences = sent_tokenize(user_review)

    numeric_sentences = []

    for sentence in sentences:
        cleaned_text = clean_text(sentence)
        processed_text = process_text(cleaned_text)

        sentence_numeric = [
            word_to_index.get(word, 0)
            for word in processed_text.split()
            if word in word_to_index
        ]

        if sentence_numeric:
            numeric_sentences.append(torch.tensor(sentence_numeric))

    if numeric_sentences:
        padded_sequences = pad_sequence(
            numeric_sentences, batch_first=True, padding_value=0
        )
        return padded_sequences.to(device)
    else:
        return torch.tensor([], dtype=torch.long).to(device)


def load_model():
    model = torch.load("./model/complete_model.pth", map_location=torch.device("cpu"))
    model.eval()
    return model


def call_openai_for_prediction(book_name, sentence, is_spoiler_prob, not_spoiler_prob):
    prompt = (
        f"Given the context of the book '{book_name}' and considering the model's prediction that the sentence has a {is_spoiler_prob:.2f} probability of being a spoiler "
        f"and a {not_spoiler_prob:.2f} probability of not being a spoiler, determine if the following sentence should be considered a spoiler. "
        f"Please provide a boolean response: True for spoiler, False for non-spoiler.\n\n"
        f"Sentence: '{sentence}'"
    )

    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are a book reviewer."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=10,
        temperature=0.1,
    )

    return response.choices[0].message.content


with open("./static/word_to_index.pkl", "rb") as f:
    word_to_index = pickle.load(f)


model = load_model()

st.title("Spoiler Detection in Book Reviews")
book_name = st.text_input("Book Name", "")
user_review = st.text_area("Enter your review", height=150)

if st.button("Analyze Review"):
    if user_review and book_name:
        sentences = sent_tokenize(user_review)

        result_placeholder = st.empty()

        outputs = []

        with torch.no_grad():
            for sentence in sentences:
                processed_input = preprocess_input(
                    sentence, word_to_index, device="cpu"
                )
                probability = torch.sigmoid(model.forward(processed_input).unsqueeze(0))
                outputs.append((sentence, probability))

        results = []
        for sentence, probability in outputs:
            not_spoiler = probability[0, 0, 0, 0]
            is_spoiler = probability[0, 0, 0, 1]
            spoiler_prediction = call_openai_for_prediction(
                book_name, sentence, is_spoiler, not_spoiler
            )
            results.append([sentence, not_spoiler, is_spoiler, spoiler_prediction])

        result_text = ""
        for sentence, not_spoiler, is_spoiler, spoiler_prediction in results:
            result_text += f"**Sentence**: {sentence}\n\n"
            result_text += "**Prediction from the model only**:\n"
            result_text += f"*Not Spoiler*: {not_spoiler:.2f},\n"
            result_text += f"*Spoiler*: {is_spoiler:.2f}\n\n"
            result_text += (
                f"**Prediction integrating LLM API**: {spoiler_prediction} \n\n "
            )
            result_text += "---\n\n"

        result_placeholder.write(result_text)
    else:
        st.write("Please enter a review to analyze.")

# example book: The Three-Body Problem
# example review: In "The Three-Body Problem" by Liu Cixin, the revelation of an alien civilization planning to invade Earth completely caught me off guard. The use of sophisticated science and technology, like the unfolding of a proton into a two-dimensional surface for communication, is both ingenious and terrifying. Ye Wenjie's backstory and her eventual betrayal of humanity offer a poignant commentary on the extremes of despair and disillusionment. The climax, where Earth's scientists are covertly manipulated by the Trisolarans through the virtual reality game, raises profound questions about free will and destiny. This novel is a masterful blend of science fiction and philosophical inquiry, leaving readers eagerly anticipating the consequences of the Trisolaran invasion.
