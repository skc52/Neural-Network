import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# Sample data (for a real use case, you'd use a larger dataset)
english_sentences = [
    "I am happy",
    "I am sad",
    "I love programming",
    # Add more sentences
]

english_sentences.extend([
    "The weather is nice today",
    "I enjoy learning new languages",
    "Can you help me with this problem?",
    "I would like a cup of coffee",
    "She is reading a book",
    "We are going to the beach this weekend",
    "They have a beautiful garden",
    "What time is the meeting?",
    "I love watching movies",
    "He is a great chef"
])


french_sentences = [
    "Je suis heureux",
    "Je suis triste",
    "J'adore la programmation",
    # Add more translations
]
french_sentences.extend([
    "Il fait beau aujourd'hui",
    "J'aime apprendre de nouvelles langues",
    "Peux-tu m'aider avec ce problème ?",
    "Je voudrais une tasse de café",
    "Elle lit un livre",
    "Nous allons à la plage ce week-end",
    "Ils ont un beau jardin",
    "À quelle heure est la réunion ?",
    "J'adore regarder des films",
    "Il est un excellent chef"
])


# Tokenizer for English sentences
eng_tokenizer = Tokenizer()
eng_tokenizer.fit_on_texts(english_sentences)
eng_vocab_size = len(eng_tokenizer.word_index) + 1

# Tokenizer for French sentences
fr_tokenizer = Tokenizer()
fr_tokenizer.fit_on_texts(french_sentences)
fr_vocab_size = len(fr_tokenizer.word_index) + 1

# Convert sentences to sequences of integers
eng_sequences = eng_tokenizer.texts_to_sequences(english_sentences)
fr_sequences = fr_tokenizer.texts_to_sequences(french_sentences)

# Padding sequences
max_eng_len = max(len(seq) for seq in eng_sequences)
max_fr_len = max(len(seq) for seq in fr_sequences)

eng_sequences_padded = pad_sequences(eng_sequences, maxlen=max_eng_len, padding='post')
fr_sequences_padded = pad_sequences(fr_sequences, maxlen=max_fr_len, padding='post')

# One-hot encode French sequences for categorical crossentropy loss
fr_sequences_one_hot = to_categorical(fr_sequences_padded, num_classes=fr_vocab_size)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense

# Parameters
embedding_dim = 256
lstm_units = 512

# Encoder
encoder_input = Input(shape=(max_eng_len,))
encoder_embedding = Embedding(eng_vocab_size, embedding_dim)(encoder_input)
encoder_lstm = LSTM(lstm_units, return_sequences=True, return_state=True)
encoder_outputs, encoder_state_h, encoder_state_c = encoder_lstm(encoder_embedding)
encoder_states = [encoder_state_h, encoder_state_c]

# Decoder
decoder_input = Input(shape=(max_fr_len,))
decoder_embedding = Embedding(fr_vocab_size, embedding_dim)(decoder_input)
decoder_lstm = LSTM(lstm_units, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
decoder_dense = Dense(fr_vocab_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model
model = Model([encoder_input, decoder_input], decoder_outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

# Training the model
history = model.fit(
    [eng_sequences_padded, fr_sequences_padded[:, :-1]],  # Use all but last token for decoder input
    fr_sequences_one_hot[:, 1:],  # Use all but first token for target
    epochs=10,
    batch_size=64,
    validation_split=0.2
)

# Define inference models
encoder_model = Model(encoder_input, encoder_states)

decoder_input_infer = Input(shape=(1,))
decoder_embedding_infer = Embedding(fr_vocab_size, embedding_dim)
decoder_lstm_infer = LSTM(lstm_units, return_sequences=True, return_state=True)
decoder_output_infer, state_h, state_c = decoder_lstm_infer(
    decoder_embedding_infer(decoder_input_infer), initial_state=encoder_states
)
decoder_states = [state_h, state_c]
decoder_output_infer = decoder_dense(decoder_output_infer)

decoder_model = Model(
    [decoder_input_infer] + encoder_states,
    [decoder_output_infer] + decoder_states
)

def decode_sequence(input_seq):
    states_value = encoder_model.predict(input_seq)
    target_seq = np.array([[fr_tokenizer.word_index['<start>']]])  # Use <start> token for initial input
    stop_condition = False
    decoded_sentence = ''
    
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = fr_tokenizer.index_word.get(sampled_token_index, '<unknown>')
        
        if sampled_word == '<end>' or len(decoded_sentence.split()) > max_fr_len:
            stop_condition = True
        else:
            decoded_sentence += ' ' + sampled_word
            
        target_seq = np.array([[sampled_token_index]])
        states_value = [h, c]
    
    return decoded_sentence.strip()

# Test the model
test_sentence = "I love programming"
test_seq = eng_tokenizer.texts_to_sequences([test_sentence])
test_seq_padded = pad_sequences(test_seq, maxlen=max_eng_len, padding='post')
print(decode_sequence(test_seq_padded))
