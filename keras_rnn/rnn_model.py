from data_util import ptb_raw_data
from sys import argv
from time import time

from keras.callbacks import TensorBoard
from keras.layers import *
from keras.models import Sequential, Model
from keras.utils import to_categorical


batch_size=32
num_steps=10


class KerasBatchGenearator(object):
    def __init__(self, data, vocabulary, skip_step=5):
        self.data = data
        self.vocabulary = vocabulary
        self.current_idx = 0
        self.skip_step = skip_step

    def generate(self):
        x = np.zeros((batch_size, num_steps))
        y = np.zeros((batch_size, self.vocabulary))
        while True:
            for i in range(batch_size):
                if self.current_idx + num_steps >= len(self.data):
                    self.current_idx = 0
                x[i, :] = self.data[self.current_idx:self.current_idx + num_steps]
                y[i, :] = to_categorical(self.data[self.current_idx+num_steps+1],
                                         num_classes=self.vocabulary)
                self.current_idx += self.skip_step
            yield x, y


def LSTMModel(vocab_size, embed_size=128, hidden_size=128):
    model = Sequential()
    model.add(Embedding(vocab_size, embed_size, input_length=num_steps))
    model.add(LSTM(hidden_size, return_sequences=True))
    model.add(LSTM(hidden_size, return_sequences=False))
    model.add(Dense(vocab_size))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['categorical_accuracy'])
    return model


def RNNModel(vocab_size, embed_size=128, hidden_size=128):
    model = Sequential()
    model.add(Embedding(vocab_size, embed_size, input_length=num_steps))
    model.add(SimpleRNN(hidden_size, return_sequences=True))
    model.add(SimpleRNN(hidden_size, return_sequences=False))
    model.add(Dense(vocab_size))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['categorical_accuracy'])
    return model


# https://github.com/philipperemy/keras-attention-mechanism/blob/master/attention_lstm.py
def attention_3d_block(input):
    a = Permute((2, 1))(input)
    a = Dense(num_steps, activation='softmax')(a)
    a = Permute((2, 1))(a)
    output = merge([input, a], name='attention_mul', mode='mul')
    return output


def model_attention_applied_before_lstm(vocab_size, embed_size=128, hidden_size=128):
    input = Input(shape=(num_steps,))
    embed = Embedding(vocab_size, embed_size)(input)
    attention_layer = attention_3d_block(embed)
    lstm = LSTM(hidden_size, return_sequences=False)(attention_layer)
    dense = Dense(vocab_size, activation='softmax')(lstm)
    model = Model(input=[input], output=dense)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['categorical_accuracy'])
    return model


if __name__ == '__main__':
    train, valid, test, vocab_size = ptb_raw_data()
    train_gen = KerasBatchGenearator(train, vocab_size)
    valid_gen = KerasBatchGenearator(valid, vocab_size)
    tensorboard = TensorBoard(log_dir="logs/{}".format(int(time())))

    if argv[1] == 'lstm':
        m = LSTMModel(vocab_size=vocab_size)
    elif argv[1] == 'rnn':
        m = RNNModel(vocab_size=vocab_size)
    elif argv[1] == 'att':
        m = model_attention_applied_before_lstm(vocab_size=vocab_size)

    m.summary()
    m.fit_generator(train_gen.generate(),
                    steps_per_epoch=len(train) // (batch_size * num_steps),
                    epochs=50,
                    validation_data=valid_gen.generate(),
                    validation_steps=len(valid) // (batch_size * num_steps),
                    callbacks=[tensorboard])