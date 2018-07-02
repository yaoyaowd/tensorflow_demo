from config import *
from keras.models import Model
from keras.layers import Input, Conv2D, Flatten, Dense, Dropout, ELU, Lambda
from load_data import generate_data_batch, split_train_val


def get_model():
    input_frame = Input(shape=(HEIGHT, WIDTH, CHANNELS))
    x = Lambda(lambda z: z/127.5-1.)(input_frame)

    x = Conv2D(24, (5, 5), strides=(2, 2))(x)
    x = ELU()(x)
    x = Dropout(0.2)(x)
    x = Conv2D(36, (5, 5), strides=(2, 2))(x)
    x = ELU()(x)
    x = Dropout(0.2)(x)
    x = Conv2D(48, (5, 5), strides=(2, 2))(x)
    x = ELU()(x)
    x = Dropout(0.2)(x)
    x = Conv2D(64, (3, 3))(x)
    x = ELU()(x)
    x = Dropout(0.2)(x)
    x = Conv2D(64, (3, 3))(x)
    x = ELU()(x)
    x = Dropout(0.2)(x)
    x = Flatten()(x)
    x = Dense(100)(x)
    x = ELU()(x)
    x = Dense(10)(x)
    x = ELU()(x)
    out = Dense(1)(x)
    model = Model(inputs=input_frame, outputs=out)
    model.compile(optimizer='adam', loss='mse')
    model.summary()
    return model


if __name__ == '__main__':
    train, test = split_train_val('/Users/dwang/self-driving-car/project_3_behavioral_cloning/data/driving_log.csv')
    model = get_model()
    model.fit_generator(generator=generate_data_batch(train, augment_data=True, bias=BIAS),
                        steps_per_epoch=BATCH_SIZE,
                        epochs=50,
                        validation_data=generate_data_batch(test, augment_data=False, bias=1.0),
                        validation_steps=BATCH_SIZE*100)
