import cv2
import tensorflow as tf
import numpy as np
import style_transfer_trainer
import utils
import vgg19
from run_train import add_one_dim, \
    CONTENT_LAYERS_NAME, STYLE_LAYERS_NAME, \
    CONTENT_LAYER_WEIGHTS, STYLE_LAYER_WEIGHTS

VIDEO_FILE = "/Users/dwang/Downloads/IMG_2693.mp4"
VIDEO_OUT_FILE = "/Users/dwang/Downloads/IMG_2693.avi"
MODEL_FILE = "/Users/dwang/transfer"
STYLE_FILE = "style/wave.jpg"
VGG_FILE = "/Users/dwang/transfer/imagenet-vgg-verydeep-19.mat"

if __name__ == '__main__':
    style_image = utils.load_image(STYLE_FILE)
    vgg_net = vgg19.VGG19(VGG_FILE)
    CONTENT_LAYERS = {}
    for layer, weight in zip(CONTENT_LAYERS_NAME, CONTENT_LAYER_WEIGHTS):
        CONTENT_LAYERS[layer] = weight
    STYLE_LAYERS = {}
    for layer, weight in zip(STYLE_LAYERS_NAME, STYLE_LAYER_WEIGHTS):
        STYLE_LAYERS[layer] = weight

    cap = cv2.VideoCapture(VIDEO_FILE)
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    video_out = None

    with tf.Session() as sess:
        trainer = style_transfer_trainer.StyleTransferTrainer(
            session=sess,
            content_layer_ids=CONTENT_LAYERS,
            style_layer_ids=STYLE_LAYERS,
            content_images=[],
            style_image=add_one_dim(style_image),
            net=vgg_net,
            num_epochs=0,
            batch_size=1,
            content_weight=0,
            style_weight=0,
            tv_weight=0,
            learn_rate=0,
            save_path=MODEL_FILE,
            check_period=0,
            max_size=None)
        trainer.prepare()

        while cap.isOpened():
            ret, frame = cap.read()
            if not video_out:
                video_out = cv2.VideoWriter(VIDEO_OUT_FILE, fourcc, 24, (frame.shape[1], frame.shape[0]))

            if ret:
                for i in range(frame.shape[0]):
                    for j in range(frame.shape[1]):
                        frame[i, j, 0], frame[i, j, 2] = frame[i, j, 2], frame[i, j, 0]
                output = trainer.test(add_one_dim(frame))
                output = output[0].astype(np.uint8)
                for i in range(frame.shape[0]):
                    for j in range(frame.shape[1]):
                        output[i, j, 0], output[i, j, 2] = output[i, j, 2], output[i, j, 0]
                        output[i, j, 1] = output[i, j, 1]
                print frame.shape, output.shape
                cv2.imshow('frame', output)
                video_out.write(output)
            else:
                break

            if cv2.waitKey(1) and 0xFF == ord('q'):
                break

    cap.release()
    video_out.release()
    cv2.destroyAllWindows()
