import argparse
import os
from image_model import ImageModel

parser = argparse.ArgumentParser(description="Train image classification model")
parser.add_argument("--image_path", type=str, default="",
                    help="The folder for downloaded images")
parser.add_argument("--output_path", type=str, default="",
                    help="The folder for output path")
parser.add_argument("--num_epochs", type=int, default=1,
                    help="The number of epochs for training")


def maybe_make_dir(folder):
    if not os.path.exists(folder):
        os.mkdir(folder)


def main():
    args = parser.parse_args()
    log_dir = os.path.join(args.output_path, "log")
    checkpoint_dir = os.path.join(args.output_path, "model")
    maybe_make_dir(log_dir)
    maybe_make_dir(checkpoint_dir)

    model = ImageModel()
    model.train(
        image_path=args.image_path,
        num_epochs=args.num_epochs,
        log_dir=log_dir,
        checkpoint_dir=checkpoint_dir)

if __name__ == "__main__":
    main()