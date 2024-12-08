import gzip
import os
import shutil

import gdown
import wget


def download():
    gdown.download(id="1i2Na7jRo9c95-X925l-JyQ2I0bFoCwTI")
    os.rename("model_best.pth", "saved/model_best.pth")


if __name__ == "__main__":
    download()
