# Signal vocoding (TTS) with PyTorch

<p align="center">
  <a href="#about">About</a> •
  <a href="#installation">Installation</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#credits">Credits</a> •
  <a href="#license">License</a>
</p>

## About

This repository contains an vocoder pipeline using PyTorch and HiFiGAN as main model.

## Installation

Follow these steps to install the project:

0. (Optional) Create and activate new environment using [`conda`](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) or `venv` ([`+pyenv`](https://github.com/pyenv/pyenv)).

   a. `conda` version:

   ```bash
   # create env
   conda create -n project_env python=PYTHON_VERSION

   # activate env
   conda activate project_env
   ```

   b. `venv` (`+pyenv`) version:

   ```bash
   # create env
   ~/.pyenv/versions/PYTHON_VERSION/bin/python3 -m venv project_env

   # alternatively, using default python version
   python3 -m venv project_env

   # activate env
   source project_env
   ```

1. Install all required packages

   ```bash
   pip install -r requirements.txt
   ```

2. Install `pre-commit`:
   ```bash
   pre-commit install
   ```

3. Download the final model's weights by running the script:

   ```bash
   python download_weights.py
   ```

## Training

The model was trained in 2 steps. To reproduce, train model using the following commands:

1. Train 40 epochs without augmentations:

   ```bash
   python train.py -cn=hifigan_train_config
   ```

2. Train 40 epochs with bigger gradient norm 

   ```bash
   python train.py -cn=hifigan
   ```

### Inference

   1. If you want only to decode text to speech, your directory with text should has the following format:
      ```bash
      NameOfTheDirectoryWithUtterances
      ├── UtteranceID1.txt
      ├── UtteranceID2.txt
      .
      .
      .
      └── UtteranceIDn.txt
      ```

      Run the following command:
      ```bash
      python synthesize.py -cn=hifigan_synthesize_config ++datasets.test.transcription_dir=TRANSCRIPTION_DIR ++inferencer.save_path=SAVE_PATH
      ```
      where `SAVE_PATH` is a path to save predicted audio and `TRANSCRIPTION_DIR` is directory with texts.
   2. If you have ground truth audio and want to generate vocoded audios, make sure that directory with audio has the following format:
      ```
      NameOfTheDirectoryWithUtterances
      ├── UtteranceID1.wav # may be flac or mp3
      ├── UtteranceID2.wav
      .
      .
      .
      └── UtteranceIDn.wav
      ```
      Then run the following command:
      ```bash
      python synthesize.py -cn=hifigan_inference_config ++datasets.test.audio_dir=AUDIO_DIR ++inferencer.save_path=SAVE_PATH
      ```
      where `SAVE_PATH` is a path to save predicted audio and `AUDIO_DIR` is directory with GT audio.

   3. If you want to use the TTS pipeline for a specific query, then run:
      ```bash
      python synthesize.py -cn=hifigan_synthesize_config '++datasets.test.query="QUERY"' ++inferencer.save_path=SAVE_PATH
      ```
      where `SAVE_PATH` is a path to save predicted audio and `QUERY` is text requested to be transformed to speech.

## Credits

This repository is based on a [PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template).

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)