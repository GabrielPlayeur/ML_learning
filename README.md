# ML learning

I use this repo to learn machine learning and deep learning from the ground up. It holds the resources I followed (courses, labs, notebooks) and two personal projects where I apply what I learned.

Both projects are run as small studies: a hypothesis, controlled experiments, quantified results and a discussion of the limitations. The code is there to answer the questions, not an end in itself.

## Projects

### [MusicGenreClassifier](MusicGenreClassifier/): music genre classification (GTZAN)

Question: for a fixed input representation (MFCCs), which architecture makes the best use of the time-frequency structure of a 3 s clip? I compare a logistic regression, an MLP, a CNN and an LSTM on the same split with 3 seeds, then run ablations (deltas, augmentation, LSTM pooling, track-level voting).

PyTorch · Lightning · Hydra · Weights & Biases · librosa

### [snakeRL](snakeRL/): a DQN agent that learns to play Snake

Question: what actually makes a Snake agent better? The architecture (MLP or CNN), the information it gets as input, or the algorithm? I wrote a Double DQN (target network, dueling, n-step, Huber loss) and evaluate it against hand-written heuristics that receive the same information.

PyTorch · NumPy · Matplotlib

## Resources and exercises

| Folder | Content |
|---|---|
| [introtodeeplearning-master](introtodeeplearning-master/) | Labs from the [MIT 6.S191 Introduction to Deep Learning](http://introtodeeplearning.com) course, in PyTorch. Lab 1: PyTorch basics and music generation with an RNN. Lab 2: MNIST classification (CNN) and debiasing a face detector (VAE). |
| [AudioSignalProcessing](AudioSignalProcessing/) | Audio signal processing notebooks, following Valerio Velardo's *Audio Signal Processing for ML* series: zero-crossing rate, RMS, magnitude spectrum, spectrograms, mel-spectrograms, MFCCs, and an autoencoder. |
| [AudioDeepLearning](AudioDeepLearning/) | First version of the genre classifier (MLP, then CNN, then LSTM), written in plain PyTorch before the Lightning and Hydra rewrite in MusicGenreClassifier. |

## Progression

1. **Basics**: PyTorch and first networks (MIT 6.S191, labs 1 and 2).
2. **Audio signal**: understanding what goes into the network (time, frequency, MFCCs).
3. **First supervised project**: music genres, MLP → CNN → LSTM ([AudioDeepLearning](AudioDeepLearning/)).
4. **Tooling and experimental rigor**: the same project rebuilt with Lightning, Hydra and W&B, and a controlled protocol ([MusicGenreClassifier](MusicGenreClassifier/)).
5. **Reinforcement learning**: DQN written from scratch, diagnosis of failures and evaluation against baselines ([snakeRL](snakeRL/)).

Each project has its own [uv](https://docs.astral.sh/uv/) environment (run `uv sync` in its folder).
