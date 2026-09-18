# Hangman Solver

BiLSTM + statistical priors. 6 wrong tries max. 56.2% win rate on 5,000 held-out words.

**Live demo:** [https://hangman-solver-hm4ge6wiw9hhtfmwmxvsmb.streamlit.app/](https://hangman-solver-hm4ge6wiw9hhtfmwmxvsmb.streamlit.app/)

## How it works

Each guess mixes three signals:

1. BiLSTM letter probabilities (`model_short` for words ≤ 16, `model_all` for longer)
2. Unigram / bigram / candidate-set priors
3. Information gain over remaining dictionary words

```
score = α·log(p_bilstm) + (1−α)·(β·statistical + γ·info_gain)
```

Words of 3 letters or fewer just guess vowels first.

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS / Linux

pip install -r requirements.txt
streamlit run streamlit_app.py
```

Then open http://localhost:8501

## Deploy on Streamlit Community Cloud

The app is already live at [https://hangman-solver-hm4ge6wiw9hhtfmwmxvsmb.streamlit.app/](https://hangman-solver-hm4ge6wiw9hhtfmwmxvsmb.streamlit.app/).

To deploy your own copy:

1. Push this repo to GitHub (public).
2. Open [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Choose this repository, branch `main`, and main file `streamlit_app.py`.
4. Deploy. The first boot loads the BiLSTM weights and dictionary, then the app is ready.

Python 3.11 is pinned in `.python-version` so Cloud matches local.

## Layout

```
streamlit_app.py
hangman/
  game.py
  solver.py
  bilstm_numpy.py
  ui.py
models/
  model_all.weights.h5
  model_short.weights.h5
data/
  train_words.txt
  test_words.txt
  words_250000_train.txt
notebooks/
  01_dataset_split.ipynb
  02_train_bilstm.ipynb
  03_evaluate.ipynb
```
