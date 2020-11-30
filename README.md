# Reinforcement Learning Project - The NetHack Learning Environment

This project is based on making a model capable of playing the [NLE provided by Facebook](https://github.com/facebookresearch/nle).

This repo specifically contains the code for our DQN implementation. If you're looking for our Actor-Critic implementation, you can find it [here](https://github.com/dylanHanger/rl-project).

## Breakdown

The root contains the `train.py` and `test.py` files, which are used to train a specific model, and test that the NLE installation actually functions, respectively.

The `shared` folder contains some code that is generally useful for any methods, as the repo was originally meant to have several models.

The `dqn_david` folder contains all the code relevant to the DQN model specifically.

## Running the code

To train a model yourself, just run the `train.py` script in the root directory, with the name of the folder the model is in. (This was coded how it is because originally all the models would be in this repo, but that didn't happen). To train the (only) DQN, you would run `python train.py dqn_david`. I would strongly recommend against that as it'll take long and probably not get you anything nice.
