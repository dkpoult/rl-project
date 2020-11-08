import argparse

parser = argparse.ArgumentParser()
parser.add_argument("agent", type = str, metavar = "a")

arguments = parser.parse_args()

train_script = getattr(__import__(arguments.agent, fromlist = ["train"]), "train")

train_script.train()