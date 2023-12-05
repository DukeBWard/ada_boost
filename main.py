import sys
import decision_tree

def predict(learning_type):
    hypothesis = sys.argv[2]
    file = sys.argv[3]

    if learning_type == "dt":
        decision_tree.predict(hypothesis, file)
    else:
        decision_tree.predict(hypothesis, file)


def train(learning_type):
    examples = sys.argv[2]
    hypothesisOut = sys.argv[3]

    if learning_type == "dt":
        decision_tree.train(examples, hypothesisOut)
    else:
        decision_tree.train(examples, hypothesisOut)


if __name__ == '__main__':
    mode = sys.argv[1]

    if mode == "train":
        train(sys.argv[4])

    elif mode == "predict":
        predict(sys.argv[3])
    else:
        sys.exit()
