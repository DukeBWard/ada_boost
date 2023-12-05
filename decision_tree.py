import re
import math
import pickle


# Constants
MAX_STUMPS = 50


class node:
    """
    Represents a node in the decision tree.
    """

    def __init__(self, attributes, value, results, total_results, depth, prevprediction, is_true):
        """
        Initializes a node object.

        Parameters:
        - attributes (list): List of attributes.
        - value (str/int): The value of the node.
        - results (list): List of results.
        - total_results (list): List of indices of results.
        - depth (int): The depth of the node in the tree.
        - prevprediction (str): The previous prediction made.
        - is_true (bool): Indicates if the node represents a true or false branch.
        """
        self.attributes = attributes
        self.value = value
        self.results = results
        self.total_results = total_results
        self.depth = depth
        self.prevprediction = prevprediction
        self.is_true = is_true
        self.left = None
        self.right = None


def check_zero(values):
    """
    Checks if the maximum gain is 0.

    Parameters:
    - values (list): List of gain values.

    Returns:
    - bool: True if the maximum gain is 0, False otherwise.
    """
    if max(values) == 0:
        return True
    return False


def diff_values(values, total):
    """
    Checks if all the values in a list are the same.

    Parameters:
    - values (list): List of values.
    - total (list): List of indices.

    Returns:
    - bool: True if all the values are the same, False otherwise.
    """
    value = values[total[0]]
    for i in total:
        if value != values[i]:
            return False
    return True


def train_dt(root, attributes, seen, results, total_results, depth, prevprediction):
    """
    Trains the decision tree.

    Parameters:
    - root (node): The root node of the decision tree.
    - attributes (list): List of attributes.
    - seen (list): List of indices of attributes that have been seen.
    - results (list): List of results.
    - total_results (list): List of indices of results.
    - depth (int): The depth of the current node in the tree.
    - prevprediction (str): The previous prediction made.
    """
    if depth == len(attributes) - 1:
        num_eng, num_nl = 0, 0
        for index in total_results:
            if results[index] is 'en':
                num_eng += 1
            elif results[index] is 'nl':
                num_nl += 1
        if num_eng > num_nl:
            root.value = 'en'
        else:
            root.value = 'nl'

    elif len(total_results) == 0:
        root.value = prevprediction

    elif diff_values(results, total_results):
        root.value = results[total_results[0]]

    elif len(attributes) == len(seen):
        num_eng, num_nl = 0, 0
        for index in total_results:
            if results[index] is 'en':
                num_eng += 1
            elif results[index] is 'nl':
                num_nl += 1
        if num_eng > num_nl:
            root.value = 'en'
        else:
            root.value = 'nl'

    else:
        gain = []
        total_eng, total_nl = 0, 0
        for index in total_results:
            if results[index] == 'en':
                total_eng += 1
            else:
                total_nl += 1

        for attr_i in range(len(attributes)):
            if attr_i in seen:
                gain.append(0)
                continue

            else:
                eng_true, nl_true, eng_false, nl_false = 0, 0, 0, 0

                for index in total_results:
                    if attributes[attr_i][index] is True and results[index] == 'en':
                        eng_true += 1
                    elif attributes[attr_i][index] is True and results[index] == 'nl':
                        nl_true += 1
                    elif attributes[attr_i][index] is False and results[index] == 'en':
                        eng_false += 1
                    elif attributes[attr_i][index] is False and results[index] == 'nl':
                        nl_false += 1
                gain.append(calc_remain(eng_true, eng_false, nl_true, nl_false,
                                              total_eng, total_nl))

        if check_zero(gain):
            root.value = prevprediction
            return

        max_g = gain.index(max(gain))

        seen.append(max_g)

        true_i, false_i = [], []

        for index in total_results:
            if attributes[max_g][index] is True:
                true_i.append(index)
            else:
                false_i.append(index)

        cur_pred = ''

        if total_eng > total_nl:
            cur_pred = 'en'
        else:
            cur_pred = 'nl'

        root.value = max_g

        # Create left node for the current max gain
        left = node(attributes, None, results, true_i,
                        depth + 1, cur_pred, True)

        # Create right node for the current max gain
        right = node(attributes, None, results, false_i,
                         depth + 1, cur_pred, False)

        root.left = left
        root.right = right

        # Recurse on left and right nodes
        train_dt(left, attributes, seen, results,
                 true_i, depth + 1, cur_pred)
        train_dt(right, attributes, seen, results,
                 false_i, depth + 1, cur_pred)

        del seen[-1]


def train(data, hypo_file):
    """
    Trains the decision tree and saves it to a file.

    Parameters:
    - data (str): The path to the data file.
    - hypothesisFile (str): The path to the hypothesis file.
    """
    line_array, results = fill_arrays(data)
    attributes = calc_features(line_array)

    nums = []
    for i in range(len(results)):
        nums.append(i)

    seen = []
    root = node(attributes, None, results, nums, 0, None, None)
    train_dt(root, attributes, seen, results, nums, 0, None)

    hypo_file = open(hypo_file, 'wb')
    pickle.dump(root, hypo_file)


def predict(hypothesis_file, data):
    """
    Predicts the class labels for the given data using the trained decision tree.

    Parameters:
    - hypothesis_file (str): The path to the hypothesis file.
    - data (str): The path to the data file.
    """
    hypo_obj = pickle.load(open(hypothesis_file, 'rb'))
    data = open(data)
    sentence_array = []

    for line in data:
        sentence_array.append(line.strip())

    attributes = calc_features(sentence_array)

    for i in range(0, len(sentence_array)):
        temp = hypo_obj
        while type(temp.value) != str:
            value = attributes[temp.value][i]
            if value is True:
                temp = temp.left
            else:
                temp = temp.right
        print(temp.value)


def final_prediction(stump, sentence, attributes, index):
    """
    Makes the final prediction for a sentence using a stump.

    Parameters:
    - stump (node): The stump to make the prediction with.
    - sentence (str): The sentence to predict the class label for.
    - attributes (list): List of attributes.
    - index (int): The index of the sentence.

    Returns:
    - int: 1 if the class label is 'en', -1 if the class label is 'nl'.
    """
    attr_val = stump.value
    if attributes[attr_val][index] is True:
        if stump.left.value == 'en':
            return 1
        else:
            return -1
    else:
        if stump.right.value == 'en':
            return 1
        else:
            return -1


def prediction(stump, attributes, index):
    """
    Makes a prediction for a sentence using a stump.

    Parameters:
    - stump (node): The stump to make the prediction with.
    - attributes (list): List of attributes.
    - index (int): The index of the sentence.

    Returns:
    - str: The predicted class label.
    """
    attr_val = stump.value
    if attributes[attr_val][index]:
        return stump.left.value
    else:
        return stump.right.value


def return_stump(depth, root, attributes, results, total_results, weights):
    """
    Returns a stump from the decision tree.

    Parameters:
    - depth (int): The depth of the stump.
    - root (node): The root node of the decision tree.
    - attributes (list): List of attributes.
    - results (list): List of results.
    - total_results (list): List of indices of results.
    - weights (list): List of weights.

    Returns:
    - node: The stump.
    """
    gain = []
    total_eng, total_nl = 0, 0

    for index in total_results:
        if results[index] == 'en':
            total_eng += 1
        else:
            total_nl += 1

        if results[index] == 'en':
            total_eng = total_eng + 1 * weights[index]
        else:
            total_nl = total_nl + 1 * weights[index]

    for attr_i in range(len(attributes)):
        eng_true, nl_true, eng_false, nl_false = 0, 0, 0, 0

        for index in total_results:
            if attributes[attr_i][index] is True and results[index] == 'en':
                eng_true = eng_true + 1 * weights[index]
            elif attributes[attr_i][index] is True and results[index] == 'nl':
                nl_true = nl_true + 1 * weights[index]
            elif attributes[attr_i][index] is False and results[index] == 'en':
                eng_false = eng_false + 1 * weights[index]
            elif attributes[attr_i][index] is False and results[index] == 'nl':
                nl_false = nl_false + 1 * weights[index]

        gain.append(
            calc_remain(eng_true, eng_false, nl_true, nl_false, total_eng, total_nl))

    max_g = gain.index(max(gain))
    root.value = max_g
    max_eng_true, max_nl_true, max_eng_false, max_nl_false = 0, 0, 0, 0

    for index in range(len(attributes[max_g])):
        if attributes[max_g][index] is True:
            if results[index] == 'en':
                max_eng_true = max_eng_true + 1 * weights[index]
            else:
                max_nl_true = max_nl_true + 1 * weights[index]
        else:
            if results[index] == 'en':
                max_eng_false = max_eng_false + 1 * weights[index]
            else:
                max_nl_false = max_nl_false + 1 * weights[index]

    left = node(attributes, None, results, None, depth + 1,
                    None, None)
    right = node(attributes, None, results, None, depth + 1,
                     None, None)
    if max_eng_true > max_nl_true:
        left.value = 'en'
    else:
        left.value = 'nl'
    if max_eng_false > max_nl_false:
        right.value = 'en'
    else:
        right.value = 'nl'

    root.left = left
    root.right = right

    return root


def predict(hypo_file, data):
    hypo_obj = pickle.load(open(hypo_file, 'rb'))
    data = open(data)
    sentence_array = []

    for line in data:
        sentence_array.append(line.strip())

    attributes = calc_features(sentence_array)

    state_pointer = 0
    hypo_weight = hypo_obj[1]
    hypo_arrays = hypo_obj[0]

    for sentence in sentence_array:
        total = 0
        for index in range(len(hypo_obj[0])):
            total += final_prediction(hypo_arrays[index], sentence, attributes,
                                                state_pointer) * hypo_weight[index]
        if total > 0:
            print('en')
        else:
            print('nl')
        state_pointer += 1


def train(data, hypo_file):
    line_array, results = fill_arrays(data)
    attributes = calc_features(line_array)

    weights = [1 / len(line_array)] * len(line_array)

    num_arrays, stump_vals = [], []
    hypo_weights = [1] * MAX_STUMPS

    for i in range(len(results)):
        num_arrays.append(i)

    for hypothesis in range(0, MAX_STUMPS):
        root = node(attributes, None, results, num_arrays, 0, None, None)
        stump = return_stump(0, root, attributes, results,
                             num_arrays, weights)
        error, correct, incorrect = 0, 0, 0

        for index in range(0, len(line_array)):

            if prediction(stump, attributes, index) != results[index]:
                error += weights[index]
                incorrect += 1

        for index in range(len(line_array)):

            if prediction(stump, attributes, index) == results[index]:
                weights[index] = weights[index] * error / (1 - error)
                correct += 1
        total = 0
        for weight in weights:
            total += weight
        for index in range(len(weights)):
            weights[index] /= total

        hypo_weights[hypothesis] = math.log(((1 - error) / error), 2)
        stump_vals.append(stump)

    filehandler = open(hypo_file, 'wb')
    pickle.dump((stump_vals, hypo_weights), filehandler)


def avg_len(sentence):
    total_len = 0
    for word in sentence:
        total_len += len(word)
    avg_len = total_len / len(sentence)
    return avg_len > 5


def eng_func_words(sentence):
    eng_words = ["the"]
    for word in eng_words:
        if sentence.count(word) > 0:
            return True
    return False


def nl_func_words(sentence):
    nl_words = ["de", "het", "der", "des", "den"]
    for word in nl_words:
        if sentence.count(word) > 0:
            return True
    return False


def contains_q(sentence):
    if sentence.lower().find('q'):
        return False
    else:
        return True


def contains_x(sentence):
    if sentence.lower().find('x') < 0:
        return False
    else:
        return True


def check_eng(sentence):
    if sentence.lower().count("en") > 0:
        return True
    return False


def common_nl_words(sentence):
    commonNlList = ['naar', 'be', 'ik', 'het', 'voor', 'niet', 'met', 'hij', 'zijn', 'ze', 'wij', 'ze', 'er', 'hun',
                    'zo', 'over', 'hem', 'weten', 'jouw', 'dan', 'ook', 'onze', 'deze', 'ons', 'meest']
    for word in commonNlList:
        if sentence.count(word) > 0:
            return True
    return False


def common_en_words(sentence):
    commonEnList = ['to', 'be', 'I', 'it', 'for', 'not', 'with', 'he', 'his', 'they', 'we', 'she', 'there', 'their',
                    'so', 'about', 'me', 'him', 'know', 'your', 'than', 'then', 'also', 'our', 'these', 'us', 'most']

    for word in commonEnList:
        if sentence.count(word) > 0:
            return True
    return False


def check_for_van(sentence):
    if sentence.lower().count("van") > 0:
        return True
    return False


def check_for_een(sentence):
    if sentence.lower().count("een") > 0:
        return True
    return False


def check_for_and(sentence):
    if sentence.lower().count("and") > 0:
        return True
    return False


def calc_features(sentences):
    att1 = []
    att2 = []
    att3 = []
    att4 = []
    att5 = []
    att6 = []
    att7 = []
    att8 = []
    att9 = []
    att10 = []
    att11 = []

    for sentence in sentences:
        att1.append(avg_len(sentence))
        att2.append(eng_func_words(sentence))
        att3.append(nl_func_words(sentence))
        att4.append(contains_q(sentence))
        att5.append(contains_x(sentence))
        att6.append(check_eng(sentence))
        att7.append(common_nl_words(sentence))
        att8.append(common_en_words(sentence))
        att9.append(check_for_van(sentence))
        att10.append(check_for_een(sentence))
        att11.append(check_for_and(sentence))

    attributes = [att1, att2, att3, att4, att5,
                  att6, att7, att8, att9, att10, att11]

    return attributes


def fill_arrays(file_name):
    test_file = open(file_name, encoding="utf-8")

    lines = test_file.readlines()
    results, linesList = [], []
    for line in lines:
        line = line.lower()
        line = re.sub(r'\d', ' ', line)
        line = re.sub(r'[^\w\s]', ' ', line)
        line = line.strip()

        results.append(line[0:2])
        linesList.append(line[3::])

    return linesList, results


def entropy(value):
    if value == 1:
        return 0
    return (-1) * (value * math.log(value, 2.0) + (1 - value) * math.log((1 - value), 2.0))


def calc_remain(numberEnTrue, numberEnFalse, numberNlTrue, numberNlFalse, totalEn, totalNl):
    if (numberEnTrue + numberNlTrue == 0) or (numberEnFalse + numberNlFalse == 0):
        return 0

    if numberEnTrue == 0:
        remTrue = 0
        remFalse = ((numberEnFalse + numberNlFalse) / (totalNl + totalNl)) * entropy(
            numberEnFalse / (numberNlFalse + numberEnFalse))
    elif numberEnFalse == 0:
        remFalse = 0
        remTrue = ((numberEnTrue + numberNlTrue) / (totalNl + totalEn)) * entropy(
            numberEnTrue / (numberNlTrue + numberEnTrue))
    else:
        remTrue = ((numberEnTrue + numberNlTrue) / (totalNl + totalEn)) * entropy(
            numberEnTrue / (numberNlTrue + numberEnTrue))

        remFalse = ((numberEnFalse + numberNlFalse) / (totalNl + totalEn)) * entropy(
            numberEnFalse / (numberNlFalse + numberEnFalse))

    return entropy(totalEn / (totalEn + totalNl)) - (remTrue + remFalse)
