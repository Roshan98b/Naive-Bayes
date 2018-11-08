import os
import re
import json

finite_automata = {
    0: {
        '[a-zA-Z0-9\'.@\-:]': 1,
        '.*?': 'move'
    },
    1: {
        '[a-zA-Z0-9\'.@\-:]': 1,
        '.*?': 'retract'
    }
}

def get_examples_and_class(path):
    "Finds and creates training data. Takes the path of the training data as the parameter. The path should contain directories with name of the class labels. Each class diectory should cntain files for training."
    examples = []
    V = []
    for directories in os.walk(path):
        for directory in directories[1]:
            V.append(directory)
            for filenames in os.walk(os.path.join(path, directory)):
                for filename in filenames[2]:
                    examples.append((filename, directory))
    return (examples,V)

def get_all_tokens(examples, dataset_path):
    "Creates 'vocabulary' with word count wrt to each class 'v'. Uses a simple finite automata for finding tokens"
    vocabulary = {}
    count = {}
    current_state = 0
    lexeme = ''
    for dirs in os.walk(dataset_path):
        for directory in dirs[1]:
            path = os.path.join(dataset_path, directory)
            count[directory] = 0
            for files in os.walk(path):
                for filename in files[2]:
                    with open(os.path.join(path, filename), 'rb') as f:
                        while True:
                            char = (f.read(1)).decode('ASCII')
                            temp = finite_automata[current_state]
                            lexeme += char
                            for key in temp:
                                if re.match(key, char):
                                    next = temp[key]
                                    if next == 'retract':
                                        lexeme = lexeme[:-1]
                                        if lexeme:
                                            if re.match(r'[.,:]', lexeme[-1]):
                                                lexeme = lexeme[:-1]
                                        # add to vocabulary
                                        n = 0
                                        if lexeme in vocabulary:
                                            n = vocabulary[lexeme][directory]['count']
                                        else:
                                            vocabulary[lexeme] = {}
                                            for d in dirs[1]:
                                                vocabulary[lexeme][d] = {}
                                                vocabulary[lexeme][d]['count'] = 0
                                                vocabulary[lexeme][d]['P'] = 0
                                        vocabulary[lexeme][directory]['count'] = n+1
                                        count[directory] += 1
                                        lexeme = '' 
                                        next = 0
                                        f.seek(-1, 1)
                                    elif next == 'move':
                                        lexeme = ''
                                        next = 0
                                        continue
                                    current_state = next
                                    break
                            if not char:
                                break       
    return (vocabulary, count)

def get_v_examples(examples, v):
    "Returns subset of examples belonging to class 'v'"
    docs = [i for i in examples if i[1] == v]
    return docs

# Training using Naive Bayes Algorithm 
def learn_naive_bayes_text(examples, V, dataset_path):
    "'examples' contain set of documents names with class labels. 'V' contains set of class labels"
    P = {}
    vocabulary, count = get_all_tokens(examples, dataset_path)
    for v in V:
        docs = get_v_examples(examples, v)
        P[v] = len(docs)/len(examples)
        n = count[v]
        for w in vocabulary:
            nk =  vocabulary[w][v]['count']
            vocabulary[w][v]['P'] = (nk+1)/(n+len(vocabulary))
    return {'class_prob': P, 'vocabulary': vocabulary}

def get_test_vocabulary(path, main_vocabulary):
    "Create Vocabulary for test data"
    current_state = 0
    lexeme = ''
    test_vocabulary = {}
    with open(path, 'rb') as f:
        while True:
            char = (f.read(1)).decode('ASCII')
            temp = finite_automata[current_state]
            lexeme += char
            for key in temp:
                if re.match(key, char):
                    next = temp[key]
                    if next == 'retract':
                        lexeme = lexeme[:-1]
                        if lexeme:
                            if re.match(r'[.,:]', lexeme[-1]):
                                lexeme = lexeme[:-1]
                        # check if 'lexeme' is present in 'main_vocabulary'
                        if lexeme in main_vocabulary:
                            # add 'lexeme' to test_vocabulary
                            test_vocabulary[lexeme] = main_vocabulary[lexeme] 
                        lexeme = ''
                        next = 0
                        f.seek(-1, 1)
                    elif next == 'move':
                        lexeme = ''
                        next = 0
                        continue
                    current_state = next
                    break
            if not char:
                break
    return test_vocabulary

# Classify using Naive Bayes Algorithm
def classify_naive_bayes_text(path, trained_params, V):
    "Test documents path, class labels and object returned from training are taken as parameters. returns a string with class label."
    class_label = ''
    Vnb = {}
    vocabulary = get_test_vocabulary(path, trained_params['vocabulary'])
    if not vocabulary:
        return 'Cannot classify. Vocabulary is empty.'
    for v in V:
        value = trained_params['class_prob'][v]
        for w in vocabulary:
            value *= vocabulary[w][v]['P']
        Vnb[v] = value
    flag = 1
    for i in Vnb:
        if Vnb[i]:
            flag = 0
            break
    if flag:
        return 'Cannot classify. Probability is too low.'
    max = 0
    for i in Vnb:
        if max < Vnb[i]:
            max = Vnb[i]
            class_label = i 
    return class_label 