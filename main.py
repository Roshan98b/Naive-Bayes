import sys
from naive_bayes import get_examples_and_class, learn_naive_bayes_text, classify_naive_bayes_text

train_path = sys.argv[1]
test_path = sys.argv[2]

examples, V = get_examples_and_class(train_path)

print ('Class Labels')
print (V)
print ('')
print ('Training Data')
for filename, label in examples:
    print (filename+' : '+label)
print ('')
trained_params = learn_naive_bayes_text(examples, V, train_path)

class_label = classify_naive_bayes_text(test_path, trained_params, V) 
print(test_path+' : '+class_label)