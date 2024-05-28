from csv import DictReader
import numpy as np

vocab = {}
#Format training data
def process_data(filename):
    vocab_size = 0
    examples = []

    with open(filename, 'r') as f:
        reader = DictReader(f)
        for row in reader:
            label = row['rating'] == '1'
            words = row['text'].split(' ')
            for word in words:
                if word not in vocab:
                    vocab[word] = vocab_size
                    vocab_size += 1
            examples.append((label, [vocab[word] for word in words]))

    return examples, vocab_size


ret = process_data('reviews_tr.csv')
training = ret[0]
dim = ret[1]

#Online Perceptron
def learn(data):
    w = np.zeros(dim)
    for ex in data:
        label = ex[0]
        word_ids = ex[1]
        weighted_sum = 0
        for word in word_ids:
            weighted_sum += w[word]
        if weighted_sum > 0 and label == 0:
            w[word_ids] -= 1
        elif weighted_sum <= 0 and label == 1:
            w[word_ids] += 1
    return w

w = learn(training)

# Run on training data
training_errors = 0
for row in training:
    pred = 1 if w[row[1]].sum() > 0 else 0
    if row[0] != pred:
        training_errors += 1
training_error_rate = training_errors / len(training)

print(f"Training Error Rate: {training_error_rate}")

# Run on test data
ret2 = process_data('reviews_te.csv')
test = ret2[0]
predictions = []

with open('reviews_te.csv', 'r') as f:
    reader = DictReader(f)
    for row in reader:
        label = int(row['rating']) == 1
        words = row['text'].split(' ')
        word_ids = [vocab[word] for word in words if word in vocab]
        prediction = 1 if w[word_ids].sum() > 0 else 0
        predictions.append((label, prediction))


num_errors = sum(label != prediction for label, prediction in predictions)
test_error_rate = num_errors / len(predictions)

print(f"Test Error Rate: {test_error_rate}")

reverse_vocab = {v: k for k, v in vocab.items()}
# Find most positive words
most_pos_weights = np.argsort(-w)[:10]
most_pos_words = []
for weight in most_pos_weights:
    most_pos_words.append((next(k for k,v in vocab.items() if v==weight)))

print(most_pos_words)
# Find most positive words
most_neg_weights = np.argsort(w)[:10]
most_neg_words = []
for weight in most_neg_weights:
    most_neg_words.append((next(k for k,v in vocab.items() if v==weight)))

print(most_neg_words)

#average perceptron variant
def learn_avg(data):
    w = np.zeros(dim)
    w_sum = w
    for ex in data:
        label = ex[0]
        word_ids = ex[1]
        weighted_sum = 0
        for word in word_ids:
            weighted_sum += w[word]
        if weighted_sum > 0 and label == 0:
            w[word_ids] -= 1
        elif weighted_sum <= 0 and label == 1:
            w[word_ids] += 1
        w_sum = w_sum + w
    return w_sum

w = learn_avg(training)