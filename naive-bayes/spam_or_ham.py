# built-in python library for interacting with .csv files
import csv
# natural language toolkit is a great python library for natural language processing
import nltk
# built-in python library for utility functions that introduce randomness
import random
# built-in python library for measuring time-related things
import time


def get_length_bucket(sms_length):
    """
    buckets the sms length into either short / medium / long
    """
    if sms_length < 30:
        return "short"
    elif sms_length < 90:
        return "medium"
    else:
        return "long"


def sms_features(sms):
    """
     Returns a dictionary of the features of the sms we want our model
    to be based on, e.g. sms_length.

    So if the sms was "Hey!", the output of this function would be
    {
        "length": "short"
    }

    If the sms was "Hey this is a really great idea and I think that we should totally implement this technique",
    then the output would be
    {
        "length": "medium"
    }
    """
    return {
        "length": get_length_bucket(len(sms)),
        "contains_win": "win" in sms,
        "contains_invite": "invite" in sms,
        "contains_offer": "offer" in sms,
        "contains_discount": "discount" in sms,
        "contains_free": "free" in sms,
        "contains_prize": "prize" in sms,
        "contains_urgent": "urgent" in sms,
        "contains_exclamation": "!" in sms,
        "contains_question": "?" in sms,
        "contains_emojismile": ":-)" in sms,
        "contains_sorry": "sorry" in sms,
        "contains_love": "love" in sms,
        "contains_guaranteed": "guaranteed" in sms,
        "contains_loveandyou": "love" and "you" in sms
    }


def get_feature_sets():
    with open('sms_spam_or_ham.csv') as csvfile:
        reader = csv.DictReader(csvfile, fieldnames=('label','sms'))
        rows = []
        for row in reader:
            data = (row['sms'], row['label'])
            rows.append(data)

    output_data = []

    for row in rows[:1000]:
    # get the sms body and compute the feature dictionary
    # add the tuple of feature_dict, label to output_data
        feature_dict = sms_features(rows[0][0])

    # add the tuple of feature_dict, label to output_data
        data = (feature_dict, rows[0][1])
        output_data.append(data)

    return output_data

def get_training_and_validation_sets(feature_sets):
    """
    This takes the output of `get_feature_sets`, randomly shuffles it to ensure we're
    taking an unbiased sample, and then splits the set of features into
    a training set and a validation set.
    """
    # randomly shuffle the feature sets
    random.shuffle(feature_sets)

    # get the number of data points that we have
    count = len(feature_sets)
    # 20% of the set, also called "corpus", should be training, as a rule of thumb, but not gospel.

    # we'll slice this list 20% the way through
    slicing_point = int(.20 * count)

    # the training set will be the first segment
    training_set = feature_sets[:slicing_point]

    # the validation set will be the second segment
    validation_set = feature_sets[slicing_point:]
    return training_set, validation_set


def run_classification(training_set, validation_set):
    # train the NaiveBayesClassifier on the training_set
    classifier = nltk.NaiveBayesClassifier.train(training_set)
    # let's see how accurate it was
    accuracy = nltk.classify.accuracy(classifier, validation_set)
    print "The accuracy was.... {}".format(accuracy)
    return classifier

def predict(classifier, new_sms):
    """
    Given a trained classifier and a fresh data point (an sms),
    this will predict its label, either spam or ham.
    """
    return classifier.classify(sms_features(new_sms))


# Now let's use the above functions to run our program
start_time = time.time()

print "Let's use Naive Bayes!"

our_feature_sets = get_feature_sets()
our_training_set, our_validation_set = get_training_and_validation_sets(our_feature_sets)
print "Size of our data set: {}".format(len(our_feature_sets))

print "Now training the classifier and testing the accuracy..."
classifier = run_classification(our_training_set, our_validation_set)

classifier.show_most_informative_features()

end_time = time.time()
completion_time = end_time - start_time
print "It took {} seconds to run the algorithm".format(completion_time)