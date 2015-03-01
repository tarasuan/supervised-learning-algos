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

    for row in rows[:10]:
    # get the sms body and compute the feature dictionary
    # add the tuple of feature_dict, label to output_data
        feature_dict = sms_features(rows[0][0])

    # add the tuple of feature_dict, label to output_data
        data = (feature_dict, rows[0][1])
        output_data.append(data)

    return output_data