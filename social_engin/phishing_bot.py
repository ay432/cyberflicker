import json
import tweepy
from adodbapi.examples.xls_read import extended


def create_tweepy_api(api_key, api_secret_key, access_token, access_token_secret):
    """
    Create and authenticate a Tweepy API object.

    :param api_key: Twitter API Key
    :param api_secret_key: Twitter API Secret Key
    :param access_token: Twitter Access Token
    :param access_token_secret: Twitter Access Token Secret
    :return: Tweepy API object
    """
    # Authenticate to Twitter
    auth = tweepy.OAuthHandler(api_key, api_secret_key)
    auth.set_access_token(access_token, access_token_secret)

    # Create the API object
    api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True, compression=True)

    # Verify authentication
    try:
        api.verify_credentials()
        print("Authentication OK")
    except tweepy.TweepError as e:
        print(f"Error during authentication: {e}")

    return api

# Replace the placeholders with your actual Twitter API credentials
API_KEY = "your_api_key"
API_SECRET_KEY = "your_api_secret_key"
ACCESS_TOKEN = "your_access_token"
ACCESS_TOKEN_SECRET = "your_access_token_secret"

# Create the Tweepy API
tweepy_api = create_tweepy_api(API_KEY, API_SECRET_KEY, ACCESS_TOKEN, ACCESS_TOKEN_SECRET)

# Example usage: Get your Twitter account details
user = tweepy_api.me()
print(f"Authenticated as: {user.name} (@{user.screen_name})")

user_id = "realTrump"
user = tweepy_api.get_user(user_id)
for friend in user.friends:
    print(friend.screen_name)

count = 200
user_tweets = tweepy_api.user_timeline(screen_name=user_id, count=count, tweet_mode=extended)
print(len(user_tweets))

corpus = []
for tweet in user_tweets:
    corpus.append(tweet.full_text)
text = "\n".join(corpus)

print(text)

import re
def RemoveURLs(string):
    url = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', "", string)
    return url

processed_text = RemoveURLs(text)
import markovify
text_model = markovify.Text(processed_text)

generate_tweets = []
for i in range(5):
    generate_tweets.append(text_model.make_sentence())
print(generate_tweets)

phishing_link = "https://urlzs.com/Dw8s"

output_tweets = [x + " " + phishing_link for x in generate_tweets]
for output_tweet in output_tweets:
    print(output_tweet)
