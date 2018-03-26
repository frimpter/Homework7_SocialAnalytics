

```python
# HW7.1 Social Analytics - News Mood
import json
import tweepy
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
# TextBlob documentation/tutorial: http://textblob.readthedocs.io/en/dev/quickstart.html
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
# VADER documentation on GitHub: https://github.com/cjhutto/vaderSentiment

analyzer = SentimentIntensityAnalyzer()

# Twitter API Keys
consumer_key = "LL7NVFqNgt9m6dxBQQGTDy9oz"
consumer_secret = "tPyaUhp6GcShpS4BPaBWFVTeaw1VLZ9NeURyWC4y4AF6JVV8jl"
access_token = "975006981789487104-65UpeGTARtfxFhXXzlC4LkKjF1k7Dg8"
access_token_secret = "BoHhMe5jt20S6iFTsyyTMXOUHBbE0jHBgRjDUPRCkYFaP"

# Set up Tweepy API Authentication
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, parser=tweepy.parsers.JSONParser())

```


```python
# Identify news organizations. I am going to change them a little to 2 progressives, 2 newspapers, 2 conservatives:
# @NPR, @MotherJones, @nytimes, @washingtonpost, @drudge_report, @BreitbartNews
# Sentiment analysis from most recent 100 tweets
# For each tweet, pull the source account, tweet, date, and sentiment scores

news_outlets = ["@NPR", "@MotherJones", "@nytimes", "@washingtonpost", "@drudge_report", "@BreitbartNews"]

records = []

for source in news_outlets:
    tweeter = api.user_timeline(source, count=100, result_type="recent")
    tweet_count = 1
    for tweet in tweeter:
        compound = analyzer.polarity_scores(tweet["text"])["compound"]
        pos = analyzer.polarity_scores(tweet["text"])["pos"]
        neu = analyzer.polarity_scores(tweet["text"])["neu"]
        neg = analyzer.polarity_scores(tweet["text"])["neg"]
        records.append({
            "Number": tweet_count,
            "Source": source,
            "Tweet": tweet["text"],
            "Positive Score": pos,
            "Negative Score": neg,
            "Neutral Score": neu,
            "Compound Score": compound
        })
        tweet_count += 1

master_df = pd.DataFrame(records)
master_df = master_df[["Source", "Number", "Negative Score", "Neutral Score", "Positive Score", "Compound Score", "Tweet"]]
master_df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Source</th>
      <th>Number</th>
      <th>Negative Score</th>
      <th>Neutral Score</th>
      <th>Positive Score</th>
      <th>Compound Score</th>
      <th>Tweet</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>@NPR</td>
      <td>1</td>
      <td>0.196</td>
      <td>0.804</td>
      <td>0.000</td>
      <td>-0.5994</td>
      <td>Her sister, Cheryl Brown Henderson, confirmed ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>@NPR</td>
      <td>2</td>
      <td>0.000</td>
      <td>0.847</td>
      <td>0.153</td>
      <td>0.5574</td>
      <td>Linda Brown, who was at the center of the U.S....</td>
    </tr>
    <tr>
      <th>2</th>
      <td>@NPR</td>
      <td>3</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>Jenny and the Mexicats' one-of-a-kind musical ...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>@NPR</td>
      <td>4</td>
      <td>0.000</td>
      <td>0.842</td>
      <td>0.158</td>
      <td>0.4588</td>
      <td>What does a revitalized Anacostia River mean f...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>@NPR</td>
      <td>5</td>
      <td>0.274</td>
      <td>0.726</td>
      <td>0.000</td>
      <td>-0.5267</td>
      <td>FTC Confirms It's Investigating Facebook For P...</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Graph the compound sentiment for each tweet by source

sns.set(style="darkgrid", font_scale=1.5)

sns.lmplot(x="Number", y="Compound Score", data=master_df, size=10, hue="Source", legend_out=True, fit_reg=False)

plt.title("Sentiment Analysis of Media Outlet Tweets - March 26, 2018")
plt.ylabel("Compound Sentiment Score\n← Negative      Positive  →")
plt.ylim(-1.05,1.05)
plt.xlabel("Last 100 Tweets (1 is most recent)")
plt.xlim(0, 101)

plt.savefig("HW7.1_Sentiment Analysis by Tweets.png")

plt.show()
```


![png](output_2_0.png)



```python
# Graph mean compund scores for each news source

sentiment_df = master_df.groupby(["Source"]).mean()
sentiment_df = sentiment_df[["Compound Score"]]

sns.set(style="darkgrid", font_scale=1.5)

sentiment_df.plot(kind="bar", figsize=(10,10))

plt.title("Overall Sentiment Analysis of Media Outlet Tweets - March 26, 2018")
plt.ylabel("Mean Compound Sentiment Score\n← Negative      Positive  →")
plt.ylim(-0.2,0.2)
plt.xlabel("Media Source")
plt.hlines(0, -1, 6, lw=3, colors="#000000", alpha=0.5)

plt.savefig("HW7.1_Sentiment Analysis by Source.png")

plt.show()
```


![png](output_3_0.png)



```python
print("Three Trends from Media Outlet Social Analytics: ")
print("\n1. The initial data suggest that a large proportion of all tweets may be categorized as neutral.")
print("\n2. Although a statistical analysis has not been conducted, there does not appear to be an immediately apparent trend in the directionality of sentiments related to the journalistic, liberal-, or conservative-leaning outlets.")
print("\n3. Sentiment analysis results should not be generalized based on a small sample, as this is related as much \nto the news of the day as to the voice of the media outlet. Comparative sentiment analysis of media outlets \nwould be best understood for tweets by the outlet, and by the anlayzed tweets, on an apples-to-apples basis \nin terms of the stories or topics being tweeted about.")
```

    Three Trends from Media Outlet Social Analytics: 
    
    1. The initial data suggest that a large proportion of all tweets may be categorized as neutral.
    
    2. Although a statistical analysis has not been conducted, there does not appear to be an immediately apparent trend in the directionality of sentiments related to the journalistic, liberal-, or conservative-leaning outlets.
    
    3. Sentiment analysis results should not be generalized based on a small sample, as this is related as much 
    to the news of the day as to the voice of the media outlet. Comparative sentiment analysis of media outlets 
    would be best understood for tweets by the outlet, and by the anlayzed tweets, on an apples-to-apples basis 
    in terms of the stories or topics being tweeted about.

