
import streamlit as st  
from textblob import TextBlob
import pandas as pd
import altair as alt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import plotly.express as px


# Fxn
def convert_to_df(sentiment):
	sentiment_dict = {'polarity':sentiment.polarity,'subjectivity':sentiment.subjectivity}
	sentiment_df = pd.DataFrame(sentiment_dict.items(),columns=['metric','value'])
	return sentiment_df

def analyze_token_sentiment(docx):
	analyzer = SentimentIntensityAnalyzer()
	pos_list = []
	neg_list = []
	neu_list = []
	for i in docx.split():
		res = analyzer.polarity_scores(i)['compound']
		if res > 0.1:
			pos_list.append(i)
			pos_list.append(res)

		elif res <= -0.1:
			neg_list.append(i)
			neg_list.append(res)
		else:
			neu_list.append(i)

	result = {'positives':pos_list,'negatives':neg_list,'neutral':neu_list}
	return result 

# Function to extract tweets from CSV
def extract_tweets(keyword, num_tweets):
    df = pd.read_csv("Twitter_Data.csv")
    df = df[df["clean_text"].str.contains(keyword, case=False)]
    df = df[:num_tweets]
    return df

def analyze_sentiment(text):
    sentiments = []
    for tweet in text:
        sentiment = TextBlob(tweet).sentiment.polarity
        if sentiment > 0:
            sentiments.append(('positive', sentiment))
        elif sentiment < 0:
            sentiments.append(('negative', sentiment))
        else:
            sentiments.append(('neutral', sentiment))
    return sentiments

def get_sentiment_counts(sentiments):
    pos_count = sum(1 for sentiment in sentiments if sentiment[0] == 'positive')
    neg_count = sum(1 for sentiment in sentiments if sentiment[0] == 'negative')
    neu_count = sum(1 for sentiment in sentiments if sentiment[0] == 'neutral')
    tweet_sentiments = [(tweet, sentiment) for tweet, sentiment in sentiments]
    return pos_count, neg_count, neu_count, tweet_sentiments


def main():
    st.title("Sentiment Analysis NLP App")
    st.subheader("Streamlit Projects")

    menu = ["Home", "Extract from Twitter", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.subheader("Home")
        with st.form(key='nlpForm'):
            raw_text = st.text_area("Enter Text Here")
            submit_button = st.form_submit_button(label='Analyze')

        # layout
        col1,col2 = st.columns(2)
        if submit_button:

            with col1:
                st.info("Results")
                sentiment = TextBlob(raw_text).sentiment
                st.write(sentiment)

                # Emoji
                if sentiment.polarity > 0:
                    st.markdown("Sentiment:: Positive :smiley: ")
                elif sentiment.polarity < 0:
                    st.markdown("Sentiment:: Negative :angry: ")
                else:
                    st.markdown("Sentiment:: Neutral 😐 ")

                # Dataframe
                result_df = convert_to_df(sentiment)
                st.dataframe(result_df)

                # Visualization
                c = alt.Chart(result_df).mark_bar().encode(
                    x='metric',
                    y='value',
                    color='metric')
                st.altair_chart(c,use_container_width=True)

            with col2:
                st.info("Token Sentiment")
                token_sentiments = analyze_token_sentiment(raw_text)
                st.write(token_sentiments)

    elif choice == "Extract from Twitter":
        st.subheader("Extract from Twitter")
        with st.form(key='twitterForm'):
            keyword = st.text_input("Enter keyword to search on Twitter")
            num_tweets = st.number_input("Enter number of tweets to fetch", min_value=1, max_value=1000, step=1)
            submit_button = st.form_submit_button(label='Extract')
	
        if submit_button:
            # Fetch tweets
            tweets_df = extract_tweets(keyword, num_tweets)
            if not tweets_df.empty:
                st.write(tweets_df)
                # Analyze sentiment
                sentiments = analyze_sentiment(tweets_df['clean_text'])
			for tweet, sentiment, polarity in tweet_sentiments:
			    if sentiment == 'positive':
				st.markdown(f"> {tweet} :smiley:")
			    elif sentiment == 'negative':
				st.markdown(f"> {tweet} :angry:")
			    else:
				st.markdown(f"> {tweet} 😐")

                pos_count, neg_count, neu_count, tweet_sentiments = get_sentiment_counts(sentiments)
                st.write("Sentiment Analysis")
                st.write("Positive: ", pos_count)
                st.write("Negative: ", neg_count)
                st.write("Neutral: ", neu_count)
                st.write("Tweet Sentiments:")
                                # Pie chart
                pie_data = {'Positive': pos_count, 'Negative': neg_count, 'Neutral': neu_count}
                pie_df = pd.DataFrame.from_dict(pie_data, orient='index', columns=['count'])
                fig = px.pie(pie_df, values='count', names=pie_df.index, title='Sentiment Distribution')
                st.plotly_chart(fig)
            else:
                st.warning("No tweets found.")

    else:
        st.subheader("About")


if __name__ == '__main__':
	main()
