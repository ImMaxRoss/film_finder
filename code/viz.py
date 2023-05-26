import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def word_plot(
    data,target_column,value_column,n_words=10,
    theme=None
):
    # original plot modified from Flatiron School

    # Set up figure and axes
    item_count = data[target_column].nunique()
    fig, ax = plt.subplots(nrows=item_count+1,
        figsize=(12, item_count*4))

    # Empty dict to hold words that have already been plotted and their colors
    plotted_words_and_colors = {}
    # Establish color palette to pull from
    # (If you get an error message about popping from an empty list, increase this #)
    color_palette = sns.color_palette(theme, n_colors=(n_words*item_count))

    # Creating a plot for each unique sentiment
    data_by_sentiment = [y for _, y in data.groupby(target_column, as_index=False)]
    top_words = {}
    for i, sentiment_df in enumerate(data_by_sentiment):
        # Find top 10 words in this genre
        # print(all_words_in_tweet.str.split(' '))
        all_words_in_tweet = sentiment_df[value_column].explode()
        top_n = all_words_in_tweet.value_counts()[:n_words]
        
        # Select appropriate colors, reusing colors if words repeat
        colors = []
        for word in top_n.index:
            if word not in plotted_words_and_colors:
                new_color = color_palette.pop(0)
                plotted_words_and_colors[word] = new_color
            colors.append(plotted_words_and_colors[word])
        
        # Select axes, plot data, set title
        ax_ = ax[i]
        ax_.bar(top_n.index, top_n.values, color=colors)
        ax_.set_title(sentiment_df.iloc[0][target_column].title())
        top_words[sentiment_df.iloc[0][target_column].title()] = top_n
        for label in ax_.get_xticklabels():
            label.set(rotation=15)

    all_words_ = data[value_column].explode()
    top_all = all_words_.value_counts()[:n_words]

    colors = []
    for word in top_all.index:
        if word not in plotted_words_and_colors:
            new_color = color_palette.pop(0)
            plotted_words_and_colors[word] = new_color
        colors.append(plotted_words_and_colors[word])
    ax[-1].bar(top_all.index,top_all.values,color=colors)
    ax[-1].set(
        title="Top Words Across Sentiments"
    )
    plt.xticks(rotation=15)
    plt.suptitle("Word Breakdown by Sentiment")

    fig.tight_layout()
    repeats = pd.DataFrame(
        top_words
    ).dropna().index
    return (top_words, repeats)