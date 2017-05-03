# NewsMaker
A django based news provider service which automatically associates an image with the news article you write, summarizes the article and labels it based on different news topics.


TECHNOLOGIES/LIBRARIES USED :
● DJANGO
● PYTHON NLTK LIBRARY
● BING IMAGE SEARCH API
● SCIKIT-LEARN TOOLS FOR PYTHON

a .  Assigning images to news articles :
One of the features of our app helps in automatically assigning images to news text.
We do this by combining text analysis with Bing Image search API.

b .  News Article Classification :
We used scikit-learn tools for the implementation of our classification feature. 
Using this, we are able to analyze a collection of documents and train the system and assign a category to our news article.

c .  Summarisation  :
To achieve summarization we have extracted the important snippets out ofthe news article that contains key information about the text.
We are taking the most meaningful sentence ( using our sentence intersection algorithm ) from every paragraph and conjoining all the selected sentences to form a summary.
Therefore, to get an ideal summary, the text should be atleast 500 words, with 3+ paragraphs, and 3-4 sentences/paragraph .
