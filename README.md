# Random-Indexing-in-Information-retrieval

For finding the similar words using the context vector and cosine similarity

Steps are as follow:
1) We first tokenize the files and for each word, create an index vector of fixed length(1024 used here) containing 5% -1 or 1.
2) This we can do by using random seed values which is unique in every iteration and using a particular threshold.
3) Then, we create the context vector using the following formula.

Statement - 
The fox jumps quicker than the rabbit.

If we have to find the context vector of "quicker", what we do is 

CV(quicker) = CV(quicker) + 0.25* IV(fox) + 0.5* IV(jumps) + 0.5* IV(rabbit)

Remember, we do not have to consider the stopwords.

4) Then we calculate the similarity using the cosine similarity rule between the context vector of one word with all other
words in the vocabulary and sorting in the reverse order.
5) And finally we can query how many words we want of the same context as the given word.
