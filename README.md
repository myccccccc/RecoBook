# Simple Book Recommenders

​	Some different simple book recommenders using 

​	1. Naïve Bayes 

​	2. TF-iDF 

​	3. Matrix Factorization

​	4. Neural Networks

​	Original Goodreads Dataset can be downloaded [here](https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/home?authuser=0). I only used 11219 users, 8000 books and 1590749 rating data for training, validation and test.

<img src="\pictures\7.png" alt="7"  />

​																		  Fig.1 sample book data

<img src=".\pictures\8.png" alt="8" style="zoom: 45%;" />

​																		  Fig.2 sample rating data

<img src=".\pictures\9.png" style="zoom: 33%;" />

<img src=".\pictures\10.png" style="zoom: 45%;" />

​																		  Fig.3 data distribution





##  Popularity & Naïve Bayes

​	The Popularity model is a really simple approach, it ranks items based on their popularity among users and suggesting the top k to all users, all users have same predicted ratings to all books(average book rating)

|       | RMSE  | nDGC  | Diversity Score |
| :---- | ----- | ----- | --------------- |
| Train | 0.954 | 0.662 | 0               |
| Test  | 0.950 | 0.795 | 0               |

​	Each book in the dataset has a description giving us a very high-level idea of the key themes and topics in the book. The Naïve Bayes model first break these book description into lists of words, remove punctuation and stop words, and derive rating probabilities for each user based on the book vocabulary. We train our model on each user individually.

|       | RMSE  | nDCG  | Diversity Score |
| ----- | ----- | ----- | --------------- |
| Train | 0.057 | 1.000 | 0.092           |
| Test  | 1.194 | 0.807 | 0.092           |



## TF-iDF

​	[Term frequency–inverse document frequency](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) is a numerical statistic that is intended to reflect how important a word is to a document in a collection or corpus. Using this concept, we represent the book dataset as a TF-IDF matrix of size B×W where W is the number of words in our dictionary, B is the number of books. Then we compute an affinity factor between books i and j in the form of cosine similarity. and obtain a B×B Affinity Matrix. Then use these similarities as weights to predict user ratings.
|       | RMSE  | nDCG  | Diversity Score |
| ----- | ----- | ----- | --------------- |
| Train | 0.910 | 0.616 | 0.130           |
| Test  | 0.917 | 0.690 | 0.130           |



## Matrix Factorization

​	Using the concept of [Matrix Factorization](http://albertauyeung.com/2017/04/23/python-matrix-factorization.html), we are able to assume the existence of k latent features such that our U×B rating matrix R can be represented as the product of two lower-dimension matrices: P of size U×k and Q of size B×k. The [original implementation](http://albertauyeung.com/2017/04/23/python-matrix-factorization.html) of Matrix Factorization is not perfect, there appears to be gradient exploding when train on very large matrix. I have to limit the abs value of gradient to 0.1 while updating parameters.

```Python
change = self.alpha * (e * self.Q[j, :] - self.beta * self.P[i,:])
self.P[i, :] += np.array(list(map(lambda x: 0 if abs(x) > 0.1 else x, change)))
change = self.alpha * (e * P_i - self.beta * self.Q[j,:])
self.Q[j, :] += np.array(list(map(lambda x: 0 if abs(x) > 0.1 else x, change)))
```

​	And I also added `val_size ` and `decay_rate` in `mf.py`.

​	First, I tried different k(# of latent features) for training.

![11](.\pictures\11.png) 

<img src=".\pictures\12.png" alt="12" style="zoom: 67%;" />

​	As we can see with smaller k, we have high training error, as k increases, the training error continue to decrease. But when we look at the validation error, as k increases, the validation error will first decrease and then increase. Because with more latent features(larger k) the model might over fit to the training data, So I choose k=4 in the final MF model. (but there are still many other hyper parameters like alpha, beta and  initialization need to be tuned in the future)

<img src=".\pictures\6.png" alt="6" style="zoom: 70%;" />
|       | RMSE  | nDCG  | Diversity Score |
| ----- | ----- | ----- | --------------- |
| Train | 0.806 | 0.607 | 0.046           |
| Test  | 0.854 | 0.688 | 0.046           |









