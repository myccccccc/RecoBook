import numpy as np
import pandas as pd

def rmse(y,h):
    """RMSE
    Args:
        y: real_table y
        h: predicted_table h
    Returns:
        RMSE
    """
    a = y-h
    a = a.reshape(a.size)
    a = a[~np.isnan(a)]

    return np.sqrt(sum(a**2)/len(a))

def dcg_k(r, k):
    """ Discounted Cumulative Gain (DGC)  
    Args:
        r: True Ratings in Predicted Rank Order (1st element is top recommendation)
        k: Number of results to consider
    Returns:
        DCG
    """
  
    r = np.asfarray(r)[:k]
    return np.sum(2**r / np.log2(np.arange(2, r.size + 2)))      



def ndcg_k(r, k=10):
    """Normalized Discounted Cumulative Gain (NDCG)
    Args:
        r: True Ratings in Predicted Rank Order (1st element is top recommendation)
        k: Number of results to consider
    Returns:
        NDCG
    """
    dcg_max = dcg_k(sorted(r, reverse=True), k)
    if not dcg_max:
        return 0.
    return dcg_k(r, k) / dcg_max

def divSco_k(r, tail, k=10):
    """Diversity Score
    Args:
        r: bookids in Predicted Rank Order (1st element is top recommendation)
        tail: list of less popular/less known books
        k: Number of results to consider
    Returns:
        Diversity Score
    """
    count = 0
    for bookid in r[:k]:
        if bookid in tail:
            count += 1
    return count / k
	
def gettail(reallyfinalratings):
	"""Get tail list to compute diversity score
	Args:
		reallyfinalratings: pd dataframe
	Returns:
		tail dataframe
	"""
	tailcomp = reallyfinalratings[["newbook_id", "rating"]].groupby("newbook_id").agg(len).rename(columns={"rating":"count"}).sort_values(by='count', ascending=False).reset_index()
	tot = sum(tailcomp['count'])
	tailcomp['popshare']= [x/tot for x in tailcomp['count']]
	tailcomp['popsharecumsum']= tailcomp['popshare'].cumsum()
	tailcomp['category']= ['Head' if x<0.95 else "Tail" for x in tailcomp['popsharecumsum']]
	tail = tailcomp[tailcomp['category'] == 'Tail']
	return tail
