from sklearn.datasets import fetch_20newsgroups
#cats = ['alt.atheism', 'sci.space']
#n_train = fetch_20newsgroups(subset='train', categories=cats)
#list(n_train.target_names)
n_train = fetch_20newsgroups(subset='train')
n_test = fetch_20newsgroups(subset='test')
