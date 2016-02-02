import numpy as np, re, theanets

chars = re.sub(r'\s+', ' ', open('corpus.txt').read().lower())
txt = theanets.recurrent.Text(chars, min_count=10)
A = 1 + len(txt.alpha)  # of letter classes

# create a model to train: input -> gru -> relu -> softmax.
net = theanets.recurrent.Classifier([
    A, (100, 'gru'), (1000, 'relu'), A])

x = txt.classifier_batches(100, 32)()
print x[0].shape, x[1].shape

# train the model iteratively; draw a sample after every epoch.
seed = txt.encode(txt.text[300017:300050])
for tm, _ in net.itertrain(txt.classifier_batches(100, 32), momentum=0.9):
    print('{}|{} ({:.1f}%)'.format(
        txt.decode(seed),
        txt.decode(net.predict_sequence(seed, 40)),
        100 * tm['acc']))

