import pickle
import os

path_in = "/data_local/slm/processed_train_one_billion.txt"
path_out = "/data_local/slm/vocab.pkl"

if os.path.exists(path_out):
    vocab = pickle.load(open(path_out, "rb"))
else:
    vocab = {}
    with open(path_in, "r") as fr, open(path_out, "wb") as f:
        for line in fr.readlines():
            line = line.lower()
            for w in line.strip().split():
                if w not in vocab:
                    vocab[w] = 1
                else:
                    vocab[w] += 1

        pickle.dump(vocab, f, protocol=0)

keep_vocab = {}
for s in vocab:
    if vocab[s] > 10:
        keep_vocab[s] = vocab[s]

sorted_vocab = sorted(keep_vocab.items(), key=lambda s: s[1], reverse=True)
print(len(sorted_vocab))
print(sorted_vocab[:100])
#
pickle.dump(sorted_vocab, open("gec_vocab.pkl", "wb"))
