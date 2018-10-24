from utils.normalize import unicodeToAscii, normalizeString
import torchvision.datasets as dset
import pickle

"""
convert word into indexes.
After executing this script, save voc instance which have vocabulary as a pickle file.
When training and testing, load and use it.
"""
class Voc:
    def __init__(self):
        self.PAD_token = 0
        self.SOS_token = 1
        self.EOS_token = 2
        self.word2index = {}
        self.word2count = {}
        self.index2word = {self.PAD_token: "PAD", self.SOS_token: "SOS", self.EOS_token: "EOS"}
        self.num_words = 3

    def addSentence(self, sentence):
        for word in sentence.split(" "):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1


if __name__ == "__main__":
    cap = dset.CocoCaptions(root="/mnt/mqs02/data/nishimura/mscoco/train2017/",
                            annFile="../data/annotations/captions_train2017.json")
    voc = Voc()
    for img, captions in cap:
        for caption in captions:
            normalized_sentence = normalizeString(caption)
            voc.addSentence(normalized_sentence)
    with open("voc.pkl", "wb") as f:
        pickle.dump(f)
    import ipdb; ipdb.set_trace()
