import os
import logging

from nltk.corpus import stopwords

from gensim import corpora, utils
from gensim.matutils import corpus2dense
from gensim.models import TfidfModel

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA


STOP_LIST = set(stopwords.words('english'))
logging.basicConfig(level=logging.DEBUG)


class ClassifierNotTrainedException(Exception):

    DEFAULT_MESSAGE = 'Classifier is not trained, run <train> method before proceeding further'

    def __init__(self, msg=None):
        super(ClassifierNotTrainedException, self).__init__(msg or self.DEFAULT_MESSAGE)


class Texts:

    def __init__(self, sources):
        super(Texts, self).__init__()
        self.sources = sources
        self.sentences = []

    def to_vector(self):
        self.sentences = []
        for source in self.sources:
            with utils.open(source) as fin:
                for item_no, line in enumerate(fin):
                    words = utils.to_unicode(line).split()
                    words = [word for word in words if word not in STOP_LIST]
                    self.sentences.append(words)
        return self.sentences


class TfIdfClassifierWrapper:

    base_classifier_class = LogisticRegression

    def __init__(self, training_path=None, test_path=None, apply_pca=False, pca_n_components=50):
        super(TfIdfClassifierWrapper, self).__init__()

        self.training_path = training_path or './dataset/trainingset/'
        self.test_path = test_path or './dataset/testset/'

        self.sources = []
        self.labels = []

        self.classifier = self.base_classifier_class(solver='liblinear')
        self.pca = PCA(n_components=pca_n_components) if apply_pca else None
        self.is_trained = False

        self.logger = logging.getLogger('tf_idf')

        self.training_text_matrix = None
        self.training_sources_length = None

    def process_dataset(self, path, is_positive):
        path_suffix = 'positive' if is_positive else 'negative'
        for home, dirs, files in os.walk(path + path_suffix):
            for filename in files:
                self.sources.append(home + '/' + filename)
                self.labels.append(int(is_positive))

    def train(self):
        self.process_dataset(self.training_path, True)
        self.process_dataset(self.training_path, False)

        self.training_sources_length = len(self.sources)
        self.logger.debug(f'After train set processing: sources len {len(self.sources)}, labels len {len(self.labels)}')

        self.process_dataset(self.test_path, True)
        self.process_dataset(self.test_path, False)

        self.logger.debug(f'After full processing: sources len {len(self.sources)}, labels len {len(self.labels)}')

        corpus = Texts(self.sources).to_vector()
        dictionary = corpora.Dictionary(corpus)
        corpus = [dictionary.doc2bow(text) for text in corpus]
        model = TfidfModel(corpus)
        corpus = [text for text in model[corpus]]
        self.training_text_matrix = corpus2dense(corpus, num_terms=len(dictionary.token2id)).T

        if self.pca:
            self.training_text_matrix = self.pca.fit_transform(self.training_text_matrix)

        self.classifier.fit(self.training_text_matrix[:self.training_sources_length],
                            self.labels[:self.training_sources_length])

        self.is_trained = True

    def write_classification_report(self):
        if self.is_trained:
            pred_labels = self.classifier.predict(self.training_text_matrix[self.training_sources_length:])
            print(f'Logistic: {classification_report(self.labels[self.training_sources_length:], pred_labels)}')
        else:
            raise ClassifierNotTrainedException()


if __name__ == '__main__':
    tf_idf_classifier = TfIdfClassifierWrapper(apply_pca=True)
    tf_idf_classifier.train()
    tf_idf_classifier.write_classification_report()
