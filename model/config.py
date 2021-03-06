import os
import platform

from .general_utils import get_logger
from .data_utils import get_trimmed_glove_vectors, load_vocab, \
        get_processing_word


class Config():
    def __init__(self, load=True):
        """Initialize hyperparameters and load vocabs

        Args:
            load_embeddings: (bool) if True, load embeddings into
                np array, else None

        """
        # directory for training outputs
        if not os.path.exists(self.dir_output):
            os.makedirs(self.dir_output)

        # create instance of logger
        self.logger = get_logger(self.path_log)

        # load if requested (default)
        if load:
            self.load()


    def load(self):
        """Loads vocabulary, processing functions and embeddings

        Supposes that build_data.py has been run successfully and that
        the corresponding files have been created (vocab and trimmed GloVe
        vectors)

        """
        # 1. vocabulary
        self.vocab_words = load_vocab(self.filename_words)
        self.vocab_tags  = load_vocab(self.filename_tags)
        self.vocab_chars = load_vocab(self.filename_chars)

        self.nwords     = len(self.vocab_words)
        self.nchars     = len(self.vocab_chars)
        self.ntags      = len(self.vocab_tags)

        # 2. get processing functions that map str -> id
        self.processing_word_fuc = get_processing_word(self.vocab_words,
                                                       self.vocab_chars, lowercase=True, chars=self.use_chars)
        self.processing_tag_fuc  = get_processing_word(self.vocab_tags,
                                                       lowercase=False, allow_unk=False)

        # 3. get pre-trained embeddings
        self.embeddings = (get_trimmed_glove_vectors(self.filename_trimmed)
                if self.use_pretrained else None)


    # general config
    dir_output = "results/test/"
    dir_model  = dir_output + "model.weights/"
    path_log   = dir_output + "log.txt"

    # embeddings
    dim_word = 300
    dim_char = 100

    # glove files
    sysstr = platform.system()
    if sysstr == "Windows":
        data_prefix = "d:/dataset/seq_tagging/" # 在windows下只好用绝对路径了
    else:
        data_prefix = "../../dataset/seq_tagging/"
    filename_glove = "{}/glove.6B/glove.6B.{}d.txt".format(data_prefix ,dim_word)
    # trimmed embeddings (created from glove_filename with build_data.py)
    filename_trimmed = "{}/glove.6B.{}d.trimmed.npz".format(data_prefix ,dim_word)
    use_pretrained = True

    # dataset
    # filename_dev = "data/coNLL/eng/eng.testa.iob"
    # filename_test = "data/coNLL/eng/eng.testb.iob"
    # filename_train = "data/coNLL/eng/eng.train.iob"

    # 为了演示,将dev,test,train设成相同
    filename_dev = filename_test = filename_train = "{}/test.txt".format(data_prefix) # test

    max_iter = None # if not None, max number of examples in Dataset

    # vocab (created from dataset with build_data.py)
    filename_words = "{}/words.txt".format(data_prefix)
    filename_tags = "{}/tags.txt".format(data_prefix)
    filename_chars = "{}/chars.txt".format(data_prefix)

    # training
    train_embeddings = False
    nepochs          = 15
    dropout          = 0.5
    batch_size       = 8
    lr_method        = "adam"
    lr               = 0.001
    lr_decay         = 0.9
    clip             = -1 # if negative, no clipping
    nepoch_no_imprv  = 3

    # model hyperparameters
    hidden_size_char = 8 # lstm on chars, 100
    hidden_size_lstm = 10 # lstm on word embeddings, 300

    # NOTE: if both chars and crf, only 1.6x slower on GPU
    use_crf = True # if crf, training is 1.7x slower on CPU
    use_chars = True # if char embedding, training is 3.5x slower on CPU

    tfdbg_mode = False
    debug_mode = True
