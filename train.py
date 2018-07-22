# https://www.jianshu.com/p/0511e938deb4
# tf crf解释: https://www.jianshu.com/p/c36974c8aa7d

from model.data_utils import CoNLLDataset
from model.ner_model import NERModel
from model.config import Config


def main():
    # create instance of config
    config = Config()

    # build model
    model = NERModel(config)
    model.build()
    # model.restore_session("results/crf/model.weights/") # optional, restore weights
    # model.reinitialize_weights("proj")

    # create datasets
    print("dev filename:", config.filename_dev)
    dev   = CoNLLDataset(config.filename_dev, config.processing_word_fuc,
                         config.processing_tag_fuc, config.max_iter)
    print("train filename:", config.filename_train)
    train = CoNLLDataset(config.filename_train, config.processing_word_fuc,
                         config.processing_tag_fuc, config.max_iter)

    # train model
    model.train(train, dev)

if __name__ == "__main__":
    main()
