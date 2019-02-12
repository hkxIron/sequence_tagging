#conll2003 dataset: https://github.com/hkxIron/NeuroNER/blob/master/data/conll2003/en/test.txt
import numpy as np
import os


# shared global variables to be imported from model also
UNK = "$UNK$"
NUM = "$NUM$"
NONE = "O"


# special error message
class MyIOError(Exception):
    def __init__(self, filename):
        # custom error message
        message = """
ERROR: Unable to locate file {}.

FIX: Have you tried running python build_data.py first?
This will build vocab file from your train, test and dev sets and
trimm your word vectors.
""".format(filename)
        super(MyIOError, self).__init__(message)


class CoNLLDataset(object):
    """Class that iterates over CoNLL Dataset

    __iter__ method yields a tuple (words, tags)
        words: list of raw words
        tags: list of raw tags

    If processing_word and processing_tag are not None,
    optional preprocessing is appplied

    Example:
        ```python
        data = CoNLLDataset(filename)
        for sentence, tags in data:
            pass
        ```

    """
    def __init__(self, filename, processing_word_fuc=None, processing_tag_fuc=None,
                 max_iter=None):
        """
        Args:
            filename: path to the file
            processing_words: (optional) function that takes a word as input
            processing_tags: (optional) function that takes a tag as input
            max_iter: (optional) max number of sentences to yield

        """
        self.filename = filename
        self.processing_word_fuc = processing_word_fuc
        self.processing_tag_fuc = processing_tag_fuc
        self.max_iter = max_iter
        self.length = None

    # 实现了迭代器的接口,因此可以遍历了
    def __iter__(self):
        niter = 0
        with open(self.filename, 'r', encoding='UTF-8') as f:
            words, tags = [], []
            for line in f:
                line = line.strip()
                if (len(line) == 0 or line.startswith("-DOCSTART-")): # 遇到空行,说明一个句子结束
                    if len(words) != 0:
                        niter += 1
                        if self.max_iter is not None and niter > self.max_iter:
                            break
                        yield words, tags # 生成generator
                        words, tags = [], [] # 返回值后,清空 words与tags
                else:
                    # line: European B-ORG
                    #       Union I-ORG
                    ls = line.split(' ')
                    # word: European tag: B-ORG
                    word, tag = ls[0],ls[-1]
                    if self.processing_word_fuc is not None:
                        # word会被替换成char_ids以及word_id,即 European => (char_ids=[22, 20, 0, 25, 10, 24, 9, 1], word_ids=9)
                        word = self.processing_word_fuc(word)
                    if self.processing_tag_fuc is not None:
                        # tag会被替换成tag_id,即 B-ORG => 2
                        tag = self.processing_tag_fuc(tag)
                    words += [word] # 与 words.append([word]) 一样
                    tags += [tag]


    def __len__(self):
        """Iterates once over the corpus to set and store length"""
        if self.length is None:
            self.length = 0
            for _ in self:
                self.length += 1

        return self.length


def get_vocabs(datasets):
    """Build vocabulary from an iterable of datasets objects

    Args:
        datasets: a list of dataset objects

    Returns:
        a set of all the words in the dataset

    """
    print("Building vocab...")
    vocab_words = set()
    vocab_tags = set()
    for dataset in datasets: # train, dev, test
        for words, tags in dataset: # dataset.iter()
            vocab_words.update(words) # words为list, 因此是将words中的所有元素都加入vocab_words
            vocab_tags.update(tags)
    print("- done. {} tokens".format(len(vocab_words)))
    return vocab_words, vocab_tags


def get_char_vocab(dataset):
    """Build char vocabulary from an iterable of datasets objects

    Args:
        dataset: a iterator yielding tuples (sentence, tags)

    Returns:
        a set of all the characters in the dataset

    """
    # vocab_char得到的都是单词拆分后的char
    vocab_char = set()
    for words, _ in dataset:
        for word in words:
            vocab_char.update(word) # 集合update方法：是把要传入的元素拆分，做为个体传入到集合中

    return vocab_char


def get_glove_vocab(filename):
    """Load vocab from file

    Args:
        filename: path to the glove vectors

    Returns:
        vocab: set() of strings
    """
    print("Building vocab...")
    vocab = set()
    with open(filename, 'r', encoding='UTF-8') as f:
        for line in f:
            word = line.strip().split(' ')[0]
            vocab.add(word)
    print("- done. {} tokens".format(len(vocab)))
    return vocab


def write_vocab(vocab, filename):
    """Writes a vocab to a file

    Writes one word per line.

    Args:
        vocab: iterable that yields word
        filename: path to vocab file

    Returns:
        write a word per line

    """
    print("Writing vocab...")
    with open(filename, "w") as f:
        for i, word in enumerate(vocab):
            if i != len(vocab) - 1: # 不是最后一行,要加\n
                f.write("{}\n".format(word))
            else:
                f.write(word)
    print("- done. {} tokens".format(len(vocab)))


def load_vocab(filename):
    """Loads vocab from a file

    Args:
        filename: (string) the format of the file must be one word per line.

    Returns:
        d: dict[word] = index

    """
    try:
        d = dict()
        with open(filename, 'r', encoding='UTF-8') as f:
            for idx, word in enumerate(f):
                word = word.strip()
                d[word] = idx

    except IOError:
        raise MyIOError(filename)
    return d


def export_trimmed_glove_vectors(vocab, glove_filename, trimmed_filename, dim):
    """Saves glove vectors in numpy array

    Args:
        vocab: dictionary vocab[word] = index
        glove_filename: a path to a glove file
        trimmed_filename: a path where to store a matrix in npy
        dim: (int) dimension of embeddings

    """
    embeddings = np.zeros([len(vocab), dim])
    with open(glove_filename, 'r', encoding='UTF-8') as f:
        for line in f:
            line = line.strip().split(' ')
            word = line[0]
            embedding = [float(x) for x in line[1:]]
            if word in vocab:
                word_idx = vocab[word]
                embeddings[word_idx] = np.asarray(embedding)

    np.savez_compressed(trimmed_filename, embeddings=embeddings) # 将以"embeddings"为key进行存储


def get_trimmed_glove_vectors(filename):
    """
    Args:
        filename: path to the npz file

    Returns:
        matrix of embeddings (np array)

    """
    try:
        with np.load(filename) as data:
            return data["embeddings"]

    except IOError:
        raise MyIOError(filename)

"""
1. processing_word_fuc = get_processing_word(lowercase=True) # 生成lambda func时会确定外层的参数
2. word = self.processing_word_fuc(word) # 调用的时候,会确定内层的func参数
感觉这样实现了类的功能,但比类要轻量级
"""
def get_processing_word(vocab_words=None, vocab_chars=None,
                    lowercase=False, chars=False, allow_unk=True):
    """Return lambda function that transform a word (string) into list,
    or tuple of (list, id) of int corresponding to the ids of the word and
    its corresponding characters.

    Args:
        vocab: dict[word] = idx

    Returns:
        f("cat") = ([12, 4, 32], 12345)
                 = (list of char ids, word id)

    """
    def f(word):
        # 0. get chars of words
        if vocab_chars is not None and chars == True:
            char_ids = []
            for char in word:
                # ignore chars out of vocabulary
                if char in vocab_chars:
                    char_ids += [vocab_chars[char]]

        # 1. preprocess word
        if lowercase: # 大写转换为小写
            word = word.lower()
        if word.isdigit(): # 数字替换
            word = NUM

        # 2. get id of word
        if vocab_words is not None:
            if word in vocab_words:
                word = vocab_words[word]
            else:
                if allow_unk:
                    word = vocab_words[UNK] # 未登录词
                else:
                    raise Exception("Unknow key is not allowed. Check that "\
                                    "your vocab (tags?) is correct")

        # 3. return tuple char ids, word id
        if vocab_chars is not None and chars == True:
            return char_ids, word
        else:
            return word

    return f


def _pad_sequences(sequences, pad_tok, max_length):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with

    Returns:
        a list of list where each sublist has same length

    examples:
    seq= [[1], [2,4,6]]
    seq_padded = [[1,0,0],[2,4,6]]
    """
    sequence_padded, sequence_length = [], []
    # 这里的sequence_length是真实的length,而非padding之后的length
    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_length] + [pad_tok]*max(max_length - len(seq), 0) # 将一个序列追加到另一个序列后面
        sequence_padded +=  [seq_]
        sequence_length += [min(len(seq), max_length)]

    return sequence_padded, sequence_length


def pad_sequences(sequences, pad_tok, nlevels=1):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with
        nlevels: "depth" of padding, for the case where we have characters ids

    Returns:
        a list of list where each sublist has same length

    """
    if nlevels == 1:
        max_length = max(map(lambda x : len(x), sequences)) #对sequences中的每个元素求长度,然后取最大的长度,即得到此batch中的最长句子长度
        sequence_padded, sequence_length = _pad_sequences(sequences,
                                            pad_tok, max_length)

    elif nlevels == 2:
        # 取得batch中这些单词的最大长度
        max_length_word = max([max(map(lambda x: len(x), seq))
                               for seq in sequences])
        sequence_padded, sequence_length = [], []
        # 先对单词填充到相同长度
        for seq in sequences:
            # all words are same length now
            sp, sl = _pad_sequences(seq, pad_tok, max_length_word)
            sequence_padded += [sp]
            sequence_length += [sl]
        # 再对句子填充到相同长度
        max_length_sentence = max(map(lambda x : len(x), sequences))
        sequence_padded, _ = _pad_sequences(sequence_padded,
                [pad_tok]*max_length_word, max_length_sentence)
        sequence_length, _ = _pad_sequences(sequence_length, 0,
                max_length_sentence)

    return sequence_padded, sequence_length


def minibatches(data, minibatch_size):
    """
    Args:
        data: generator of (sentence, tags) tuples
        minibatch_size: (int)

    Yields:
        list of tuples

    """
    x_batch, y_batch = [], []
    # x:[tuple(char_ids=[6,24,9,1], word_id=18), tuple(char_ids=[2,18,24,0,0,24], word_id=20)]
    # y:tag_ids=[6,1,7,7,4,3,7]
    for (x, y) in data:
        if len(x_batch) == minibatch_size:
            # x_batch: [x=[char_ids=([6,9,1],[2,3,4],...,[5]), word_id=(18,20,15,...)],x=[...]]
            # y_batch:[tag_id=[6,1,7,7,4],tag_id=[6,3,4,9],...]
            yield x_batch, y_batch
            x_batch, y_batch = [], []
        """
        >>> x
        [(char_ids=[6, 24, 9, 1], word_id=18), (char_ids=[2, 18, 24, 0, 0, 24], word_id=20), ([23, 18, 3, 24, 4], 15), ([18, 1], 11), ([7, 24, 8], 12), ([15, 25, 0, 26], 0), ([5], 2)]
        >>> new_x=zip(*x)
        >>> new_x 
        [char_ids=([6, 24, 9, 1], [2, 18, 24, 0, 0, 24], [23, 18, 3, 24, 4], [18, 1], [7, 24, 8], [15, 25, 0, 26], [5]), 
        word_id=(18, 20, 15, 11, 12, 0, 2)]
        """

        if type(x[0]) == tuple:
            x = zip(*x) # 将元素按列组合起来,即将char_ids组合在一起,word_id组合成单独一列tuple
        x_batch += [x] # append
        y_batch += [y]

    # 将最后一批剩余的数据生成generator
    if len(x_batch) != 0:
        yield x_batch, y_batch


def get_chunk_type(tag_id, idx_to_tag_dict):
    """
    Args:
        tag_id: id of tag, ex 4
        idx_to_tag_dict: dictionary {4: "B-PER", 5: "I-LOC", 6:"B-LOC"}

    Returns:
        tuple: "B", "PER"

    """
    tag_name = idx_to_tag_dict[tag_id]
    tag_class = tag_name.split('-')[0] # 代表下标开始或者在其中:B,I,O
    tag_type = tag_name.split('-')[-1] # 类别是人名还是地方:LOC, PER, O
    return tag_class, tag_type


def get_chunks(tag_id_seq, tag_to_id):
    """Given a sequence of tags, group entities and their position

    Args:
        tag_id_seq: [4, 4, 0, 0, ...] sequence of label tags
        tag_to_id: dict["O"] = 4

    Returns:
        list of (chunk_type, chunk_start, chunk_end)

    Example:
        tag_id_seq = [4, 5, 0, 3]
        tags = {"B-PER": 4, "I-PER": 5, "B-LOC": 3}
        result = [("PER", 0, 2), ("LOC", 3, 4)]
        result表示 PER在tag_id_seq中index=0处开始,index=2处结束(不含)
                  LOC在tag_id_seq中index=3处开始,index=4处结束
    """
    default = tag_to_id[NONE] # NONE = "O"
    idx_to_tag = {idx: tag for tag, idx in tag_to_id.items()}
    chunks = []
    chunk_type, chunk_start = None, None
    for i, tag_id in enumerate(tag_id_seq):
        # End of a chunk 1
        if tag_id == default and chunk_type is not None:
            # Add a chunk.
            chunk = (chunk_type, chunk_start, i)
            chunks.append(chunk)
            chunk_type, chunk_start = None, None

        # End of a chunk + start of a chunk!
        elif tag_id != default:
            # class: B,I  type:LOC,PER
            tok_chunk_class, tok_chunk_type = get_chunk_type(tag_id, idx_to_tag)
            if chunk_type is None:
                chunk_type, chunk_start = tok_chunk_type, i
            elif tok_chunk_type != chunk_type or tok_chunk_class == "B":
                chunk = (chunk_type, chunk_start, i)
                chunks.append(chunk)
                chunk_type, chunk_start = tok_chunk_type, i
        else:
            pass

    # end condition
    if chunk_type is not None:
        chunk = (chunk_type, chunk_start, len(tag_id_seq))
        chunks.append(chunk)

    return chunks
