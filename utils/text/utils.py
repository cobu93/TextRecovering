import torch
import pickle
import math
import random
import re

################################################################
###     Classes definitions                                  ###
################################################################
class Book:
    def __init__(self, name, author, path):
        self.name = name
        self.author = author
        self.path = path
        self.content = self.__join_paragraphs__(self.__read_file__(self.path))

    def __read_file__(self, filename):
        content = []
        with open(filename) as f:
            for line in f:
                content.append(line.strip())

        return content

    # Join paragraphs defined in multi-lines
    def __join_paragraphs__(self, content):
        num_lines = len(content)

        paragraphs = []
        paragraph = []
        num_line = 0

        while num_line < num_lines:
            
            line = content[num_line].strip()
            
            while line:
                paragraph.append(line)
                num_line += 1

                if(num_line >= num_lines):
                    break

                line = content[num_line].strip()
            
            if len(paragraph) > 0:
                paragraphs.append(' '.join(paragraph))

            num_line += 1


            paragraph.clear()

        return paragraphs

    def __str__(self):
        return 'Name: {}   Author: {}'.format(self.name, self.author)

    __repr__ = __str__
        
    

class Vocabulary:

    UNKNOWN_STR = '<unk>'
    PAD_STR = '<pad>'
    START_STR = '<sos>'
    END_STR = '<eos>'

    UNKNOWN_TOKEN = 0
    PAD_TOKEN = 1   # Used for padding short sentences
    START_TOKEN = 2   # Start-of-sentence token
    END_TOKEN = 3   # End-of-sentence token

    def __init__(self, name):
        self.name = name
        self.word2index = {
                            self.UNKNOWN_STR: self.UNKNOWN_TOKEN, 
                            self.PAD_STR: self.PAD_TOKEN, 
                            self.START_STR: self.START_TOKEN, 
                            self.END_STR: self.END_TOKEN
                        }
        self.index2word = {
                            self.UNKNOWN_TOKEN: self.UNKNOWN_STR, 
                            self.PAD_TOKEN: self.PAD_STR, 
                            self.START_TOKEN: self.START_STR, 
                            self.END_TOKEN: self.END_STR
                        }
        self.word2count = {
                            self.UNKNOWN_STR: 0,
                            self.PAD_STR: 0,
                            self.START_STR: 0,
                            self.END_STR: 0
                        }   
        self.num_words = len(self.index2word)

    def __len__(self):
        return len(self.word2index)

    def add_word(self, word):
        if word not in self.word2index:
            # First entry of word into vocabulary
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            # Word exists; increase word count
            self.word2count[word] += 1
    
    def to_word(self, index):
        if index in self.index2word:
            return self.index2word[index]
        else:
            return self.index2word[self.UNKNOWN_TOKEN]


    def to_index(self, word):
        if word in self.word2index:
            return self.word2index[word]
        else:
            return self.word2index[self.UNKNOWN_STR]

    def to_words(self, indices):
        _indices_ = []
        for index in indices:
            _indices_.append(self.to_word(index))
        return _indices_

    def to_indices(self, words):
        _words_ = []
        for word in words:
            _words_.append(self.to_index(word))
        return _words_
        




class BookDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        data, 
        vocabulary,
        unknown_char=Vocabulary.UNKNOWN_STR, 
        pad_char=Vocabulary.PAD_STR, 
        start_char=Vocabulary.START_STR, 
        end_char=Vocabulary.END_STR
    ):
        # Defines dataset
        self.data = []
        self.vocabulary = vocabulary

        # For all sentences
        for sentence in data:
            # Add start character
            example = [vocabulary.to_index(start_char)]
            # Get sentence indices
            example.extend(vocabulary.to_indices(sentence))
            # Add end character
            example.append(vocabulary.to_index(end_char))
            self.data.append(example)
            
        self.data = torch.LongTensor(self.data)


    def change_vocab(self, vocab, start_char=Vocabulary.START_STR, end_char=Vocabulary.END_STR):
        # For all sentences
        new_data = []
        for sentence in self.data:
            # Add start character
            
            example = vocab.to_indices(
                    self.vocabulary.to_words(sentence.cpu().numpy())
                )
            
            
            new_data.append(example)

        self.vocabulary = vocab
        self.data = torch.LongTensor(new_data)



    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

################################################################
###     Functions                                            ###
################################################################

def load_books(filename, author=None):
    with open(filename, 'rb') as f:
        books = pickle.load(f)

    # Find current author's books
    if author is not None:
        books = [book for book in books if book.author == author]

    return books

TRAIN_DATASET_STATE = 'train_dataset'
TEST_DATASET_STATE = 'test_dataset'
VOCAB_STATE = 'vocab_dataset'

def save_text_state(filename, vocab, train_dataset, test_dataset):
    with open(filename, 'wb') as f:
        pickle.dump({
            TRAIN_DATASET_STATE: train_dataset,
            TEST_DATASET_STATE: test_dataset,
                VOCAB_STATE: vocab
        }, f)

def load_text_state(filename):
    with open(filename, 'rb') as f:
        text_state = pickle.load(f)

    train_dataset = text_state[TRAIN_DATASET_STATE]
    test_dataset = text_state[TEST_DATASET_STATE]
    vocab = text_state[VOCAB_STATE]

    return vocab, train_dataset, test_dataset


def build_vocab(dataset, vocab_name):
    vocab = Vocabulary(vocab_name)

    for sentence in dataset:
        for word in sentence:
            vocab.add_word(word)
    
    return vocab

def build_dataset(books, min_sentence_length, max_sentence_length):
    dataset = []
    for book in books:
        for sentence in book.content:
            balanced_sentence = [word.lower().strip() for word in sentence[:max_sentence_length]]
            if len(balanced_sentence) >= min_sentence_length:
                balanced_sentence.extend([Vocabulary.PAD_STR] * (max_sentence_length - len(sentence)))
                dataset.append(balanced_sentence)

    return dataset



def build_text_state(books, min_sentence_length, max_sentence_length, validation_partition, vocab_name='vocab', shuffle=True):
    dataset = build_dataset(books, min_sentence_length, max_sentence_length)

    if shuffle:
        random.shuffle(dataset)

    # Partition dataset
    partition_idx = math.floor(len(dataset) * (1 - validation_partition))
    train_dataset = dataset[:partition_idx]
    test_dataset = dataset[partition_idx:]

    # Creates vocabulary and trim sentences
    vocab = build_vocab(train_dataset, vocab_name)

    train_dataset = BookDataset(train_dataset, vocab)
    test_dataset = BookDataset(test_dataset, vocab)

    return vocab, train_dataset, test_dataset









# Removes chapters titles
DELETE_CHAPTER_REGEX = r'^\s*(chapter|chap|\d|[MDCLXVI]+)+'
# Remove unwanted characers
DELETE_CHARACTER_REGEX = r'[-|_|\+|"|\(|\)]+'

# delete chapters (titles)
def delete_chapters(content, regex=DELETE_CHAPTER_REGEX):
    paragraphs = []

    for paragraph in content:
        if not re.search(regex, paragraph.strip(), flags=re.IGNORECASE):
            paragraphs.append(paragraph)

    return paragraphs

# Remove unwanted characters
def delete_character(content, regex=DELETE_CHARACTER_REGEX):
    paragraphs = []

    for paragraph in content:
        clean_line = re.sub(regex, ' ', paragraph.strip(), flags=re.IGNORECASE).strip()
        if clean_line:
            paragraphs.append(clean_line)        

    return paragraphs   

def process_book_content(book):
    text = delete_chapters(book.content)
    text = delete_character(text)
    return text



# Honorific titles regex
ABREVS_REGEX = r'(Mr|Mrs|Ms|Dr|Prof|Jr|Hon|Rev|St|[A-Z])\.'
# Sentence splitting
SPLIT_SENTENCE_REGEX = r'[\.|;]+'
# Splitting as words
WORDS_EXTRACTION_REGEX = r'([\W|\'|:|\?|!|])'
# Replace valid separators(non idea separator) with this char
TEMPORAL_CHAR = '#'

def split_sentences(text, 
    abrevs_regex=ABREVS_REGEX,
    split_regex=SPLIT_SENTENCE_REGEX, 
    words_extraction_regex=WORDS_EXTRACTION_REGEX, 
    temp_char=TEMPORAL_CHAR
    ):

    abrevs_temp_str = ''.join([temp_char * 5])
    
    # Removes abrevs
    result_abrevs_remove_group = re.findall(abrevs_regex, text)
    result_abrevs_remove = re.sub(abrevs_regex, abrevs_temp_str, text)
    
    abrevs_group = 0
    
    sentences_words = []  
    
    # Split sentences
    sentences = re.split(split_regex, result_abrevs_remove)

    for sentence in sentences:
        # Restore abrevs
        while re.search(abrevs_temp_str, sentence):
            sentence = sentence.replace(abrevs_temp_str, result_abrevs_remove_group[abrevs_group])
            abrevs_group += 1

        # Split in words
        splitted_sentence = re.split(words_extraction_regex, sentence)
        splitted_sentence = [sp for sp in splitted_sentence if sp.strip() ]
        if len(splitted_sentence) > 0:
            sentences_words.append(splitted_sentence)
    
    return sentences_words