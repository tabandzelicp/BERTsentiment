import transformers

MAX_LEN = 512
TRAINING_BATCH_SIZE = 32
VALIDATION_BATCH_SIZE = 8
EPOCHS = 2
TRAINING_PATH = '../input/imdb.csv'
BERT_PATH = '../input/bert-base-uncased/'
MODEL_PATH = 'model.bin'
# Bert tokenizer is WorldPiece tokenizer
TOKENIZER = transformers.BertTokenizer.from_pretrained(
    vocab_file = BERT_PATH,
    do_lower_case = True
)

