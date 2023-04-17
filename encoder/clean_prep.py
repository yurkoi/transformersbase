from datasets import load_dataset, Dataset, DatasetDict, ClassLabel
from transformers import DataCollatorWithPadding
from transformers import BertTokenizerFast
from torch.utils.data import DataLoader
import configs


checkpoint = 'bert-base-uncased'
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')


def map_start_to_label(review):
    if review["stars"] < 3:
        review["stars"] = 0
    elif review["stars"] == 3:
        review["stars"] = 1
    else:
        review["stars"] = 2
    return review


def tokenize_fn(batch):
    return tokenizer.batch_encode_plus(batch['fulltext'], max_length=128, padding='max_length', truncation=True)


def clean_prepare():
    dataset = load_dataset("amazon_reviews_multi", "en")
    df_imdb_train = dataset['train'].to_pandas()
    df_imdb_val = dataset['validation'].to_pandas()
    maps_cart = {indx: i for i, indx in enumerate(df_imdb_train['product_category'].value_counts().index)}
    df_imdb_train['product_category'] = df_imdb_train['product_category'].apply(lambda x: maps_cart[x])
    df_imdb_val['product_category'] = df_imdb_val['product_category'].apply(lambda x: maps_cart[x])
    df_imdb_train['stars'] = df_imdb_train['stars'].apply(lambda x: int(x))
    df_imdb_val['stars'] = df_imdb_val['stars'].apply(lambda x: int(x))
    df_imdb_train['fulltext'] = df_imdb_train['review_title'] + " [sep] " + df_imdb_train['review_body']
    df_imdb_val['fulltext'] = df_imdb_val['review_title'] + " [sep] " + df_imdb_val['review_body']
    df_imdb_train['charLen'] = df_imdb_train['fulltext'].apply(lambda x: len(x))
    df_imdb_val['charLen'] = df_imdb_val['fulltext'].apply(lambda x: len(x))
    df_imdb_train['wordLen'] = df_imdb_train['fulltext'].apply(lambda x: len(x.split()))
    df_imdb_val['wordLen'] = df_imdb_val['fulltext'].apply(lambda x: len(x.split()))
    tds = Dataset.from_pandas(df_imdb_train)
    vds = Dataset.from_pandas(df_imdb_val)
    dataset = DatasetDict()
    dataset['train'] = tds
    dataset['validation'] = vds
    dataset = dataset.map(map_start_to_label)

    # convert feature from Value to ClassLabel
    class_feature = ClassLabel(names=['negative', 'neutral', 'positive'])
    dataset = dataset.cast_column("stars", class_feature)

    tokenized_datasets = dataset.map(tokenize_fn, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    tokenized_datasets = tokenized_datasets.remove_columns(
        ['review_id', 'product_id', 'reviewer_id', 'review_body', 'review_title', 'language', 'product_category',
         'fulltext', 'charLen', 'wordLen', ])
    tokenized_datasets = tokenized_datasets.rename_column("stars", "labels")

    train_loader = DataLoader(
        tokenized_datasets['train'],
        shuffle=True,
        batch_size=configs.BATCH_SIZE,
        collate_fn=data_collator
    )

    valid_loader = DataLoader(
        tokenized_datasets['validation'],
        batch_size=configs.BATCH_SIZE,
        collate_fn=data_collator
    )
    return train_loader, valid_loader


if __name__ == "__main__":
    pass
