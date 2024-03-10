import tensorflow as tf
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification, create_optimizer, DataCollatorWithPadding
from sklearn.model_selection import train_test_split
from datasets import Dataset
import re
import evaluate
import json

pd.set_option('display.max_columns', None)
desired_width = 320
pd.set_option('display.width', desired_width)
pd.set_option('display.max_rows', None)
np.set_printoptions(threshold=np.inf)
tf.random.set_seed(33)

# 0. Loading data -----------------------------------------------------------------------------------------------------
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# 1. Preprocessing  - inspired by these notebooks: --------------------------------------------------------------------
# https://www.kaggle.com/code/burhanuddinlatsaheb/transformer-model-comparision-for-disaster-tweet/notebook
# https://www.kaggle.com/code/diegomachado/seqclass-5-mybestsolution-0-843/notebook
def decontracted(phrase):
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

# def clean_dataset(text):
#     emoji_pattern = re.compile("["
#                                u"\U0001F600-\U0001F64F"  # emoticons
#                                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
#                                u"\U0001F680-\U0001F6FF"  # transport & map symbols
#                                u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
#                                u"\U00002500-\U00002BEF"  # chinese char
#                                u"\U00002702-\U000027B0"
#                                u"\U00002702-\U000027B0"
#                                u"\U000024C2-\U0001F251"
#                                u"\U0001f926-\U0001f937"
#                                u"\U00010000-\U0010ffff"
#                                u"\u2640-\u2642"
#                                u"\u2600-\u2B55"
#                                u"\u200d"
#                                u"\u23cf"
#                                u"\u23e9"
#                                u"\u231a"
#                                u"\ufe0f"  # dingbats
#                                u"\u3030"
#                                "]+", flags=re.UNICODE)
#     text = emoji_pattern.sub(r'', text)
#     return text

def preprocessing(text):
    # text = clean_dataset(text)
    text = text.replace('#','')
    text = decontracted(text)
    text = re.sub('\S*@\S*\s?','',text)
    text = re.sub('http[s]?:(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+','',text)
    return text.strip()

train_df['text'] = train_df['text'].apply(lambda x: preprocessing(x))
test_df['text'] = test_df['text'].apply(lambda x: preprocessing(x))

# 2. Creating training and validation sets -----------------------------------------------------------------------------
X_train, X_valid = train_test_split(train_df, test_size=0.2, random_state=33, stratify=train_df['target'])

train = Dataset.from_pandas(X_train)
valid = Dataset.from_pandas(X_valid)
test = Dataset.from_pandas(test_df)

# 3. Model, tokenizer, data_collator + tokenization and creation of datasets --------------------------------------------
model_checkpoint = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = TFAutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=2)

def preprocess(df):
    return tokenizer(df['text'])


tokenized_train = train.map(preprocess, batched=True)
tokenized_valid = valid.map(preprocess, batched=True)
tokenized_test = test.map(preprocess, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")

tf_train_set = tokenized_train.to_tf_dataset(
    columns=["attention_mask", "input_ids"],
    label_cols=["target"],
    shuffle=True,
    collate_fn=data_collator,
    batch_size=16,
)

tf_valid_set = tokenized_valid.to_tf_dataset(
    columns=["attention_mask", "input_ids"],
    label_cols=["target"],
    shuffle=False,
    collate_fn=data_collator,
    batch_size=16,
)

tf_test_set = tokenized_test.to_tf_dataset(
    columns=['attention_mask', 'input_ids'],
    shuffle=False,
    collate_fn=data_collator,
    batch_size=16
    )

# 4. Setting parameters, hyperparameters, and training -----------------------------------------------------------------
batch_size = 16
num_epochs = 5
batches_per_epoch = len(tokenized_train) // batch_size
total_train_steps = int(batches_per_epoch * num_epochs)
optimizer, schedule = create_optimizer(init_lr=2e-5, num_warmup_steps=0, num_train_steps=total_train_steps)
early_stopping = tf.keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)

model.compile(optimizer=optimizer, metrics=['accuracy'])
model.fit(x=tf_train_set, validation_data=tf_valid_set, epochs=num_epochs, callbacks=[early_stopping])


# 5. Creating labels and evaluating the model --------------------------------------------------------------------------
labels = []
for elem in tf_valid_set:
    labels.extend(elem[1].numpy())

labels = np.array(labels)

clf_metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])
preds = model.predict(tf_valid_set)
preds = np.argmax(preds['logits'], axis=1)
scores = clf_metrics.compute(predictions=preds, references=labels)
print(scores)

# 6. Saving the metrics ------------------------------------------------------------------------------------------------
with open(f'nlp_scores.json', 'w') as json_file:
    json.dump(scores, json_file)

# 7. Predictions and submission ----------------------------------------------------------------------------------------
result = model.predict(tf_test_set)
result = np.argmax(result['logits'], axis=1)

results = pd.DataFrame({'id': test_df['id'],
                        'target': result})
results.to_csv(f'roberta_lr=2e-05.csv', index=False)

# 8. Summary -----------------------------------------------------------------------------------------------------------
"""
Models:     On this code I have testes 3 models (distilbert_uncased, bert_uncased and roberta). Roberta was a clean winner.
Preprocessing:  The best results I have got using unhashed preprocessing functions.
Learning_rate:  I looped through the list of [1e-06, 2e-06, ...,9e-05] and 2e-05 was a clean winner.
Best public score for single model:     0.84155 - generated by the code above. 
Best public score for the ensemble:     0.844 - I averaged predictions of all roberta_models with different learning 
rates trained on unpreprocessed data, and also a few best ones trained on preprocessed data. Moreover I added a few 
distilbert_uncased and bert_uncased models (27 models total). It is not very computation optimal, so I didn't put it here.

Potential tasks for optimizing predictions: 
1. Polishing the character preprocessing functions. 
2. Playing out with various random_states and learning rates not only for roberta but for bert and distilbert as well. 
3. Adding a few another 'bert' models to the ensemble, bert-large (not enough ram), different roberta. 
4. Reducing number of models in ensemble, leaving only a few ones with different and strong biases to optimize 
generalization of the ensemble.
"""


