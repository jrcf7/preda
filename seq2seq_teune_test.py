import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# Setting up the device for GPU usage
from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'
print("Device: {}".format(device))

# Data

import pickle
import pandas as pd

SEED = 41
train_size = .8

file_name = 'path/to/dreambank_hvdc_txt.pickle'
with open(file_name, 'rb') as f:
    dreambank_hvdc = pickle.load(f)


columns=["Series", "Gender", "Year", "Age", "Report", "NoWords", "Encodings_dict"]
target_df_column = "Encodings_dict"

dreambank_hvdc_df = pd.DataFrame(
    dreambank_hvdc,
    columns=columns,
)

dreambank_hvdc_df['Encodings_dict'] = pd.Series(
    dreambank_hvdc_df['Encodings_dict'], dtype="string"
)


prefix_based_dataset = []

for dream_report, spelled_annotation, srs, gdr, yr, age in dreambank_hvdc_df[["Report", target_df_column, "Series", "Gender", "Year", "Age"]].values:
    annotation_features = spelled_annotation.split("\n")
    for feature_annotation in annotation_features:
        feature, annotation = feature_annotation.split(" : ", 2)
        if feature in ["Objectives", "Settings", "Modifiers"]:
            continue
        prefix_based_dataset.append([feature, dream_report, annotation, srs, gdr, yr, age])

prefix_based_dataset_df = pd.DataFrame(
    prefix_based_dataset,
    columns=["prefix", "dream_report", "annotation", "Series", "Gender", "Year", "Age"],
)

import nltk
nltk.download('punkt_tab')

prefix_based_dataset_df["Token_N°"] = [
    len(txt.split(" ")) for txt in prefix_based_dataset_df["annotation"].values
]

prefix_based_dataset_df["Token_N°_text"] = [
    len(txt.split(" ")) for txt in prefix_based_dataset_df["dream_report"].values
]


prefix_to_use = ['Characters', 'Activities', 'Emotion', 'Friendliness', 'Misfortune', 'Good Fortune']
prefix_based_dataset_df = prefix_based_dataset_df[
    prefix_based_dataset_df["prefix"].isin(prefix_to_use)
]

# shuffle the DataFrame rows
prefix_based_dataset_df = prefix_based_dataset_df.sample(
    frac = 1, random_state=SEED
).reset_index(drop=True)


encoder_max_length = 512
decoder_max_length = 128


prefix_based_dataset_df_Train = prefix_based_dataset_df.sample(
    frac=train_size,
    random_state=SEED
).reset_index(drop=True)

prefix_based_dataset_df_Train["set"] = ["Train"]*len(prefix_based_dataset_df_Train)

prefix_based_dataset_df_test = prefix_based_dataset_df.drop(
    prefix_based_dataset_df_Train.index
).reset_index(drop=True)

prefix_based_dataset_df_test["set"] = ["test"]*len(prefix_based_dataset_df_test)


# Model
### Setup

import torch
import numpy as np

torch.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True


import evaluate

learning_rate = .001
weight_decay = 0.01
batch_size = 32 # small-128, base-32, large-8
max_new_tokens = decoder_max_length
model_name = "google-t5/t5-base"
optim = "adamw_torch"
metric = evaluate.load("rouge") # load_metric("rouge")
per_device_train_batch_size = batch_size
per_device_eval_batch_size = batch_size
epochs = 20
seed = SEED
target_text = "annotation"
source_text = "dream_report"
prefix_clm = "prefix"
train_size = .8
actual_model = model_name if "/" not in model_name else model_name.split("/")[1]
output_dir = "PreDA_{}".format(actual_model)
push_to_hub = False
group_by_length = True
length_column_name = "Token_N°"
warmup_step = 10 # 10 for small/base; 0? for large
prefixes = list(set(prefix_based_dataset_df["prefix"]))


from transformers import Seq2SeqTrainingArguments

args = Seq2SeqTrainingArguments(
    output_dir,
    do_train=True,
    do_eval=True,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    metric_for_best_model="eval_loss",
    learning_rate=learning_rate,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    label_smoothing_factor=0.1,
    weight_decay=weight_decay,
    save_total_limit=2,
    num_train_epochs=epochs,
    predict_with_generate=True,
    generation_max_length=decoder_max_length,
    fp16=True, # for small & base              <-----
    # fp16_full_eval=True, # for large         <-----
    optim=optim,
    load_best_model_at_end=True,
    warmup_steps=warmup_step,
    logging_steps=10,
    group_by_length=group_by_length,
    length_column_name=length_column_name,
    push_to_hub=push_to_hub,
)


# Prepare and format the data for train and test

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq
from datasets import Dataset

model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=tokenizer.pad_token_id)


# Borrowed from https://github.com/huggingface/transformers/blob/master/examples/seq2seq/run_summarization.py

nltk.download("punkt", quiet=True)

def postprocess_text(preds, labels):
    preds  = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds  = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels

metric = evaluate.load("rouge")

def batch_tokenize_preprocess(
    batch,
    tokenizer,
    max_source_length,
    max_target_length,
    source_text,
    target_text,
    prefix_clm
    ):

    source, target, prefix = batch[source_text], batch[target_text], batch[prefix_clm]
    source = [
        "{} : {}".format(prefix[i], s)
        for i, s in enumerate(source)
    ]

    source_tokenized = tokenizer(
        source, padding="max_length", truncation=True, max_length=max_source_length
    )
    target_tokenized = tokenizer(
        target, padding="max_length", truncation=True, max_length=max_target_length
    )

    batch = {k: v for k, v in source_tokenized.items()}
    # Ignore padding in the loss
    batch["labels"] = [
        [-100 if token == tokenizer.pad_token_id else token for token in l]
        for l in target_tokenized["input_ids"]
    ]
    return batch

def compute_metrics(eval_preds):
    preds, labels = eval_preds

    # Replace -100 as we can not predict them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    preds  = np.where(preds != -100, preds, tokenizer.pad_token_id)

    # decode preds and labels
    decoded_preds  = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # rougeLSum expects newline after each sentence
    decoded_preds  = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    return result


train_data_txt = Dataset.from_pandas(prefix_based_dataset_df_Train)
test_data_txt = Dataset.from_pandas(prefix_based_dataset_df_test)

train_data = train_data_txt.map(
    lambda batch: batch_tokenize_preprocess(
        batch, tokenizer, encoder_max_length, decoder_max_length, source_text, target_text, prefix_clm
    ),
    batched=True,
    remove_columns=train_data_txt.column_names,
)

test_data = test_data_txt.map(
    lambda batch: batch_tokenize_preprocess(
        batch, tokenizer, encoder_max_length, decoder_max_length, source_text, target_text, prefix_clm
    ),
    batched=True,
    remove_columns=test_data_txt.column_names,
)

print("Train set: {}".format(len(train_data)))
print("Test set: {}".format(len(test_data)))


# Setup the trained object

from transformers import Seq2SeqTrainer

trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=train_data,
    eval_dataset=test_data,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)


### Train

trainer.train()


### Gather final results

tr_df = []
for dct in trainer.state.log_history:
    for k,v in dct.items():
        if (k == "step") or (k == "epoch"): continue
        tr_df.append([dct["epoch"], dct["step"], k, v])

tr_df = pd.DataFrame(tr_df, columns=["Epoch", "Step", "Variable", "Value"])

from tqdm import tqdm

full_model_predictions = []
single_rouges = []

for pref, test_dream, annotation, series, gender, yr, age, tknn, tknnctx, stt in tqdm(prefix_based_dataset_df_test.values):

    mok_inputs = ["{} : {}".format(pref, test_dream)]

    inputs = tokenizer(
        mok_inputs,
        max_length=encoder_max_length,
        truncation=True,
        return_tensors="pt"
    )

    output = trainer.model.generate(
        **inputs.to(device),
        do_sample=False,
        max_length=decoder_max_length,
    )

    decoded_output = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
    generated_text = nltk.sent_tokenize(decoded_output.strip())[0]

    full_model_predictions.append(generated_text)
    results = metric.compute(predictions=[generated_text], references=[annotation])

    single_rouges.append(list(results.values()))

prefix_based_dataset_df_test["Model Predictions"] = full_model_predictions

rouge_df = pd.DataFrame(
    single_rouges,
    columns=["Rouge 1", 'Rouge 2', "Rouge L", "Rouge L Sum"]
)

prefix_based_dataset_df_test = pd.concat(
    [prefix_based_dataset_df_test, rouge_df],
    axis=1
)

drm_id_dct = {dr:i for i, dr in enumerate(set(prefix_based_dataset_df_test["dream_report"]))}

prefix_based_dataset_df_test["Dream_id"] = [
    drm_id_dct[dr] for dr in prefix_based_dataset_df_test["dream_report"]
]


### Save results

pnrt_mdl_id = model_name.replace("/", "-")

tr_df.to_csv(f"preda_{pnrt_mdl_id}_loss_rouge.csv", index=False)
prefix_based_dataset_df_test.to_csv(f"preda_{pnrt_mdl_id}_results_full.csv", index=False)
prefix_based_dataset_df_Train.to_csv(f"preda_{pnrt_mdl_id}_train_set.csv", index=False)

