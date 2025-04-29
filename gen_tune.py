# f"<|user|>\nreport: {rpt}\nprefix:{prfx}{tokenizer.eos_token}\n<|assistant|>\nannotation: {anntn}{tokenizer.eos_token}\n"


# Dependecies
import os

os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'

import pickle, nltk, math, tqdm
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import Dataset

#  Inports
push_to_hub = True
batch_size = 8
train_epochs = 20
mdl_id = "meta-llama/Llama-3.2-3B"
mn = mdl_id.split("/")[1]
tokenizer = AutoTokenizer.from_pretrained(mdl_id)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(mdl_id)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


# Data processing
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

prefix_based_dataset_df = prefix_based_dataset_df.sample(
    frac = 1, random_state=SEED
).reset_index(drop=True)

prefix_based_dataset_df["llm_txt"] = [
    f"{prfx} : {rpt}\nannotation: {anntn}{tokenizer.eos_token}"
    for prfx, rpt, anntn, _, _, _, _, _, _ in prefix_based_dataset_df.values
]

prefix_based_dataset_df_Train = prefix_based_dataset_df.sample(
    frac=train_size,
    random_state=SEED
).reset_index(drop=True)

prefix_based_dataset_df_Train["set"] = ["Train"]*len(prefix_based_dataset_df_Train)

prefix_based_dataset_df_test = prefix_based_dataset_df.drop(
    prefix_based_dataset_df_Train.index
).reset_index(drop=True)

prefix_based_dataset_df_test["set"] = ["test"]*len(prefix_based_dataset_df_test)

train_data = [
    {"text":llm_txt}
    for _, _, _, _, _, _, _, _, _, llm_txt, _ in prefix_based_dataset_df_Train.values
]

hf_train = Dataset.from_list(train_data)

test_data = [
    {"text":llm_txt}
    for _, _, _, _, _, _, _, _, _, llm_txt, _ in prefix_based_dataset_df_test.values
]
hf_test = Dataset.from_list(test_data)

def preprocess_function(examples):
    return tokenizer(
        [x for x in examples["text"]],
        max_length=128,
        truncation=True,
    )

tokenized_hf_train = hf_train.map(
    preprocess_function,
    batched=True,
    num_proc=4,
    remove_columns=hf_train.column_names,
)

tokenized_hf_test = hf_test.map(
    preprocess_function,
    batched=True,
    num_proc=4,
    remove_columns=hf_test.column_names,
)


# Training
def compute_ppl(preds):
    return math.exp(preds["eval_results"])

training_args = TrainingArguments(
    output_dir=f"drama_{mn}",
    eval_strategy="epoch",
    save_strategy="epoch",
    #eval_steps=10,
    num_train_epochs=train_epochs,
    learning_rate=.001,
    weight_decay=0.01,
    push_to_hub=push_to_hub,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    label_smoothing_factor=0.1,
    metric_for_best_model="eval_loss",
    load_best_model_at_end=True,
    save_total_limit=2,
    hub_private_repo=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_hf_train,
    eval_dataset=tokenized_hf_test,
    data_collator=data_collator,
    tokenizer=tokenizer,
    # compute_metrics=compute_ppl
)

trainer.train()


# Testing
eval_results = trainer.evaluate()
print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")


# Save and Push model
tr_df = []
for dct in trainer.state.log_history:
    for k,v in dct.items():
        if (k == "step") or (k == "epoch"): continue
        tr_df.append([dct["epoch"], dct["step"], k, v])

tr_df = pd.DataFrame(tr_df, columns=["Epoch", "Step", "Variable", "Value"])
tr_df.to_csv(f"preda_{mn}_loss_rouge.csv", index=False) # no rouge, due to consistency


prefix_based_dataset_df_Train.to_csv(f"preda_{mn}_train_set.csv", index=False)
prefix_based_dataset_df_test.to_csv(f"preda_{mn}_test_set.csv", index=False)

if push_to_hub:
    trainer.push_to_hub()
