from  tqdm import tqdm
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling
from datasets import Dataset
import evaluate

metric = evaluate.load("rouge")

mdl_id = "path_to_tuned/drama_Llama-3.2-3B"
model = AutoModelForCausalLM.from_pretrained(mdl_id)
model.to("cuda")

tokenizer = AutoTokenizer.from_pretrained(mdl_id)
tokenizer.pad_token = tokenizer.eos_token

mn = mdl_id.split("/")[1]

prefix_based_dataset_df_test = pd.read_csv(f"{mn}_test_set.csv")

full_model_predictions = []
single_rouges = []

for prfx, rpt, annot in tqdm(prefix_based_dataset_df_test[["prefix", "dream_report", "annotation"]].values):

    qr = f"<|user|>\nreport: {rpt}\nprefix:{prfx}{tokenizer.eos_token}\n<|assistant|>"

    inputs = tokenizer(
        qr,
        return_tensors = "pt"
    ).to("cuda")

    outputs = model.generate(
        **inputs, 
        max_new_tokens=150,
        use_cache=True,
        do_sample=False,
        temperature=0,
        top_p=0,
        pad_token_id=tokenizer.eos_token_id,
    )
    answer=tokenizer.batch_decode(outputs)
    answer = answer[0].split("annotation: ")[-1]
    results = metric.compute(predictions=[answer], references=[annot])

    full_model_predictions.append(answer)
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

prefix_based_dataset_df_test.to_csv(f"{mn}_results_full.csv", index=False)