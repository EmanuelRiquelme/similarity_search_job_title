import pandas as pd
import torch.nn.functional as F
import torch
from transformers import AutoTokenizer, AutoModel

model_ckpt = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModel.from_pretrained(model_ckpt)

def get_embedding(job_title):
    encoded_input = tokenizer(job_title, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        model_output = model(**encoded_input)
    token_embeddings = model_output.last_hidden_state
    attention_mask = encoded_input.attention_mask
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    vector_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )
    return vector_embeddings

def cosine_similarity(x, y):
    # Compute dot product
    dot_product = torch.dot(x, y)
    # Compute norms
    norm_x = torch.norm(x)
    norm_y = torch.norm(y)
    # Compute cosine similarity
    cosine_similarity = dot_product / (norm_x * norm_y)
    return cosine_similarity
def make_spreadsheet(dataset,predictions):
    name = dataset['Name'].tolist()
    df = {
        'name': name,
        'job': predictions
         }
    df = pd.DataFrame(df)
    df.to_csv('prediction.csv')

def inference(dataset,labels):
    vector_embeddings = get_embedding(labels)
    primitive_names = {primitive_name: vector_embeddings[i] for i, primitive_name in enumerate(labels)}
    jobs_list = dataset['Job'].tolist()
    predictions = []
    for job_listed in jobs_list:
        job_title = get_embedding(job_listed).squeeze(0)
        acc = []
        for job,emb in primitive_names.items():
            acc.append(cosine_similarity(emb,job_title))
        predictions.append(labels[acc.index(max(acc))])
    make_spreadsheet(dataset,predictions)
    print('predictions saved on prediction.csv')


if __name__ == '__main__':
    dataset = pd.read_csv('dataset.csv')
    labels = pd.read_csv('labels.csv')['job'].tolist()
    inference(dataset,labels)

