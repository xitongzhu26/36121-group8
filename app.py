import json
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn

with open("tag2id.json", "r", encoding="utf-8") as f:
    tag2id = json.load(f)
id2tag = {i: tag for tag, i in tag2id.items()}
num_labels = len(tag2id)

class BERTTagRecommender(nn.Module):
    def __init__(self, model_name, output_dim):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(outputs.pooler_output)

model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = BERTTagRecommender(model_name, num_labels)
model.load_state_dict(torch.load("checkpoint_model.pt", map_location=torch.device("cpu")))
model.eval()

def predict_tags(prompt, top_k=8):
    try:
        if "[SEP]" not in prompt:
            return "Error: input must contain '[SEP]' to separate context_tags and target_character."
        parts = prompt.split("[SEP]")
        context_tags = parts[0].strip().split(",")
        target_character = parts[1].strip()
        text = ", ".join(context_tags) + " [SEP] " + target_character
        encoded = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=64)
        with torch.no_grad():
            logits = model(encoded["input_ids"], encoded["attention_mask"])
            probs = torch.sigmoid(logits).squeeze().numpy()
        top_indices = probs.argsort()[::-1][:top_k]
        results = [(id2tag[i], float(probs[i])) for i in top_indices if probs[i] > 0.2]
        return results
    except Exception as e:
        return f"Error: {str(e)}"

demo = gr.Interface(
    fn=predict_tags,
    inputs=gr.Textbox(label="text", placeholder="tag1, tag2, tag3 [SEP] character"),
    outputs=gr.Textbox(label="output"),
    title="Prompt Tagging"
)

if __name__ == "__main__":
    demo.launch()
