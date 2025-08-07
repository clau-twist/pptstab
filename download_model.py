from transformers import BertModel, BertTokenizer

# Define the model from the Hub and a clean local path
model_name = "Rostlab/prot_bert"
save_directory = "./prot_bert"

# Download and save the tokenizer and model
print("Saving tokenizer...")
tokenizer = BertTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(save_directory)

print("Saving model...")
model = BertModel.from_pretrained(model_name)
model.save_pretrained(save_directory)

print(f"Model and tokenizer saved to {save_directory}")
