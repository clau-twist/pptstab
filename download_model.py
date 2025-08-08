from transformers import BertModel, BertTokenizer

# Define the model from the Hub and a clean local path
model_name = "Rostlab/prot_bert"
save_directory = "./prot_bert"

# Download and save the tokenizer and model
print("Loading model...")
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# Make all model tensors contiguous in memory
print("Ensuring all tensor are contiguous...")
for param in model.parameters():
    param.data = param.data.contiguous()

print("Saving model...")
tokenizer.save_pretrained(save_directory)
model.save_pretrained(save_directory)

print(f"Model and tokenizer saved to {save_directory}")
