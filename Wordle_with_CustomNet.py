import torch
import transformers
from models.CustomNet import CustomNet

# Define the input data (a word and some numbers)
input_word = "example"
input_numbers = torch.tensor([[1.0, 2.0, 3.0]])

# Tokenize the input word using the RoBERTa tokenizer
tokenizer = transformers.RobertaTokenizer.from_pretrained('roberta-base')
input_ids = tokenizer.encode(input_word, add_special_tokens=True, return_tensors='pt')
attention_mask = torch.ones_like(input_ids)

# Create an instance of the CustomNet model
net = CustomNet('roberta-base', num_numbers=3)

# Set the device to use for computation
device = torch.device('cpu')
net.to(device)

# Make a prediction with the model
with torch.no_grad():
    output = net(input_ids, attention_mask, input_numbers)
    # output = net(input_ids, input_numbers)

# Convert the output probabilities to a list of values
output_values = output.squeeze().tolist()

# Print the output values
print(output_values)