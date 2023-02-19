import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer
from models.CustomNetV2 import CustomNetV2
from tqdm import tqdm

def resize_word_list(word_list, max_len=5):
    """Resize a list of words into a fixed length, truncating or padding as necessary"""
    resized_list = []
    for word in word_list:
        # Truncate or pad word to fixed length
        if len(word) > max_len:
            resized_word = word[:max_len]
        else:
            resized_word = word + '0'*(max_len-len(word))
        resized_list.append(resized_word)
    return resized_list

# Define dataset class
word_column = 'Word'
input_columns = ['Month', 'Day', 'WeekNum', 'Contest number']
target_columns = ['Number of reported results', 'Number in hard mode', '1 try', '2 tries', '3 tries', '4 tries', '5 tries', '6 tries', '7 or more tries (X)']
class TrainingDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        word = row['Word']

        # Resize every word into 5 tokens, if word is longer than 5 tokens, truncate it, if word is shorter than 5 tokens, pad it with 0
        word = resize_word_list([word])[0]

        month = row['Month']
        day = row['Day']
        weeknum = row['WeekNum']
        contest_num = row['Contest number']
        targets = row[['Number of reported results', 'Number in hard mode', '1 try', '2 tries', '3 tries', '4 tries', '5 tries', '6 tries', '7 or more tries (X)']].values.astype(float)

        # Tokenize the input word
        input_ids = self.tokenizer.encode(word, add_special_tokens=True, return_tensors='pt', max_length=5, truncation=True)
        # fixed length of 5 tokens
        if input_ids.shape[1] < 5:
            input_ids = torch.cat((input_ids, torch.zeros((1, 5-input_ids.shape[1])).long()), dim=1)

        # attention_mask = torch.ones(5)
        attention_mask = torch.ones(input_ids.shape)

        # Process the input_chars, length same as input_ids
        input_chars = torch.zeros((1, 5*26))
        word_lower = word.lower()
        for i in range(5):
            if word_lower[i] != '0':
                input_chars[0, 26*i+ord(word_lower[i])-97] = 1


        # print(input_ids.shape)
        # print(attention_mask.shape)
        # print(input_chars.shape)


        # Combine inputs into a dictionary
        inputs = {
            'input_ids': input_ids.squeeze(0),
            'attention_mask': attention_mask.squeeze(0),
            'input_numbers': torch.tensor([[month, day, weeknum, contest_num]]).float(),
            'input_chars': input_chars.float()
        }

        # # Print inputs and targets shapes
        # print(f"Input IDs shape: {inputs['input_ids'].shape}")
        # print(f"Attention mask shape: {inputs['attention_mask'].shape}")
        # print(f"Input numbers shape: {inputs['input_numbers'].shape}")
        # print(f"Targets shape: {targets.shape}")


        return inputs, targets

# Define function to train model
from tqdm import tqdm

def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    epoch_count = 0

    with tqdm(total=len(dataloader)) as pbar:
        for batch_inputs, batch_targets in dataloader:
            # Move batch to device
            batch_inputs = {key: value.to(device) for key, value in batch_inputs.items()}
            batch_targets = batch_targets.to(device).float()

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            batch_predictions = model(**batch_inputs)
            loss = criterion(batch_predictions, batch_targets)

            # Backward pass and update weights
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # epoch_count += 1
            total_loss_str = '{:.4e}'.format(total_loss)
            pbar.update(1)
            pbar.set_description(f"Batch loss: {total_loss_str}")

            # if epoch_count % 10 == 0:
            #     torch.save(model.state_dict(), "checkpoints/"+"wordle"+"_"+str(epoch_count)+"batches")

    return total_loss / len(dataloader)

# Load data and tokenizer
data = pd.read_csv('Data_V1.2.csv')
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# Resize every word into 5 tokens, if word is longer than 5 tokens, truncate it, if word is shorter than 5 tokens, pad it with 0
data[word_column] = resize_word_list(data[word_column])

# Create dataset and dataloader
dataset = TrainingDataset(data, tokenizer)
dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

# Create model and move to device
model = CustomNetV2('roberta-base', num_numbers=4)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Define loss function and optimizer
# using categorical cross-entropy loss
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)

# # print first 6 rows of dataloader
# print(dataloader.dataset.data.head(6))
# # stack expects each tensor to be equal size, print shape of each tensor in dataloader
# for batch_inputs, batch_targets in dataloader:
#     print(batch_inputs['input_ids'].shape)
#     print(batch_inputs['attention_mask'].shape)
#     print(batch_inputs['input_numbers'].shape)
#     print(batch_targets.shape)
#     break

# drop the first line of the data
# dataloader.dataset.data = dataloader.dataset.data.iloc[2:]

# Train model
num_epochs = 100000
for epoch in range(num_epochs):
    # Train for one epoch and get average loss
    train_loss = train(model, dataloader, optimizer, criterion, device)
    train_loss_str='{:.4e}'.format(train_loss)
    # Print the loss for the epoch
    print(f"Epoch {epoch+1} loss: {train_loss_str}")
    if epoch % 50 == 0:
        torch.save(model.state_dict(), "/mnt/d/checkpoints/"+"wordleV2"+"_"+str(epoch)+"epochs"+train_loss_str)

# Save model
torch.save(model.state_dict(), "checkpoints/"+"wordleV2"+"_"+str(num_epochs)+"final"+train_loss_str)