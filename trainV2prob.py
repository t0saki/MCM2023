import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from transformers import RobertaTokenizer
from models.CustomNetV2prob import CustomNetV2prob
from tqdm import tqdm
from utils import resize_word_list, NormalizeData, NormalizeDataCustom

# Define dataset class
word_column = 'Word'
input_columns = ['Month', 'Day', 'WeekNum', 'Contest number', 'isHoliday']
target_columns = ['Average', 'Sigma', '1 try', '2 tries', '3 tries', '4 tries', '5 tries', '6 tries', '7 or more tries (X)']
target_columns_percentage = ['1 try', '2 tries', '3 tries', '4 tries', '5 tries', '6 tries', '7 or more tries (X)']
target_columns_no_percentage = ['Average', 'Sigma']

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
        isHoliday = row['isHoliday']
        # targets = row[['Number of reported results', 'Number in hard mode', '1 try', '2 tries', '3 tries', '4 tries', '5 tries', '6 tries', '7 or more tries (X)']].values.astype(float)
        targets = row[target_columns_no_percentage].values.astype(float)

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

        # Combine inputs into a dictionary
        inputs = {
            'input_ids': input_ids.squeeze(0),
            'attention_mask': attention_mask.squeeze(0),
            'input_numbers': torch.tensor([[month, day, weeknum, contest_num, isHoliday]]).float(),
            'input_chars': input_chars.float()
        }

        return inputs, targets

# Define function to train model

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

    return total_loss / len(dataloader)

# Load data and tokenizer
data = pd.read_csv('CustomSet.csv')
data_dict = data.to_dict('records')

# Print first 5 rows of data
print(data.head())
for column in data.columns:
    print(column)

# Read the means and stds from means_stds.txt, 2 lines
with open('means_stds_prob.txt', 'r') as f:
    means_stds = f.read()
means_stds = means_stds.splitlines()
# printed with '[' and ']', so remove them
means_stds[0] = means_stds[0][1:-1]
means_stds[1] = means_stds[1][1:-1]
# split the string into a list
means_stds[0] = means_stds[0].split(', ')
means_stds[1] = means_stds[1].split(', ')
# convert the list of strings to list of floats
means = [float(i) for i in means_stds[0]]
stds = [float(i) for i in means_stds[1]]

# Normalize the data
data = NormalizeDataCustom(data, input_columns, target_columns_no_percentage, means, stds)

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# Resize every word into 5 tokens, if word is longer than 5 tokens, truncate it, if word is shorter than 5 tokens, pad it with 0
data[word_column] = resize_word_list(data[word_column])

# Create dataset and dataloader
dataset = TrainingDataset(data, tokenizer)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Create model and move to device
model = CustomNetV2prob('roberta-base', num_numbers=5)
model.load_state_dict(torch.load("/mnt/d/checkpoints2/probV3_1100epochs 2.7993e-01"))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Define loss function and optimizer
# using categorical cross-entropy loss
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

# # print first 6 rows of dataloader
print(dataloader.dataset.data.head(6))

# Save loss to csv
with open('lossprob.csv', 'w') as f:
    f.write('Epoch, Loss\n')

# Train model
num_epochs = 1000000
for epoch in range(num_epochs):
    # Train for one epoch and get average loss
    train_loss = train(model, dataloader, optimizer, criterion, device)
    train_loss_str='{:.4e}'.format(train_loss)
    # Print the loss for the epoch
    print(f"Epoch {epoch+1} loss: {train_loss_str}")
    # Save loss to csv
    with open('lossprob.csv', 'a') as f:
        f.write(f'{epoch+1}, {train_loss_str}\n')
    if epoch % 50 == 0:
        torch.save(model.state_dict(), "/mnt/d/checkpoints2/"+"probV3"+"_"+str(epoch)+"epochs "+train_loss_str)

# Save model
torch.save(model.state_dict(), "checkpoints/"+"probV3"+"_"+str(num_epochs)+"final "+train_loss_str)