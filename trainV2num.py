import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from transformers import RobertaTokenizer
from models.CustomNetV2 import CustomNetV2
from tqdm import tqdm
from utils import resize_word_list, NormalizeData

# Define dataset class
word_column = 'Word'
input_columns = ['Month', 'Day', 'WeekNum', 'Contest number', 'isHoliday']
target_columns = ['Number of reported results', 'Number in hard mode', '1 try', '2 tries', '3 tries', '4 tries', '5 tries', '6 tries', '7 or more tries (X)']
target_columns_percentage = ['1 try', '2 tries', '3 tries', '4 tries', '5 tries', '6 tries', '7 or more tries (X)']
target_columns_no_percentage = ['Number of reported results', 'Number in hard mode']

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
            'input_numbers': torch.tensor([[month, day, weeknum, contest_num, isHoliday]]).float(),
            'input_chars': input_chars.float()
        }

        # # Print inputs and targets shapes
        # print(f"Input IDs shape: {inputs['input_ids'].shape}")
        # print(f"Attention mask shape: {inputs['attention_mask'].shape}")
        # print(f"Input numbers shape: {inputs['input_numbers'].shape}")
        # print(f"Targets shape: {targets.shape}")


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

            # if epoch_count % 10 == 0:
            #     torch.save(model.state_dict(), "checkpoints/"+"wordle"+"_"+str(epoch_count)+"batches")

    return total_loss / len(dataloader)

# Load data and tokenizer
data = pd.read_csv('Data_V1.3.csv')
data_dict = data.to_dict('records')
# tries percentage should be divided by 100
data[target_columns_percentage] = data[target_columns_percentage].div(100)

# Normalize the data, target_columns without percentage
data, means, stds = NormalizeData(data, input_columns, target_columns_no_percentage)
# record the means and stds to a file
with open('means_stds.txt', 'w') as f:
    f.write(str(means) + '\n' + str(stds))

# tuple data to dict, recovery column names from data_dict
# data_new = []
# for i in range(len(data)):
#     data_new.append({**data_dict[i], **data.iloc[i].to_dict()})

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# Resize every word into 5 tokens, if word is longer than 5 tokens, truncate it, if word is shorter than 5 tokens, pad it with 0
data[word_column] = resize_word_list(data[word_column])

# Create dataset and dataloader
dataset = TrainingDataset(data, tokenizer)
dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

# # Normalize the dataloader
# mean_input = torch.mean(torch.cat([batch[0]['input_numbers'] for batch in dataloader]))
# std_input = torch.std(torch.cat([batch[0]['input_numbers'] for batch in dataloader]))
# print(mean_input, std_input)
# mean_output = torch.mean(torch.cat([batch[1] for batch in dataloader]))
# std_output = torch.std(torch.cat([batch[1] for batch in dataloader]))
# print(mean_output, std_output)
# normalizer_input = transforms.Normalize(mean=mean_input, std=std_input)
# normalizer_output = transforms.Normalize(mean=mean_output, std=std_output)
# for batch in dataloader:
#     batch[0]['input_numbers'] = normalizer_input(batch[0]['input_numbers'])
#     batch[1] = normalizer_output(batch[1])

# Create model and move to device
model = CustomNetV2('roberta-base', num_numbers=5)
model.load_state_dict(torch.load("/mnt/d/checkpoints/wordleV2.1_1000epochs 1.7660e-02"))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Define loss function and optimizer
# using categorical cross-entropy loss
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)

# # print first 6 rows of dataloader
print(dataloader.dataset.data.head(6))
# # stack expects each tensor to be equal size, print shape of each tensor in dataloader
# for batch_inputs, batch_targets in dataloader:
#     print(batch_inputs['input_ids'].shape)
#     print(batch_inputs['attention_mask'].shape)
#     print(batch_inputs['input_numbers'].shape)
#     print(batch_targets.shape)
#     break

# drop the first line of the data
# dataloader.dataset.data = dataloader.dataset.data.iloc[2:]

# Save loss to csv
with open('loss.csv', 'w') as f:
    f.write('Epoch, Loss\n')

# Train model
num_epochs = 100000
for epoch in range(num_epochs):
    # Train for one epoch and get average loss
    train_loss = train(model, dataloader, optimizer, criterion, device)
    train_loss_str='{:.4e}'.format(train_loss)
    # Print the loss for the epoch
    print(f"Epoch {epoch+1} loss: {train_loss_str}")
    # Save loss to csv
    with open('loss.csv', 'a') as f:
        f.write(f'{epoch+1}, {train_loss_str}\n')
    if epoch % 50 == 0:
        torch.save(model.state_dict(), "/mnt/d/checkpoints/"+"wordleV2.1"+"_"+str(epoch)+"epochs "+train_loss_str)

# Save model
torch.save(model.state_dict(), "checkpoints/"+"wordleV2.1"+"_"+str(num_epochs)+"final "+train_loss_str)