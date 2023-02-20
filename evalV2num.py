import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer
from models.CustomNetV2num import CustomNetV2num
from tqdm import tqdm
from utils import resize_word_list, NormalizeData, DenormalizeData, NormalizeDataWithMeansStds

# Define dataset class
word_column = 'Word'
input_columns = ['Month', 'Day', 'WeekNum', 'Contest number', 'isHoliday']
target_columns = ['Number of reported results', 'Number in hard mode', '1 try', '2 tries', '3 tries', '4 tries', '5 tries', '6 tries', '7 or more tries (X)']
target_columns_percentage = ['1 try', '2 tries', '3 tries', '4 tries', '5 tries', '6 tries', '7 or more tries (X)']
target_columns_no_percentage = ['Number of reported results', 'Number in hard mode']
class EvaluationDataset(Dataset):
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

# Define the evaluation function
def evaluate(model, data_loader, criterion, device):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    outputs_list = []
    with torch.no_grad():
        for batch_inputs, batch_targets in data_loader:
            # Move batch to device
            batch_inputs = {key: value.to(device) for key, value in batch_inputs.items()}
            batch_targets = batch_targets.to(device).float()

            outputs = model(**batch_inputs)

            # from torchviz import make_dot
            # # from models.CustomNetV2num import CustomNetV2num
            # dot = make_dot(outputs, params=dict(model.named_parameters()))
            # dot.render(filename='custom_net_v2_num', format='png')
            # exit()

            outputs_list.append(outputs)
            loss = criterion(outputs, batch_targets)
            # acc = binary_accuracy(outputs, targets)

            epoch_loss += loss.item()
            # epoch_acc += acc.item()

    return epoch_loss / len(data_loader), outputs_list

# Load the data
data = pd.read_csv('CustomSet.csv')
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# Normalize the data
data[target_columns_percentage] = data[target_columns_percentage].div(100)

# Read the means and stds from means_stds.txt, 2 lines
with open('means_stds_num.txt', 'r') as f:
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

data = NormalizeDataWithMeansStds(data, input_columns, target_columns_no_percentage, means, stds)

# Load the model
model = CustomNetV2num('roberta-base', num_numbers=5)
model.load_state_dict(torch.load("/mnt/d/checkpoints2/numV3_1100epochs 1.5926e-01"))

# Define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Define the data loader
dataset = EvaluationDataset(data, tokenizer)
data_loader = DataLoader(dataset, batch_size=16, shuffle=False)

# Define the loss function
criterion = torch.nn.MSELoss()

# Evaluate the model
loss, outputs_list = evaluate(model, data_loader, criterion, device)
print(f'| Loss: {loss:.4f} |')

# Save the outputs to csv
outputs = torch.cat(outputs_list, dim=0)
outputs = outputs.cpu().numpy()
outputs = pd.DataFrame(outputs, columns=target_columns_no_percentage)

# Write raw outputs to csv
outputs.to_csv('raw_outputs_num.csv', index=False)

# Denormalize the data
for col in target_columns_no_percentage:
    index = target_columns_no_percentage.index(col)
    outputs.iloc[:, index] = DenormalizeData(outputs.iloc[:, index], means[len(input_columns)+index], stds[len(input_columns)+index])

print(outputs)

outputs.to_csv('outputs_num.csv', index=False)