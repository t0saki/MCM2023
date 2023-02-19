import torch
import torch.nn as nn
import transformers

class CustomNet(nn.Module):
    def __init__(self, roberta_model_name, num_numbers):
        super(CustomNet, self).__init__()

        # Load pre-trained RoBERTa model and tokenizer
        self.roberta = transformers.RobertaModel.from_pretrained(roberta_model_name)
        self.tokenizer = transformers.RobertaTokenizer.from_pretrained(roberta_model_name)

        # Freeze RoBERTa weights
        for param in self.roberta.parameters():
            param.requires_grad = False

        # Define FCN for number input
        self.numbers_fcn = nn.Sequential(
            nn.Linear(num_numbers, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # Define combined FCN
        self.combined_fcn = nn.Sequential(
            nn.Linear(self.roberta.config.hidden_size + 32, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 9)
        )

    def forward(self, input_ids, attention_mask, input_numbers):
        # Get RoBERTa output
        roberta_output = self.roberta(input_ids, attention_mask=attention_mask)[0]
        pooled_output = roberta_output[:, 0, :]  # extract CLS token

        # Get number input output
        numbers_output = self.numbers_fcn(input_numbers)
        numbers_output = numbers_output.squeeze(1)

        # Concatenate RoBERTa and number outputs
        # print(pooled_output.shape)
        # print(numbers_output.shape)
        combined = torch.cat((pooled_output, numbers_output), dim=1)

        # Get combined output
        output = self.combined_fcn(combined)

        # Split output into two numbers and 7 probabilities that sum to 1
        output_nums, output_probs = output[:, :2], output[:, 2:]
        output_probs = nn.functional.softmax(output_probs, dim=1)

        # Return final output
        return torch.cat((output_nums, output_probs), dim=1)