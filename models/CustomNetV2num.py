import torch
import torch.nn as nn
import transformers

class CustomNetV2num(nn.Module):
    def __init__(self, roberta_model_name, num_numbers):
        super(CustomNetV2num, self).__init__()

        # Load pre-trained RoBERTa model and tokenizer
        self.roberta = transformers.RobertaModel.from_pretrained(roberta_model_name)
        self.tokenizer = transformers.RobertaTokenizer.from_pretrained(roberta_model_name)

        # Freeze RoBERTa weights
        for param in self.roberta.parameters():
            param.requires_grad = False

        # Define FCN for number input
        self.numbers_fcn = nn.Sequential(
            nn.Linear(num_numbers, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # Define FCN for 5 character input
        self.chars_fcn = nn.Sequential(
            nn.Linear(5*26, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # Define combined FCN
        self.combined_fcn = nn.Sequential(
            nn.Linear(self.roberta.config.hidden_size + 64*2, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
            nn.Dropout(0.2)
        )

    def forward(self, input_ids, attention_mask, input_numbers, input_chars):
        # Get RoBERTa output
        roberta_output = self.roberta(input_ids, attention_mask=attention_mask)[0]
        pooled_output = roberta_output[:, 0, :]  # extract CLS token

        # Get number input output
        numbers_output = self.numbers_fcn(input_numbers)
        numbers_output = numbers_output.squeeze(1)

        # Get character input output
        chars_output = self.chars_fcn(input_chars)
        chars_output = chars_output.squeeze(1)

        # print(pooled_output.shape)
        # print(numbers_output.shape)
        # print(chars_output.shape)
        # print(input_chars.shape)

        # Concatenate RoBERTa and number outputs
        combined = torch.cat((pooled_output, numbers_output, chars_output), dim=1)

        # Get combined output
        output = self.combined_fcn(combined)

        # Split output into two numbers and 7 probabilities that sum to 1
        # output_nums, output_probs = output[:, :2], output[:, 2:]
        # output_probs = nn.functional.softmax(output_probs, dim=1)

        # Return final output
        # return torch.cat((output_nums, output_probs), dim=1)
        return output