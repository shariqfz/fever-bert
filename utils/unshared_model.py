import torch
from torch import nn
# from transformers import BertTokenizer, BertModel
from transformers import BertModel
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers import AdamW

class UnsharedModel(nn.Module):
    def __init__(self, model_name_or_path, from_tf, cache_dir):
        super(UnsharedModel, self).__init__()
        self.encoder1 = BertModel.from_pretrained(model_name_or_path, from_tf=from_tf, cache_dir=cache_dir)
        self.encoder2 = BertModel.from_pretrained(model_name_or_path, from_tf=from_tf, cache_dir=cache_dir)
        self.classifier = nn.Sequential(
            nn.Linear(768 * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 3)
        )

    def forward(self, input_ids1, input_ids2, attention_mask1, attention_mask2, token_type_ids1, token_type_ids2, labels):
        # Ensure input tensors are 2D
        if input_ids1.dim() == 1:
            input_ids1 = input_ids1.unsqueeze(0)
        if input_ids2.dim() == 1:
            input_ids2 = input_ids2.unsqueeze(0)
        if attention_mask1.dim() == 1:
            attention_mask1 = attention_mask1.unsqueeze(0)
        if attention_mask2.dim() == 1:
            attention_mask2 = attention_mask2.unsqueeze(0)
        if token_type_ids1.dim() == 1:
            token_type_ids1 = token_type_ids1.unsqueeze(0)
        if token_type_ids2.dim() == 1:
            token_type_ids2 = token_type_ids2.unsqueeze(0)
        
        output1 = self.encoder1(input_ids=input_ids1, token_type_ids=token_type_ids1, attention_mask=attention_mask1)[1]
        output2 = self.encoder2(input_ids=input_ids2, token_type_ids=token_type_ids2, attention_mask=attention_mask2)[1]
        output = torch.cat((output1, output2), dim=1)
        logits = self.classifier(output)
        loss = nn.CrossEntropyLoss()(logits.view(-1, self.classifier[-1].out_features), labels.view(-1))
        return loss, logits

# Load the BERT tokenizer
# tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

# # Assume claims and evidences are pre-defined lists of claims and corresponding evidences
# inputs1 = tokenizer(claims, padding=True, truncation=True, return_tensors="pt")
# inputs2 = tokenizer(evidences, padding=True, truncation=True, return_tensors="pt")
# labels = torch.tensor(labels)

# # Create a DataLoader for our training set
# dataset = TensorDataset(inputs1['input_ids'], inputs1['attention_mask'], inputs2['input_ids'], inputs2['attention_mask'], labels)
# dataloader = DataLoader(dataset, sampler=RandomSampler(dataset), batch_size=32)

# Load the model and specify the optimizer
# model = UnsharedModel()
# optimizer = AdamW(model.parameters(), lr=1e-5)


