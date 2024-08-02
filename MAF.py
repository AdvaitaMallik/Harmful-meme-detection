# Install necessary packages
# pip install torch torchvision torchaudio transformers pandas pillow scikit-learn
# git clone https://github.com/shawlyahsan/Bengali-Aggression-Memes

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, BertModel, BertTokenizer
from sklearn.metrics import classification_report

# Device configuration
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class MemeDataset(Dataset):
    def __init__(self, csv_file, img_dir, clip_processor, bert_tokenizer):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.clip_processor = clip_processor
        self.bert_tokenizer = bert_tokenizer
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.data.iloc[idx, 0])
        image = Image.open(img_name).convert("RGB")
        caption = self.data.iloc[idx, 1]
        label = self.data.iloc[idx, 2]

        if label[0] == 'p':
            label = np.array([1, 0, 0, 0, 0])
        elif label[0] == 'r':
            label = np.array([0, 1, 0, 0, 0])
        elif label[0] == 'g':
            label = np.array([0, 0, 1, 0, 0])
        elif label[0] == 'o':
            label = np.array([0, 0, 0, 1, 0])
        else:
            label = np.array([0, 0, 0, 0, 1])
        label = label.astype(np.float64)

        # Preprocess image
        image = self.clip_processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)

        # Preprocess text
        text = self.bert_tokenizer(caption, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
        text_input_ids = text["input_ids"].squeeze(0)
        text_attention_mask = text["attention_mask"].squeeze(0)

        return image, text_input_ids, text_attention_mask, torch.tensor(label, dtype=torch.float)

def get_dataloader(csv_file, img_dir, clip_processor, bert_tokenizer, batch_size):
    dataset = MemeDataset(csv_file, img_dir, clip_processor, bert_tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

class MultiModalAggressionModel(nn.Module):
    def __init__(self):
        super(MultiModalAggressionModel, self).__init__()
        self.bangla_bert_model = bangla_bert_model
        self.clip_model = clip_model
        # Freeze BERT and CLIP model parameters
        for param in self.bangla_bert_model.parameters():
            param.requires_grad = False

        for param in self.clip_model.parameters():
            param.requires_grad = False

        self.text_transform = nn.Linear(768, 512)
        self.attention = nn.MultiheadAttention(embed_dim=512, num_heads=8)
        self.fc = nn.Linear(512 + 512, 5)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, images, input_ids, attention_mask):
        images, input_ids, attention_mask = images.to(device), input_ids.to(device), attention_mask.to(device)
        
        image_features = self.clip_model.get_image_features(images)
        text_outputs = self.bangla_bert_model(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_outputs.last_hidden_state[:, 0, :]
        text_features = self.text_transform(text_features)

        query = text_features.unsqueeze(1)
        key_value = image_features.unsqueeze(1)
        attention_output, _ = self.attention(query, key_value, key_value)

        combined_features = torch.cat((attention_output.squeeze(1), text_features), dim=1)
        logits = self.fc(combined_features)
        output = self.softmax(logits)
        return output

def validate_model(model, val_loader, criterion):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, input_ids, attention_mask, labels in val_loader:
            images, input_ids, attention_mask, labels = images.to(device), input_ids.to(device), attention_mask.to(device), labels.to(device)
            outputs = model(images, input_ids, attention_mask)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)

            _, preds = torch.max(outputs, 1)
            _, labels = torch.max(labels, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        epoch_loss = running_loss / len(val_loader.dataset)
        print(f"Validation Loss: {epoch_loss:.4f}")
        print(classification_report(all_labels, all_preds, target_names=['PAg', 'RAg', 'GAg', 'Oth', 'NoAg']))

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, input_ids, attention_mask, labels in train_loader:
            images, input_ids, attention_mask, labels = images.to(device), input_ids.to(device), attention_mask.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images, input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
        validate_model(model, val_loader, criterion)

# Load pretrained models
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
bangla_bert_model = BertModel.from_pretrained("sagorsarker/bangla-bert-base")
bangla_bert_tokenizer = BertTokenizer.from_pretrained("sagorsarker/bangla-bert-base")

clip_model.to(device)
bangla_bert_model.to(device)

# Main script
if __name__ == "__main__":
    # File paths
    train_csv = './Bengali-Aggression-Memes/Dataset/training_set.csv'
    val_csv = './Bengali-Aggression-Memes/Dataset/validation_set.csv'
    test_csv = './Bengali-Aggression-Memes/Dataset/testing_set.csv'
    img_dir = './Bengali-Aggression-Memes/Dataset/Img/'
    
    # Hyperparameters
    batch_size = 16
    learning_rate = 1e-4
    num_epochs = 5

    # Data loaders
    train_loader = get_dataloader(train_csv, img_dir, clip_processor, bangla_bert_tokenizer, batch_size)
    val_loader = get_dataloader(val_csv, img_dir, clip_processor, bangla_bert_tokenizer, batch_size)
    test_loader = get_dataloader(test_csv, img_dir, clip_processor, bangla_bert_tokenizer, batch_size)

    # Model, criterion, optimizer
    model = MultiModalAggressionModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    # Train the model
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs)

    # Evaluate on test data
    print("Evaluating on test data...")
    validate_model(model, test_loader, criterion)
    try:
        os.mkdir('model')
    except Exception as err:
        print(err)
        raise
    torch.save(model.state_dict(), './model/state_dictionary')
    torch.save(model, './model/entire_model')