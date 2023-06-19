##############################################################################
"""
Import modules
"""
##############################################################################

from transformers import T5Tokenizer, T5EncoderModel
import torch
import re
import os
import torch
import nltk
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.optim as optim
import pandas as pd
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, multilabel_confusion_matrix
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
import json as js
import seaborn as sns
# from torch.utils.tensorboard import SummaryWriter
nltk.download('punkt')



##############################################################################
"""
Initalize parameters and train
"""
##############################################################################  

def main():
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    df,  label_vocab = read_data()  
    
    vocab = [' '.join(string) for string in df['Sequence']]
    vocab = list(set(nltk.word_tokenize(' '.join(vocab))))
    unique_counts = df['Activity'].value_counts()

    cfg = {
        'vocab':vocab,
        'label_vocab':label_vocab,
        'n_classes': len(label_vocab),
        'vocab_size': len(vocab),  # 21, including padding
        'max_seq_len': 102,
        'random_state':42,
        
        'd_model': 1024,
        'num_layers': 1,
        'nhead': 8,
        'd_ff': 1024,
        'dropout_rate': 0.1,
        
        'num_epochs': 301,
        'batch_size': 512,
        'num_workers': 2,
        'device':device,
        
        'lr': 0.001,
        'scheduler_factor': 0.5,
        'patience': 50,
        'cooldown':50,
        
        'checkpointing':False
    }
    # print(df)
    print(label_vocab)
    print(len(label_vocab))
    # print(unique_counts)

    run_train(df, cfg)


##############################################################################
"""
Initalize prot_t5
"""
##############################################################################    

def get_prot5(cfg):
    transformer_link = "Rostlab/prot_t5_xl_half_uniref50-enc"
    print("Loading: {}".format(transformer_link))
    model = T5EncoderModel.from_pretrained(transformer_link)#, local_files_only=True, cache_dir='//scratch/mlsample/apm598_project/cache_dir')
    model.full() if cfg['device']=='cpu' else model.half() # only cast to full-precision if no GPU is available
    model = model.to(cfg['device'])
    model.eval()
    tokenizer = T5Tokenizer.from_pretrained(transformer_link, do_lower_case=False)#, local_files_only=True, cache_dir='//scratch/mlsample/apm598_project/cache_dir' )
    return model, tokenizer

##############################################################################
"""
Read Data into Pandas Dataframe
"""
##############################################################################

def read_data():
    patent_path = '//scratch/mlsample/apm598_project/data/patent_amps.csv'
    general_path = '//scratch/mlsample/apm598_project/data/general_amps.csv'
    clinical_path = '//scratch/mlsample/apm598_project/data/clinical_amps.csv'
    specific_path = '//scratch/mlsample/apm598_project/data/specific_amps.csv'
    
    df_pat = pd.read_csv(patent_path)
    df_gen = pd.read_csv(general_path)
    df_cln = pd.read_csv(clinical_path)
    df_spc = pd.read_csv(specific_path)
    
    sources = [df_gen, df_cln, df_spc]
    
    intersection_gen = [list(df_pat.columns.intersection(source.columns)) for source in sources]
    
    
    df = pd.concat([df_pat[intersection_gen[1]], df_gen[intersection_gen[1]], df_cln[intersection_gen[1]], df_spc[intersection_gen[1]]])
    
    df = df.dropna() 
    df = df.drop_duplicates(subset=['Sequence'])
    df = df[df.Sequence.str.isalpha()]
    
    df['Activity'] = [string.lower() for string in df['Activity']]
    df['Activity'] = df['Activity'].replace({
        'anitibacterial':'antibacterial',
        'anitifungal':'antifungal',
        'antbacterial':'antibacterial',
        'anticancer':'anti-cancer',
        'antifunga':'antifungal',
        'cytotoxic': 'cytotoxicity',
        'wound healing':'wound-healing',
        'antifungall':'antifungal'
    }, regex=True)
    
    df['Activity'] = df['Activity'].replace({'antifungall':'antifungal'}, regex=True)
    
    df = df.reset_index(drop=True)
    
    unique_counts = df['Activity'].value_counts()
    mask = df['Activity'].apply(lambda x: unique_counts[x] >= 100)
    df = df[mask]

    label_vocab = df['Activity'].unique()
    
    ocab = [string.split(',') for string in label_vocab]
    ocab = sum(ocab, [])
    ocab = [string.strip() for string in ocab]
    ocab_set = set(ocab)
    
    label_vocab = df['Activity'].unique()
    
    
    ocab = [string.split(',') for string in label_vocab]
    ocab = sum(ocab, [])
    ocab = [string.strip() for string in ocab]
    label_vocab = list(set(ocab))
    
    df = df.reset_index(drop=True)
        
    return df, label_vocab


##############################################################################
"""
Define Dataset class
"""
##############################################################################

class AntimicrobialPeptideDataSet(Dataset):
    def __init__(self, data, labels, vocab, label_vocab):
        self.data = data
        self.labels = labels
        self.vocab = vocab
        self.label_vocab = label_vocab
        
        self.x = data
        self.y = torch.Tensor(self.one_hot_encode_labels(self.labels, label_vocab)).float()

        # Create a dictionary to map labels to indices
#         label_to_index = {label: idx for idx, label in enumerate(np.unique(label_vocab))}

#         # Convert the labels to class indices
#         class_indices = np.array([label_to_index[label] for label in labels])

        # Convert the class indices to a PyTorch tensor
        # self.y  = torch.Tensor(class_indices).long()
    
    def __getitem__(self, idx):
        len_X = max([[len(seq) for seq in [self.x[idx]]]])
        return self.x[idx], self.y[idx], len_X
    
    def __len__(self):
        return len(self.data)
    
    def one_hot_encode_labels(self, labels, label_vocab):
        label_to_idx = {word: i for i, word in enumerate(label_vocab)}
        labels = [label.strip().split(',') for label in labels]
        n_classes = len(label_vocab)
        label_holder = [[0 for _ in range(n_classes)] for _ in range(len(labels))]
        
        for idx, label in enumerate(labels):
            for activity in label:
                label_holder[idx][label_to_idx[activity.strip()]] = 1
        
        return label_holder
    
##############################################################################
"""
Define model
"""
##############################################################################

    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead, attn_mask=None):
        super(MultiHeadAttention, self).__init__()
        self.nhead = nhead
        self.attention = nn.MultiheadAttention(d_model, nhead)

    def forward(self, q, k, v, attn_mask=None):
        return self.attention(q, k, v, key_padding_mask=attn_mask)[0]


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, d_ff, dropout_rate):
        super(EncoderLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(d_model, nhead)
        self.position_wise_feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate)


    def forward(self, x, attn_mask):
        attn_output = self.multi_head_attention(x, x, x, attn_mask=attn_mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.position_wise_feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


class ClassificationTransformer(nn.Module):
    def __init__(self, prot_model, tokenizer, max_seq_len, d_model, nhead, device, n_classes, dropout_rate, num_layers, d_ff):
        super(ClassificationTransformer, self).__init__()
        self.max_seq_len = max_seq_len
        self.device = device
        self.tokenizer = tokenizer
        self.model = prot_model

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, nhead, d_ff, dropout_rate) for _ in range(num_layers)])
        self.fc = nn.Linear(1 * d_model, n_classes)

    def forward(self, x, len_x):
        x, attention_mask = self.prot_embeddings(x)
        x = x.to(torch.float32)
        attention_mask = attention_mask.to(torch.float32).permute(1,0)        
        for layer in self.encoder_layers:
            x = layer(x, attention_mask)

        x = torch.stack([sequence[:mySeq_len].mean(dim=0) for sequence, mySeq_len in zip(x, len_x)])
        # x = self.pool_strategy(x, len_x)
        
        x = self.fc(x)
        return x

    
    def prot_embeddings(self, x):
        sequence_examples = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in x]

        # tokenize sequences and pad up to the longest sequence in the batch
        ids = self.tokenizer.batch_encode_plus(sequence_examples, add_special_tokens=True, max_length=self.max_seq_len, padding="max_length")
        input_ids = torch.tensor(ids['input_ids']).to(self.device)
        attention_mask = torch.tensor(ids['attention_mask']).to(self.device)

        # generate embeddings
        with torch.no_grad():
            embedding_repr = self.model(input_ids=input_ids,attention_mask=attention_mask)
    
            # extract embeddings for the first ([0,:]) sequence in the batch while removing padded & special tokens ([0,:7]) 
            emb_0 = embedding_repr.last_hidden_state

        return emb_0, attention_mask
    
##############################################################################
"""
Training Loop
"""
##############################################################################

def save_checkpoint(model, optimizer, epoch, filename):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)

def load_checkpoint(model, optimizer, filename):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return model, optimizer,epoch
    
##############################################################################
"""
Initalize Dataloader and dataset
"""
##############################################################################

def load_data(df, cfg):
    X_train, X_test, y_train, y_test = train_test_split(list(df["Sequence"]), df['Activity'], test_size=0.2, random_state=cfg['random_state'], stratify=df['Activity'])
    # print(f"load_data: {y_train = }")
    train_dataset = AntimicrobialPeptideDataSet(X_train, y_train, cfg['vocab'], cfg['label_vocab'])
    test_dataset = AntimicrobialPeptideDataSet(X_test, y_test, cfg['vocab'], cfg['label_vocab'])
    
    
    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg['batch_size'], num_workers=cfg['num_workers'])#, sampler=train_sampler)
    
    # test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg['batch_size'], num_workers=cfg['num_workers'])#, sampler=train_sampler)
    return train_dataloader, test_dataloader

##############################################################################
"""
Train
"""
##############################################################################

def run_train(df, cfg):    
    train_dataloader, test_dataloader = load_data(df, cfg)
    
    model, tokenizer = get_prot5(cfg)
    
    myModel = ClassificationTransformer(
        model, tokenizer, cfg['max_seq_len'], cfg['d_model'], cfg['nhead'], cfg['device'],
        cfg['n_classes'], cfg['dropout_rate'], cfg['num_layers'], cfg['d_ff']
    ).to(cfg['device'])

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(myModel.parameters(), lr=cfg['lr'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=cfg['scheduler_factor'],
                                  patience=cfg['patience'], cooldown=cfg['cooldown'], verbose=True)
    
    if cfg['checkpointing']:
        checkpoint_path = 'myModel_checkpoint.pth'
        if os.path.isfile(checkpoint_path):
            myModel, optimizer, cfg['starting_epoch'] = load_checkpoint(myModel, optimizer, checkpoint_path)
            print(f"Loaded checkpoint and resuming from epoch {cfg['starting_epoch']}")
        else:
            cfg['starting_epoch'] = 0
            print("No checkpoint found, starting training from scratch")
    else:
        cfg['starting_epoch'] = 0
        
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []
    
    train_precisions = []
    train_recalls = []
    train_f1_scores = []
    
    test_precisions = []
    test_recalls = []
    test_f1_scores = []

    for epoch in range(cfg['starting_epoch'], cfg['num_epochs']):
        myModel.train()
        running_train_loss = 0.0
        running_train_corrects = 0
        total_train_samples = 0

        train_predictions = []
        train_true_labels = []
        
        loop_train = tqdm(train_dataloader, desc=f"Epoch [{epoch+1}/{cfg['num_epochs']}]")

        for inputs, targets, len_X in loop_train:
            inputs = inputs
            targets = targets.to(cfg['device'])

            optimizer.zero_grad()
            outputs = myModel(inputs, len_X[0])
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            scheduler.step(loss)
            
            preds = torch.sigmoid(outputs).round()
            running_train_loss += loss.item() * len(inputs)
            running_train_corrects += (preds == targets).float().mean().item() * len(inputs)
            total_train_samples += len(inputs)
            
            train_predictions.extend(preds.cpu().detach().numpy())
            train_true_labels.extend(targets.cpu().detach().numpy())

        train_acc = accuracy_score(train_true_labels, train_predictions)
        train_loss = running_train_loss / total_train_samples
        train_accuracy = running_train_corrects / total_train_samples
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        
        print(train_accuracy)
        train_precision, train_recall, train_f1, _ = precision_recall_fscore_support(train_true_labels, train_predictions, average='samples')

        train_precisions.append(train_precision)
        train_recalls.append(train_recall)
        train_f1_scores.append(train_f1)

        myModel.eval()
        running_test_loss = 0.0
        running_test_corrects = 0
        total_test_samples = 0

        test_predictions = []
        test_true_labels = []

        loop_test = tqdm(test_dataloader, desc=f"Epoch [{epoch+1}/{cfg['num_epochs']}]")

        with torch.no_grad():
            for inputs, targets, len_X in loop_test:
                inputs = inputs
                targets = targets.to(cfg['device'])

                outputs = myModel(inputs, len_X[0])
                loss = criterion(outputs, targets)

                running_test_loss += loss.item() * len(inputs)
                preds = torch.sigmoid(outputs).round()
                running_test_corrects += (preds == targets).float().mean().item() * len(inputs)
                total_test_samples += len(inputs)

                test_predictions.extend(preds.cpu().detach().numpy())
                test_true_labels.extend(targets.cpu().detach().numpy())
        
        test_acc = accuracy_score(test_true_labels, test_predictions)
        
        test_loss = running_test_loss / total_test_samples
        test_accuracy = running_test_corrects / total_test_samples
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)
        
        print(test_accuracy)
        test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(test_true_labels, test_predictions, average='samples')
                
        test_precisions.append(test_precision)
        test_recalls.append(test_recall)
        test_f1_scores.append(test_f1) 
        test_mcm = multilabel_confusion_matrix(test_true_labels, test_predictions)

        print(f'Epoch {epoch + 1}/{cfg["num_epochs"]}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}')
        
        if (epoch + 1) % 5 == 0:
            save_checkpoint(myModel, optimizer, epoch, 'myModel_checkpoint.pth')
            
            stats = {
                "train_losses": train_losses,
                "train_accuracies": train_accuracies,
                "test_losses": test_losses,
                "test_accuracies": test_accuracies,
                "train_precisions": train_precisions,
                "train_recalls": train_recalls,
                "train_f1_scores": train_f1_scores,
                "test_precisions": test_precisions,
                "test_recalls": test_recalls,
                "test_f1_scores": test_f1_scores,
                "test_predictions": [pred.tolist() for pred in test_predictions],
                'test_true_labels': [true.tolist() for true in test_true_labels],
            }
            
            # Save the stats to a JSON file
            with open("stats.json", "w") as f:
                js.dump(stats, f)
    
            # Plot loss, accuracy, precision, recall, and F1-score versus epoch
            fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10), (ax11, ax12, ax13, ax14, ax15), (ax16, ax17, ax18, ax19, ax20)) = plt.subplots(4, 5, figsize=(22, 20))
            
            # Loss plot
            ax1.plot(range(1, epoch + 2), train_losses, label='Train')
            ax1.plot(range(1, epoch + 2), test_losses, label='Test')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.set_title('Loss')
            ax1.legend()
            
            # Accuracy plot
            ax2.plot(range(1, epoch + 2), train_accuracies, label='Train')
            ax2.plot(range(1, epoch + 2), test_accuracies, label='Test')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy')
            ax2.set_title('Accuracy')
            ax2.legend()
            
            # Precision plot
            ax3.plot(range(1, epoch + 2), train_precisions, label='Train')
            ax3.plot(range(1, epoch + 2), test_precisions, label='Test')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Precision')
            ax3.set_title('Precision')
            ax3.legend()
            
            # Recall plot
            ax4.plot(range(1, epoch + 2), train_recalls, label='Train')
            ax4.plot(range(1, epoch + 2), test_recalls, label='Test')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Recall')
            ax4.set_title('Recall')
            ax4.legend()
            
            # F1-score plot
            ax5.plot(range(1, epoch + 2), train_f1_scores, label='Train')
            ax5.plot(range(1, epoch + 2), test_f1_scores, label='Test')
            ax5.set_xlabel('Epoch')
            ax5.set_ylabel('F1 Score')
            ax5.set_title('F1 Score')
            ax5.legend()
            
            # Bar chart for final values of each metric
            metrics = ['Loss', 'Accuracy', 'Precision', 'Recall', 'F1 Score']
            train_values = [train_losses[-1], train_accuracies[-1], train_precisions[-1], train_recalls[-1], train_f1_scores[-1]]
            test_values = [test_losses[-1], test_accuracies[-1], test_precisions[-1], test_recalls[-1], test_f1_scores[-1]]
            bar_width = 0.3
            x_pos = np.arange(len(metrics))
            
            ax6.bar(x_pos - bar_width / 2, train_values, bar_width, label='Train')
            ax6.bar(x_pos + bar_width / 2, test_values, bar_width, label='Test')
            ax6.set_xlabel('Metrics')
            ax6.set_ylabel('Value')
            ax6.set_title('Final Metric Values')
            ax6.set_xticks(x_pos)
            ax6.set_xticklabels(metrics)
            ax6.legend()
            
            class_labels = ['antiviral', 'antibacterial', 'insecticidal', 'anti-gram+', 'anti-gram-', 'antitumor', 'anti-cancer', 'putatively-antimicrobial', 'not found', 'antimicrobial', 'antifungal']
            test_mcm = multilabel_confusion_matrix(test_true_labels, test_predictions)
            
            for idx, (matrix, label) in enumerate(zip(test_mcm, class_labels)):
                row, col = divmod(idx, 5)
                ax = locals()[f'ax{idx + 6}']
                sns.heatmap(matrix, annot=True, fmt='d', cmap='coolwarm', ax=ax, cbar=False)
                ax.set_title(f'{label}')
                ax.set_xlabel('Predicted')
                ax.set_ylabel('True')
            
            # Hide unused subplots
            for idx in range(16, 21):
                ax = locals()[f'ax{idx}']
                ax.axis('off')
            
            plt.tight_layout()
            plt.savefig('stats.png', dpi=300)
            plt.close()




# def run_train(df, cfg):

#     # Load the model checkpoint (if available) 
#     # dist.init_process_group("gloo", init_method='env://')
#     # local_rank = torch.distributed.get_rank()
#     # torch.cuda.set_device(local_rank)
    
#     train_dataloader, test_dataloader = load_data(df, cfg)
    
#     model, tokenizer = get_prot5(cfg)
    
#     myModel = ClassificationTransformer(
#         model, tokenizer, cfg['max_seq_len'], cfg['d_model'], cfg['nhead'], cfg['device'],
#         cfg['n_classes'], cfg['dropout_rate'], cfg['num_layers'], cfg['d_ff']
#     ).to(cfg['device'])
#     # myModel = DDP(myModel, device_ids=[local_rank], output_device=local_rank)
    
    
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(myModel.parameters(), lr=cfg['lr'])
#     scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=cfg['scheduler_factor'],
#                                   patience=cfg['patience'], cooldown=cfg['cooldown'], verbose=True)
    
#     if cfg['checkpointing']:
#         checkpoint_path = 'myModel_checkpoint.pth'
#         if os.path.isfile(checkpoint_path):
#             myModel, optimizer, cfg['starting_epoch'] = load_checkpoint(myModel, optimizer, checkpoint_path)
#             print(f"Loaded checkpoint and resuming from epoch {cfg['starting_epoch']}")
#         else:
#             cfg['starting_epoch'] = 0
#             print("No checkpoint found, starting training from scratch")
#     else:
#         cfg['starting_epoch'] = 0
        
        
#     train_losses = []
#     train_accuracies = []
#     test_losses = []
#     test_accuracies = []

#     for epoch in range(cfg['starting_epoch'], cfg['num_epochs']):
#         myModel.train()
#         running_train_loss = 0.0
#         running_train_corrects = 0
#         total_train_samples = 0

#         # tqdm progress bar for training
#         loop_train = tqdm(train_dataloader, desc=f"Epoch [{epoch+1}/{cfg['num_epochs']}]")

#         for inputs, targets, len_X in loop_train:
#             inputs = inputs
#             targets = targets.to(cfg['device'])
            
#             optimizer.zero_grad()  # Zero the gradients
#             outputs = myModel(inputs, len_X[0])  # Forward pass
#             print(outputs.shape)
#             loss = criterion(outputs, targets)  # Calculate loss
#             loss.backward()  # Backward pass
#             optimizer.step()  # Optimization step
#             scheduler.step(loss)
            
#             preds = torch.nn.functional.softmax(outputs, dim=1).round()
#             print(preds)
#             running_train_loss += loss.item() * len(inputs)
#             # _, preds = torch.max(outputs, 1)  # Get predicted class indices
#             running_train_corrects += (preds == targets).sum().item()  # Count correct predictions
#             total_train_samples += len(inputs)

#         train_loss = running_train_loss / total_train_samples
#         train_accuracy = running_train_corrects / total_train_samples
#         train_losses.append(train_loss)
#         train_accuracies.append(train_accuracy)

#         myModel.eval()
#         running_test_loss = 0.0
#         running_test_corrects = 0
#         total_test_samples = 0
        
#         predictions = []
#         true_labels = []

#         # tqdm progress bar for testing
#         loop_test = tqdm(test_dataloader, desc=f"Epoch [{epoch+1}/{cfg['num_epochs']}]")

#         with torch.no_grad():
#             for inputs, targets, len_X in loop_test:
#                 inputs = inputs
#                 targets = targets.to(cfg['device'])

#                 outputs = myModel(inputs, len_X[0])  # Forward pass
#                 loss = criterion(outputs, targets)  # Calculate loss

#                 running_test_loss += loss.item() * len(inputs)
#                 # _, preds = torch.max(outputs, 1)  # Get predicted class indices
#                 preds = torch.nn.functional.softmax(outputs, dim=1).round()
#                 running_test_corrects += (preds == targets).sum().item()  # Count correct predictions
#                 total_test_samples += len(inputs)

#                 predictions.extend(preds.cpu().numpy())
#                 true_labels.extend(targets.cpu().numpy())

#                 # Compute accuracy
#                 acc = accuracy_score(true_labels, predictions)
#                 precision,recall,fbeta,support = precision_recall_fscore_support(true_labels, predictions)
                
#         test_loss = running_test_loss / total_test_samples
#         test_accuracy = running_test_corrects / total_test_samples
#         test_losses.append(test_loss)
#         test_accuracies.append(test_accuracy)

#         print(f'Epoch {epoch + 1}/{cfg["num_epochs"]}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')
        
#         # Save the model checkpoint
#         # save_checkpoint(myModel, optimizer, epoch, 'myModel_checkpoint.pth')
        
#         with open('stats.json', 'w') as f:
#             js.dump({
#                 'train_losses': f'{train_losses}',
#                 'test_losses': f'{test_losses}',
#                 'train_accuracies': f'{train_accuracies}',
#                 'test_accuracies': f'{test_accuracies}',
#                 'predictions': f'{predictions}',
#                 'true_labels': f'{true_labels}'
#             }, f)

#     # Plot loss and accuracy versus epoch
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
#     ax1.plot(range(1, cfg['num_epochs'] + 1), train_losses, label='Train Loss')
#     ax1.plot(range(1, cfg['num_epochs'] + 1), test_losses, label='Test Loss')
#     ax1.set_xlabel('Epoch')
#     ax1.set_ylabel('Loss')
#     ax1.legend()

#     ax2.plot(range(1, cfg['num_epochs'] + 1), train_accuracies, label='Train Accuracy')
#     ax2.plot(range(1, cfg['num_epochs'] + 1), test_accuracies, label='Test Accuracy')
#     ax2.set_xlabel('Epoch')
#     ax2.set_ylabel('Loss')
#     ax2.legend()
    
#     plt.savefig('stats.png', dpi=300)

    
if __name__ == "__main__":
    main()