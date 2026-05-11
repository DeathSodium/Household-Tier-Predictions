"""
MIL Attention Network - ROBUST VERSION
Tier: Shallow Neural Network
Improvements: Mini-batching, Padding, Dropout, and strict Label alignment.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import numpy as np
import joblib, os, sys
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report

class RobustBagDataset(Dataset):
    def __init__(self, df, num_cols, cat_cols, bag_ids, is_train=True):
        self.bag_ids = bag_ids
        self.is_train = is_train
        self.bags = []
        
        grouped = df.groupby('bag_id')
        label_map = {'lower': 0, 'middle': 1, 'upper': 2}
        
        for bid in bag_ids:
            group = grouped.get_group(bid)
            nums = torch.FloatTensor(group[num_cols].values)
            cats = torch.LongTensor(group[cat_cols].values)
            
            if is_train:
                label = label_map[group['label'].iloc[0]]
                self.bags.append((nums, cats, torch.LongTensor([label])))
            else:
                self.bags.append((nums, cats, bid))

    def __len__(self):
        return len(self.bags)

    def __getitem__(self, idx):
        return self.bags[idx]

def collate_mil(batch):
    # Batch size: B
    # Each item: (nums, cats, label/bid)
    nums_list = [item[0] for item in batch]
    cats_list = [item[1] for item in batch]
    
    # Pad to max size in batch
    nums_padded = pad_sequence(nums_list, batch_first=True) # [B, MaxN, num_nums]
    cats_padded = pad_sequence(cats_list, batch_first=True) # [B, MaxN, num_cats]
    
    # Create mask for attention (since some bags are shorter)
    lengths = torch.LongTensor([len(n) for n in nums_list])
    mask = torch.zeros(nums_padded.shape[:2])
    for i, l in enumerate(lengths):
        mask[i, :l] = 1
        
    labels = torch.cat([item[2].unsqueeze(0) if torch.is_tensor(item[2]) else torch.zeros(1) for item in batch])
    
    if isinstance(batch[0][2], (str, int, np.int64)):
        # Test mode
        return nums_padded, cats_padded, mask, [item[2] for item in batch]
    return nums_padded, cats_padded, mask, labels.flatten()

class RobustAttentionMIL(nn.Module):
    def __init__(self, num_feats, cat_sizes, emb_dim=8, hidden_dim=64):
        super(RobustAttentionMIL, self).__init__()
        self.embs = nn.ModuleList([nn.Embedding(size+1, emb_dim) for size in cat_sizes])
        input_dim = num_feats + (len(cat_sizes) * emb_dim)
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
        
        self.classifier = nn.Linear(hidden_dim, 3)

    def forward(self, nums, cats, mask):
        # nums: [B, N, num_nums], cats: [B, N, num_cats], mask: [B, N]
        B, N, _ = nums.shape
        
        embeddings = [self.embs[i](cats[:, :, i]) for i in range(len(self.embs))]
        x = torch.cat([nums] + embeddings, dim=2) # [B, N, input_dim]
        
        h = self.feature_extractor(x) # [B, N, hidden_dim]
        
        # Attention
        a = self.attention(h).squeeze(-1) # [B, N]
        a = a.masked_fill(mask == 0, -1e9)
        a = torch.softmax(a, dim=1) # [B, N]
        
        # Bag representation
        bag_h = torch.bmm(a.unsqueeze(1), h).squeeze(1) # [B, hidden_dim]
        
        logits = self.classifier(bag_h)
        return logits

def train_mil():
    print("="*60, flush=True)
    print("ROBUST MIL ATTENTION TRAINING", flush=True)
    print("="*60, flush=True)
    
    device = torch.device("cpu")
    data_res = joblib.load('data/processed/train_instances.joblib')
    df = data_res['data']
    num_cols = data_res['num_cols']
    cat_cols = data_res['cat_cols']
    cat_sizes = [df[c].max() + 1 for c in cat_cols]
    
    label_map = {'lower': 0, 'middle': 1, 'upper': 2}
    bag_info = df.groupby('bag_id')['label'].first().map(label_map).reset_index()
    bag_ids = bag_info['bag_id'].values
    bag_labels = bag_info['label'].values
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(bag_ids))
    
    for fold, (tr_idx, va_idx) in enumerate(skf.split(bag_ids, bag_labels)):
        print(f"\nFold {fold+1}", flush=True)
        train_ds = RobustBagDataset(df, num_cols, cat_cols, bag_ids[tr_idx])
        val_ds = RobustBagDataset(df, num_cols, cat_cols, bag_ids[va_idx])
        
        train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=collate_mil)
        val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, collate_fn=collate_mil)
        
        model = RobustAttentionMIL(len(num_cols), cat_sizes).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        best_val_f1 = 0
        for epoch in range(25):
            model.train()
            total_loss = 0
            for b_nums, b_cats, b_mask, b_labels in train_loader:
                optimizer.zero_grad()
                logits = model(b_nums, b_cats, b_mask)
                loss = criterion(logits, b_labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            model.eval()
            preds, targets = [], []
            with torch.no_grad():
                for b_nums, b_cats, b_mask, b_labels in val_loader:
                    logits = model(b_nums, b_cats, b_mask)
                    preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                    targets.extend(b_labels.cpu().numpy())
            
            val_f1 = f1_score(targets, preds, average='macro')
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save(model.state_dict(), f'models/v2/robust_mil_fold{fold}.pt')
            
            if epoch % 5 == 0 or epoch == 24:
                print(f"  Epoch {epoch:2d} | Loss: {total_loss/len(train_loader):.4f} | Val F1: {val_f1:.4f}", flush=True)
        
        model.load_state_dict(torch.load(f'models/v2/robust_mil_fold{fold}.pt'))
        model.eval()
        fold_preds = []
        with torch.no_grad():
            for b_nums, b_cats, b_mask, b_labels in val_loader:
                logits = model(b_nums, b_cats, b_mask)
                fold_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
        oof_preds[va_idx] = fold_preds
        print(f"Fold {fold+1} Best Val F1: {best_val_f1:.4f}", flush=True)

    print("\n" + "="*60, flush=True)
    print("FINAL ROBUST MIL RESULTS", flush=True)
    print("="*60, flush=True)
    print(classification_report(bag_labels, oof_preds, target_names=['lower','middle','upper']), flush=True)
    final_f1 = f1_score(bag_labels, oof_preds, average='macro')
    print(f"OOF Macro F1: {final_f1:.4f}", flush=True)
    
    # Test Submission
    test_data = joblib.load('data/processed/test_instances.joblib')
    test_df = test_data['data']
    test_bag_ids = sorted(test_df['bag_id'].unique())
    test_ds = RobustBagDataset(test_df, num_cols, cat_cols, test_bag_ids, is_train=False)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, collate_fn=collate_mil)
    
    all_probs = np.zeros((len(test_bag_ids), 3))
    for fold in range(5):
        model.load_state_dict(torch.load(f'models/v2/robust_mil_fold{fold}.pt'))
        model.eval()
        with torch.no_grad():
            ptr = 0
            for b_nums, b_cats, b_mask, _ in test_loader:
                logits = model(b_nums, b_cats, b_mask)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                all_probs[ptr : ptr+len(probs)] += probs / 5
                ptr += len(probs)
    
    final_test_preds = np.argmax(all_probs, axis=1)
    sub = pd.DataFrame({'bag_id': test_bag_ids, 'label': final_test_preds})
    os.makedirs('submissions/v2_mil_robust', exist_ok=True)
    sub.to_csv('submissions/v2_mil_robust/submission.csv', index=False)
    print("Saved to submissions/v2_mil_robust/submission.csv")

if __name__ == '__main__':
    train_mil()
