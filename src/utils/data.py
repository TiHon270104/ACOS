import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class ABSADataset(Dataset):
    def __init__(self, file_path, tokenizer_name='vinai/phobert-base', max_length=128):
        print(f"Loading dataset from {file_path}")
        # Đọc file với encoding phù hợp
        try:
            self.data = pd.read_csv(file_path, encoding='utf-8')
        except UnicodeDecodeError:
            self.data = pd.read_csv(file_path, encoding='latin-1')
            
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        
        # Maps cho sentiment và category
        self.sentiment_map = {
            'positive': 0,
            'negative': 1,
            'neutral': 2
        }
        
        self.category_map = {
            'APPEARANCE': 0,
            'TECHNICAL': 1,
            'SPECIALIZE': 2,
            'OTHER': 3,
            'CHARACTERISTIC': 4
        }
        
        # Xử lý NULL và -1 values
        self.data = self._preprocess_data()
        print(f"Loaded {len(self.data)} samples")

    def _preprocess_data(self):
        """Xử lý NULL và -1 values"""
        processed = self.data.copy()
        
        # Convert -1 thành NULL
        for col in ['aspect_start', 'aspect_end', 'opinion_start', 'opinion_end']:
            if col in processed.columns:
                processed[col] = processed[col].replace(-1, pd.NA)
        
        # Fill NULL values cho sentiment và category
        if 'sentiment' in processed.columns:
            processed['sentiment'] = processed['sentiment'].fillna('neutral')
        if 'category' in processed.columns:
            processed['category'] = processed['category'].fillna('OTHER')
            
        return processed

    def __len__(self):
        """Trả về số lượng mẫu trong dataset"""
        return len(self.data)

    def clean_text(self, text):
        """Clean text trước khi tokenize"""
        if pd.isna(text):
            return ""
        text = str(text)
        return text.strip()

    def _create_mask(self, start, end):
        """Tạo mask vector từ start/end positions"""
        mask = torch.zeros(self.max_length)
        try:
            # Xử lý multiple positions nếu có (split by |)
            starts = str(start).split('|')
            ends = str(end).split('|')
            
            for s, e in zip(starts, ends):
                s, e = int(s), int(e)
                if 0 <= s < self.max_length and 0 <= e <= self.max_length:
                    mask[s:e+1] = 1
        except:
            pass
        return mask

    def _create_pairing_mask(self, a_start, a_end, o_start, o_end):
        """Tạo mask matrix cho aspect-opinion pairs"""
        mask = torch.zeros(self.max_length, self.max_length)
        try:
            # Xử lý multiple positions
            a_starts = str(a_start).split('|')
            a_ends = str(a_end).split('|')
            o_starts = str(o_start).split('|')
            o_ends = str(o_end).split('|')
            
            for as_, ae in zip(a_starts, a_ends):
                for os_, oe in zip(o_starts, o_ends):
                    as_, ae = int(as_), int(ae)
                    os_, oe = int(os_), int(oe)
                    
                    if (0 <= as_ < self.max_length and 0 <= ae <= self.max_length and
                        0 <= os_ < self.max_length and 0 <= oe <= self.max_length):
                        mask[as_:ae+1, os_:oe+1] = 1
        except:
            pass
        return mask

    def __getitem__(self, idx):
        """Lấy một mẫu từ dataset"""
        row = self.data.iloc[idx]
        text = self.clean_text(row['text'])
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        item = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'text': text
        }

        # Thêm aspect mask nếu có
        if 'aspect_start' in row.index and pd.notna(row['aspect_start']):
            item['aspect_mask'] = self._create_mask(
                row['aspect_start'],
                row['aspect_end']
            )
        else:
            item['aspect_mask'] = torch.zeros(self.max_length)

        # Thêm opinion mask nếu có
        if 'opinion_start' in row.index and pd.notna(row['opinion_start']):
            item['opinion_mask'] = self._create_mask(
                row['opinion_start'],
                row['opinion_end']
            )
        else:
            item['opinion_mask'] = torch.zeros(self.max_length)

        # Thêm sentiment label nếu có
        if 'sentiment' in row.index:
            item['sentiment'] = torch.tensor(
                self.sentiment_map[row['sentiment']]
            )

        # Thêm category label nếu có
        if 'category' in row.index:
            item['category'] = torch.tensor(
                self.category_map[row['category']]
            )

        # Thêm pairing mask nếu có cả aspect và opinion
        if ('aspect_start' in row.index and pd.notna(row['aspect_start']) and
            'opinion_start' in row.index and pd.notna(row['opinion_start'])):
            item['pairing_mask'] = self._create_pairing_mask(
                row['aspect_start'], row['aspect_end'],
                row['opinion_start'], row['opinion_end']
            )
        else:
            item['pairing_mask'] = torch.zeros(self.max_length, self.max_length)

        return item