import pandas as pd
import codecs

class DataReader:
    @staticmethod
    def read_csv(file_path):
        """Đọc file CSV với encoding phù hợp"""
        encodings = ['utf-8', 'utf-16', 'latin-1', 'cp1252']
        
        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                # Chuẩn hóa các cột text
                text_columns = ['text']
                for col in text_columns:
                    if col in df.columns:
                        df[col] = df[col].apply(lambda x: str(x).encode('utf-8', errors='ignore').decode('utf-8'))
                return df
            except:
                continue
                
        raise ValueError(f"Không thể đọc file {file_path} với các encoding đã thử") 