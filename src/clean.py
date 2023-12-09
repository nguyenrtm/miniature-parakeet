import re

class TextCleaner:
    def __init__(self):
        pass

    def remove_dash(self, text: str):
        text = re.sub('(\\()(.*)(\\))-induced', ' \\2  induced', text) # (PAN)-induced => PAN induced
        text = re.sub('(\\()(.*)(\\))-associated', ' \\2  associated', text) # (PAN)-associated => PAN associated
        text = re.sub('-induced', ' induced', text)
        text = re.sub('-inducing', ' inducing', text)
        text = re.sub('-associated', ' associated', text)

        return text
    
    def clean(self, text: str):
        text = self.remove_dash(text)
        return text
    
    def clean_df(self, df):
        df['text'] = df['text'].apply(self.clean)
        return df