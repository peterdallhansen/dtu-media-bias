import xml.etree.ElementTree as ET
from pathlib import Path
from html.parser import HTMLParser
import re
from tqdm import tqdm
DATA_DIR = Path("./Dataset")

txt_tospace1 = re.compile('&#160;')
def cleantext(text):
    '''Clean the text extracted from XML (from Preprocessing/preprocessing.py)'''
    if not text: return ""
    text = text.replace("&amp;", "&")
    text = text.replace("&gt;", ">")
    text = text.replace("&lt;", "<")
    text = text.replace("<p>", " ")
    text = text.replace("</p>", " ")
    text = text.replace(" _", " ")
    text = text.replace("–", "-")
    text = text.replace("”", "\"")
    text = text.replace("“", "\"")
    text = text.replace("’", "'")
    text, _ = txt_tospace1.subn(' ', text)
    return text

class MyHTMLParser(HTMLParser):
    '''Replication of Preprocessing/htmlparser.py logic'''
    def __init__(self):
        kwargs = {}
        super().__init__(**kwargs)
        self.ignore = False
        self.data = []
        self.p = []
    def finishp(self):
        if len(self.p) > 0:
            self.data.append(self.p)
            self.p = []
    def handle_starttag(self, tag, attrs):
        if tag in ['script', 'style']:
            self.ignore = True
        elif tag in ['p', 'br']:
            self.finishp()
    def handle_endtag(self, tag):
        if tag in ['script', 'style']:
            self.ignore = False
        elif tag in ['p', 'br']:
            self.finishp()
    def handle_startendtag(self, tag, attrs):
        if tag in ['p', 'br']:
            self.finishp()
    def handle_data(self, data):
        if not self.ignore:
            self.p.append(data)
    def close(self):
        super().close()
        self.finishp()
    def reset(self):
        super().reset()
        self.data = []
        self.p = []
    def cleanparagraph(self, text):
        text = cleantext(text)
        text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
        text = ' '.join(text.split()).strip()
        return text
    def paragraphs(self):
        pars = []
        for par in self.data:
            if len(par) > 0:
                text = self.cleanparagraph(''.join(par)).strip()
                if text:
                    pars.append(text)
        return pars

def parse_articles_streaming(path):
    """
    Parses articles using streaming (iterparse) to be memory efficient 
    and applies the project's specific text cleaning logic.
    """
    html_parser = MyHTMLParser()
    articles = {}
    
    context = ET.iterparse(path, events=("end",))
    
    
    for event, elem in tqdm(context, desc=f"Parsing {path.name}"):
        if elem.tag == "article":
            aid = elem.get("id")
            title_raw = elem.get("title") or ""
            title = cleantext(title_raw)
            
            xml_str = ET.tostring(elem, encoding="utf-8", method="xml").decode()
            
            html_parser.reset()
            html_parser.feed(xml_str)
            html_parser.close()
            
            pars = html_parser.paragraphs()
            text = " ".join(pars)
            
            articles[aid] = {"id": aid, "title": title, "text": text}
            
            # Clear element to free memory
            elem.clear()
            
    return articles
def parse_labels_streaming(path):
    labels = {}
    context = ET.iterparse(path, events=("end",))
    for event, elem in context:
        if elem.tag == "article":
            aid = elem.get("id")
            label = elem.get("hyperpartisan")
            if label:
                labels[aid] = 1 if label == "true" else 0
            elem.clear()
    return labels
def load_split(name):
    """
    name ∈ {"training", "validation", "test"}
    """
    modifier = "byarticle"
    
    if name == "validation": 
        if not list(DATA_DIR.glob(f"articles-{name}-{modifier}-*.xml")):
             print(f"Note: 'byarticle' split not found for '{name}', falling back to 'bypublisher'.")
             modifier = "bypublisher"
    try:
        articles_file = next(DATA_DIR.glob(f"articles-{name}-{modifier}-*.xml"))
        gt_file = next(DATA_DIR.glob(f"ground-truth-{name}-{modifier}-*.xml"))
    except StopIteration:
        raise FileNotFoundError(f"Could not find dataset files for split '{name}' in {DATA_DIR.absolute()}")
    print(f"Loading {name} split from: {articles_file.name}")
    
    articles = parse_articles_streaming(articles_file)
    labels = parse_labels_streaming(gt_file)
    merged = []
    for aid, art in articles.items():
        if aid in labels:
            merged.append({**art, "label": labels[aid]})
        else:
             pass
             
    return merged
