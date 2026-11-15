import random
from dotenv import load_dotenv
from deepeval.synthesizer import Synthesizer
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
import  glob
import json

load_dotenv()

def load_chunks():
    texts = []
    for path in glob.glob("video_text_track/*.vtt"):
        with open(path) as f:
            lines = [ln.strip() for ln in f
                     if ln.strip() and ln.strip() != "WEBVTT" and "-->" not in ln]
        texts.append(" ".join(lines))
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    return splitter.split_text("\n\n".join(texts))

chunks = load_chunks()
sample = random.sample(chunks, min(15, len(chunks)))    
contexts = [[c] for c in sample]                      

synthesizer = Synthesizer(model="gpt-4.1-mini")             
goldens = synthesizer.generate_goldens_from_contexts(
    contexts=contexts,
    include_expected_output=True,       
    max_goldens_per_context=1,          
)

rows = []
for i, gold in enumerate(goldens, 1):
    rows.append({
        "id": f"g{i:03d}",
        "query": gold.input,
        "ideal_answer": gold.expected_output,
        "source": "TODO-verify",        
    })
