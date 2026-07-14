from pathlib import Path
import numpy as np
import pandas as pd

EMBEDDINGS_FILE = Path("vector_store/embeddings.npy")
OUTPUT_FILE = Path("vector_store/embeddings_preview.csv")

embeddings = np.load(EMBEDDINGS_FILE)

df = pd.DataFrame(embeddings)

df.head(20).to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

print("Đã lưu file xem thử tại:", OUTPUT_FILE)