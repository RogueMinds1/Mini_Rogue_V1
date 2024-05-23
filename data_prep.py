import re
from transformers import AutoTokenizer

def clean_text(text):
    """
    Cleans the text by removing extra spaces, punctuation, and converting to lowercase.
    """
    cleaned_text = text.lower()
    cleaned_text = re.sub(r"\s\s+", " ", cleaned_text)  # Remove extra spaces
    cleaned_text = re.sub(r"[^\w\s]", "", cleaned_text)  # Remove punctuation
    return cleaned_text

def preprocess_data(input_file, output_file, tokenizer_name):
    """
    Reads a text file, cleans it, tokenizes it, and saves the data for training.
    """
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            text = f.read()
        print("Read input file successfully.")
    except Exception as e:
        print(f"Error reading input file: {e}")
        return

    cleaned_text = clean_text(text)
    print(f"Cleaned text: {cleaned_text[:100]}...")  # Print first 100 characters of cleaned text for verification

    # Tokenize the cleaned text
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenized_text = tokenizer(cleaned_text, return_tensors="pt")

    # Save tokenized text to new file
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(tokenized_text["input_ids"][0])))
        print(f"Saved tokenized text to {output_file}.")
    except Exception as e:
        print(f"Error writing to output file: {e}")
        return

if __name__ == "__main__":
    input_file = r"/workspaces/Mini_Rogue_V1/OpensourceBooks1.txt"  # Replace with your actual file path
    output_file = "cleaned_tokenized.txt"
    tokenizer_name = "bert-base-uncased"  # Choose the tokenizer you want to use
    preprocess_data(input_file, output_file, tokenizer_name)
    print("Data preprocessing complete!")
