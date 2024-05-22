import re

def clean_text(text):
    """
    Cleans the text by removing extra spaces, punctuation, and converting to lowercase.
    """
    cleaned_text = text.lower()
    cleaned_text = re.sub(r"\s\s+", " ", cleaned_text)  # Remove extra spaces
    cleaned_text = re.sub(r"[^\w\s]", "", cleaned_text)  # Remove punctuation
    return cleaned_text

def preprocess_data(input_file, output_file):
    """
    Reads a text file, cleans it, builds a vocabulary, and saves the data for training.
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

    # Build vocabulary
    vocab = set(cleaned_text.split())
    vocab_size = len(vocab)
    print(f"Vocabulary size: {vocab_size}")

    # Save data to new file
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(cleaned_text)
        print(f"Saved cleaned text to {output_file}.")
    except Exception as e:
        print(f"Error writing to output file: {e}")
        return

    # Save vocabulary info
    try:
        with open("vocab.txt", "w", encoding="utf-8") as f:
            f.write(f"Vocabulary Size: {vocab_size}\n")
            f.write("\n".join(vocab))
        print("Saved vocabulary to vocab.txt.")
    except Exception as e:
        print(f"Error writing to vocab file: {e}")

if __name__ == "__main__":
    input_file = r"C:\RMI-CODE\Models\Mini_Rogue_V1\Mini_Rogue_V1\OpensourceBooks1.txt"  # Replace with your actual file path
    output_file = "cleaned_text.txt"
    preprocess_data(input_file, output_file)
    print("Data preprocessing complete!")
