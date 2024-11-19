import markovify


# Function to load text from file with error handling
def load_text(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        return text
    except FileNotFoundError:
        print(f"Error: File at {file_path} not found.")
        return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None


# Function to generate sentences using Markov chain
def generate_sentences(text_model, num_sentences=5, short_sentence=False, max_length=140):
    sentences = []
    if short_sentence:
        for _ in range(num_sentences):
            sentence = text_model.make_short_sentence(max_length)
            if sentence:  # Ensure a valid sentence is generated
                sentences.append(sentence)
    else:
        for _ in range(num_sentences):
            sentence = text_model.make_sentence()
            if sentence:  # Ensure a valid sentence is generated
                sentences.append(sentence)

    return sentences


# Function to save generated sentences to a file
def save_sentences(sentences, output_path="generated_sentences.txt"):
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            for sentence in sentences:
                f.write(sentence + "\n")
        print(f"Sentences saved to {output_path}")
    except Exception as e:
        print(f"Error saving sentences: {e}")


# Main functionality
def main():
    # File path and model parameters
    file_path = "C:\\Users\\jafar\\cyberflicker\\resources\\independence.txt"
    output_path = "generated_sentences.txt"
    num_sentences = 5
    short_sentence = False  # Set to True to generate short sentences
    max_length = 140

    # Load the text
    text = load_text(file_path)
    if text is None:
        return

    # Build the Markov chain model
    text_model = markovify.Text(text)

    # Generate sentences
    sentences = generate_sentences(text_model, num_sentences, short_sentence, max_length)
    print("\nGenerated Sentences:")
    for sentence in sentences:
        print(sentence)

    # Save generated sentences to a file
    save_sentences(sentences, output_path)


# Run the program
if __name__ == "__main__":
    main()