from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load the pretrained T5 model and tokenizer for question answering
tokenizer = T5Tokenizer.from_pretrained("t5-base")
model = T5ForConditionalGeneration.from_pretrained("t5-base")

# Function to generate answers using T5 model
def generate_answer(question, context):
    input_text = f"question: {question} context: {context}"

    # Tokenize the input text
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)

    # Generate the output sequence
    outputs = model.generate(inputs["input_ids"], max_length=150, num_beams=4, early_stopping=True)

    # Decode the generated output
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

# Main function for answering a question based on a paragraph
def main():
    # Define the context (paragraph)
    context = """
    The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It was named after the engineer Gustave Eiffel, whose company designed and built the tower. Completed in 1889 as the entrance arch to the 1889 World's Fair, it was initially criticized by some of France's leading artists and intellectuals for its design, but it has become a global cultural icon of France and one of the most recognizable structures in the world. The tower is 324 meters (1,063 feet) tall, about the same height as an 81-story building, and was the tallest man-made structure in the world until the Chrysler Building in New York City was completed in 1930.
    """

    # Ask a question
    question = input("Ask a question: ")

    # If a question is asked, generate an answer
    if question and context:
        print("Generating answer...")
        answer = generate_answer(question, context)
        print("Answer:")
        print(answer)

if __name__ == "__main__":
    main()
