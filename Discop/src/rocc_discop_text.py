import argparse
import sys

from config import Settings, text_default_settings
from stega_cy import encode_text, decode_text
from model import get_model, get_feature_extractor, get_tokenizer


def read_file(file_path):
    try:
        with open(file_path, 'r') as file:
            return file.read()
    except Exception as e:
        return f"Error reading file: {e}"


def run_stega_text(flag, context, message, settings: Settings = text_default_settings):
    model = get_model(settings)
    tokenizer = get_tokenizer(settings)

    single_example_output = encode_text(model, tokenizer, message, context)
    message_decoded = decode_text(model, tokenizer, single_example_output.generated_ids, context)
    message_encoded = message[:single_example_output.n_bits]
    if flag == 'enc':
        return message_encoded
    return message_decoded
    

def main(flag, text, filepath):
    if flag != "enc" and flag != "dec":
        print("Invalid flag. Please use 'encode' or 'decode'")
        sys.exit(1)

    print("Received text:", text)
    file_contents = read_file(filepath)
    Msg = run_stega_text(flag, text, file_contents)
    if flag == 'enc':
        with open("encoded_text.txt", "w") as f:
            f.write(Msg)
    else:
        with open("decoded_text.txt", "w") as f:
            f.write(Msg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="use Discop to encode or decode message text")
    parser.add_argument('flag', type=str, help="encode text or decode text, use 'enc' or 'dec'")
    parser.add_argument('text', type=str, help="Text input to be processed")
    parser.add_argument('filepath', type=str, help="The path to the file to be read")
    args = parser.parse_args()

    # execute the main function
    main(args.flag, args.text, args.filepath)
