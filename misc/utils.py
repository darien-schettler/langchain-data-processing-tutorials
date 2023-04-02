from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
import argparse


def get_vs(txt_path):
    loader = TextLoader(txt_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 100,
        chunk_overlap  = 20,
        length_function = len,
    )

def main():
    parser = argparse.ArgumentParser(description="Convert a text file into a chunked vector store")
    parser.add_argument("text_file_path", help="Path to the text file.")
    parser.add_argument("--metadata", nargs='+', action='append', help="Metadata fields in the format key=value.")

    args = parser.parse_args()

    metadata = {}
    if args.metadata:
        for item in args.metadata:
            key, value = item[0].split('=')
            metadata[key] = value

    document = create_document(args.text_file_path, metadata)
    print("Document created with content:\n", document.page_content)
    print("Metadata:\n", document.metadata)

if __name__ == "__main__":
    main()
