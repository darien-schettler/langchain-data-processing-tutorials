from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.vectorstores import Chroma
import argparse
import os


def get_vs(txt_path, db_dir="./db", search_type="similarity"):
    """ TBD

    Args:
        txt_path (str): Path to the text file
        db_dir (str, optional): Path to the directory where the vector store will be saved
        search_type (str, optional): The type of search to perform. Options are "similarity", "mmr", ...?

    Returns:
        None; prints the input and response text with the specified colors
    """
    # Step 1 - Create the document loader
    loader = TextLoader(txt_path)

    # Step 2 - Use the loader to parse the data file
    docs = loader.load()

    # Step 3 - Create the text splitter
    #     Default is RecursiveCharacterTextSplitter with the following parameters:
    #         - chunk_size=1000
    #         - chunk_overlap=0
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )

    # Step 4 - Split the documents
    sub_docs = splitter.split_documents(docs)

    # Step 5 - Initialize the embedding model
    #   Default is to use the text-embedding-ada-002 OpenAI model
    embeddings = OpenAIEmbeddings()

    # Step 6 - Create the vectorstore to use as the index
    #     Default is to use the Chroma vectorstore
    #          --> If the index already exists, load it as opposed to recreating it
    if os.path.isdir(os.path.join(db_dir, "index")) and len(os.listdir(os.path.join(db_dir, "index"))) > 3:
        db = Chroma(persist_directory=db_dir, embedding_function=embeddings)
    else:
        db = Chroma.from_documents(sub_docs, embeddings, persist_directory=db_dir)

    # Step 7 - Expose this index in a retriever interface
    retriever = db.as_retriever(search_type=search_type)



def main():
    """ NOTE: This is a simple example of how to use the langchain library to create a chunked vector store from a
    text file. Here are parts required to do this:


    Part 1 - Creating the document loader

        -------------------------------------------------------------------
                        LanguageChain Text Loader
        -------------------------------------------------------------------
        # Initialize with file path
        def __init__(self, file_path: str, encoding: Optional[str] = None):
            self.file_path = file_path
            self.encoding = encoding

        # Load from file path
        def load(self) -> List[Document]:
            with open(self.file_path, encoding=self.encoding) as f:
                text = f.read()
            metadata = {"source": self.file_path}
            return [Document(page_content=text, metadata=metadata)]
        -------------------------------------------------------------------

    Part 2 - Splitting The Text
    Part 3 - Creating The Chunked Vector Store
    Part 4 - Saving The Chunked Vector Store to a File
    """
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
