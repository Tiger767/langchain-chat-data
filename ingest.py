import argparse
import pickle
import os
from pathlib import Path

from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.document_loaders import PyMuPDFLoader
from langchain.document_loaders import BSHTMLLoader
from langchain.document_loaders import DirectoryLoader

from data_loaders.text import TextLoader


def delete_files_without_extension(folder_path, extension=None):
    if not os.path.exists(folder_path):
        print(f"The folder '{folder_path}' does not exist.")
        return

    deleted_files_count = 0
    for root, _, files in os.walk(folder_path):
        for file in files:
            if (extension is None and '.' in file) or (extension is not None and not file.endswith(extension)):
                file_path = Path(root) / file
                file_path.unlink()
                deleted_files_count += 1
                print(f"Deleted: {file_path}")


def ingest_data(directory, chunk_size, chunk_overlap, input_format, vector_postfix, dry_run=False): # 8_407_100 2_619_800
    # delete extra files
    #delete_files_without_extension("data/catalog", extension='.pdf')

    if input_format == "pdf":
        loader = DirectoryLoader(directory, loader_cls=PyMuPDFLoader, recursive=False)
    elif input_format == "text":
        loader = DirectoryLoader(directory, loader_cls=lambda path: TextLoader(path, encoding='utf-8'))
    elif input_format == "html":
        loader = DirectoryLoader(BSHTMLLoader(directory))
    else:
        raise ValueError("Invalid input format")

    raw_documents = loader.load()
    #print(raw_documents[:3])
    print(len(raw_documents))
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name='cl100k_base',
        chunk_size=100,
        chunk_overlap=20,

    )
    documents = text_splitter.split_documents(raw_documents)
    #print(documents[:3])
    print(len(documents), len(documents) * 100)

    if dry_run:
        print("Dry run completed. Stopping before embeddings are computed.")
        return

    embeddings = OpenAIEmbeddings()

    vectorstore = FAISS.from_documents(documents, embeddings)

    # Save vectorstore
    with open(f"vectorstore_{vector_postfix}.pkl", "wb") as f:
        pickle.dump(vectorstore, f)


def main():
    parser = argparse.ArgumentParser(description="Ingest data and create vectorstore")
    parser.add_argument("-d", "--directory", help="Directory containing the data", default="data/catalog_pdf/")
    parser.add_argument("-pf", "--postfix", help="Postfix for the vectorstore", default="default")
    parser.add_argument("-c", "--chunk_size", help="Chunk size for text splitting", type=int, default=100)
    parser.add_argument("-o", "--chunk_overlap", help="Overlap between chunks for text splitting", type=int, default=20)
    parser.add_argument("-f", "--input_format", help="Input format: pdf, text, or html", choices=["pdf", "text", "html"], default="pdf")
    parser.add_argument("--dry_run", help="Dry run, stops before embeddings are declared", action="store_true")

    args = parser.parse_args()

    ingest_data(args.directory, args.chunk_size, args.chunk_overlap, args.input_format, args.postfix, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
