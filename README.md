# Getting Started

LangChain primary focuses on constructing indexes with the goal of using
them as a Retriever. In order to best understand what this means, it\'s
worth highlighting what the base Retriever interface is. The
`BaseRetriever` class in LangChain is as follows:
:::

::: {#b09ac324 .cell .code execution_count="2"}
``` python
from abc import ABC, abstractmethod
from typing import List
from langchain.schema import Document

class BaseRetriever(ABC):
    @abstractmethod
    def get_relevant_documents(self, query: str) -> List[Document]:
        """Get texts relevant for a query.

        Args:
            query: string to find relevant tests for

        Returns:
            List of relevant documents
        """
```
:::

::: {#e19d4adb .cell .markdown}
It\'s that simple! The `get_relevant_documents` method can be
implemented however you see fit.

Of course, we also help construct what we think useful Retrievers are.
The main type of Retriever that we focus on is a Vectorstore retriever.
We will focus on that for the rest of this guide.

In order to understand what a vectorstore retriever is, it\'s important
to understand what a Vectorstore is. So let\'s look at that.
:::

::: {#2244801b .cell .markdown}
By default, LangChain uses [Chroma](../../ecosystem/chroma.md) as the
vectorstore to index and search embeddings. To walk through this
tutorial, we\'ll first need to install `chromadb`.

    pip install chromadb

This example showcases question answering over documents. We have chosen
this as the example for getting started because it nicely combines a lot
of different elements (Text splitters, embeddings, vectorstores) and
then also shows how to use them in a chain.

Question answering over documents consists of four steps:

1.  Create an index
2.  Create a Retriever from that index
3.  Create a question answering chain
4.  Ask questions!

Each of the steps has multiple sub steps and potential configurations.
In this notebook we will primarily focus on (1). We will start by
showing the one-liner for doing so, but then break down what is actually
going on.

First, let\'s import some common classes we\'ll use no matter what.
:::

::: {#8d369452 .cell .code execution_count="3"}
``` python
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
```
:::

::: {#07c1e3b9 .cell .markdown}
Next in the generic setup, let\'s specify the document loader we want to
use. You can download the `state_of_the_union.txt` file
[here](https://github.com/hwchase17/langchain/blob/master/docs/modules/state_of_the_union.txt)
:::

::: {#33958a86 .cell .code execution_count="19"}
``` python
from langchain.document_loaders import TextLoader
loader = TextLoader('../state_of_the_union.txt')
```
:::

::: {#489c74bb .cell .markdown}
## One Line Index Creation

To get started as quickly as possible, we can use the
`VectorstoreIndexCreator`.
:::

::: {#403fc231 .cell .code execution_count="5"}
``` python
from langchain.indexes import VectorstoreIndexCreator
```
:::

::: {#57a8a199 .cell .code execution_count="6"}
``` python
index = VectorstoreIndexCreator().from_loaders([loader])
```

::: {.output .stream .stdout}
    Running Chroma using direct local API.
    Using DuckDB in-memory for database. Data will be transient.
:::
:::

::: {#f3493fa4 .cell .markdown}
Now that the index is created, we can use it to ask questions of the
data! Note that under the hood this is actually doing a few steps as
well, which we will cover later in this guide.
:::

::: {#23d0d234 .cell .code execution_count="7"}
``` python
query = "What did the president say about Ketanji Brown Jackson"
index.query(query)
```

::: {.output .execute_result execution_count="7"}
    " The president said that Ketanji Brown Jackson is one of the nation's top legal minds, a former top litigator in private practice, a former federal public defender, and from a family of public school educators and police officers. He also said that she is a consensus builder and has received a broad range of support from the Fraternal Order of Police to former judges appointed by Democrats and Republicans."
:::
:::

::: {#ae46b239 .cell .code execution_count="8"}
``` python
query = "What did the president say about Ketanji Brown Jackson"
index.query_with_sources(query)
```

::: {.output .execute_result execution_count="8"}
    {'question': 'What did the president say about Ketanji Brown Jackson',
     'answer': " The president said that he nominated Circuit Court of Appeals Judge Ketanji Brown Jackson, one of the nation's top legal minds, to continue Justice Breyer's legacy of excellence, and that she has received a broad range of support from the Fraternal Order of Police to former judges appointed by Democrats and Republicans.\n",
     'sources': '../state_of_the_union.txt'}
:::
:::

::: {#ff100212 .cell .markdown}
What is returned from the `VectorstoreIndexCreator` is
`VectorStoreIndexWrapper`, which provides these nice `query` and
`query_with_sources` functionality. If we just wanted to access the
vectorstore directly, we can also do that.
:::

::: {#b04f3c10 .cell .code execution_count="9"}
``` python
index.vectorstore
```

::: {.output .execute_result execution_count="9"}
    <langchain.vectorstores.chroma.Chroma at 0x119aa5940>
:::
:::

::: {#297ccfa4 .cell .markdown}
If we then want to access the VectorstoreRetriever, we can do that with:
:::

::: {#b8fef77d .cell .code execution_count="10"}
``` python
index.vectorstore.as_retriever()
```

::: {.output .execute_result execution_count="10"}
    VectorStoreRetriever(vectorstore=<langchain.vectorstores.chroma.Chroma object at 0x119aa5940>, search_kwargs={})
:::
:::

::: {#2cb6d2eb .cell .markdown}
## Walkthrough

Okay, so what\'s actually going on? How is this index getting created?

A lot of the magic is being hid in this `VectorstoreIndexCreator`. What
is this doing?

There are three main steps going on after the documents are loaded:

1.  Splitting documents into chunks
2.  Creating embeddings for each document
3.  Storing documents and embeddings in a vectorstore

Let\'s walk through this in code
:::

::: {#54270abc .cell .code execution_count="11"}
``` python
documents = loader.load()
```
:::

::: {#9fdc0fc2 .cell .markdown}
Next, we will split the documents into chunks.
:::

::: {#afecb8cf .cell .code execution_count="12"}
``` python
from langchain.text_splitter import CharacterTextSplitter
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
```
:::

::: {#4bebc041 .cell .markdown}
We will then select which embeddings we want to use.
:::

::: {#9eaaa735 .cell .code execution_count="13"}
``` python
from langchain.embeddings import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()
```
:::

::: {#24612905 .cell .markdown}
We now create the vectorstore to use as the index.
:::

::: {#5c7049db .cell .code execution_count="14"}
``` python
from langchain.vectorstores import Chroma
db = Chroma.from_documents(texts, embeddings)
```

::: {.output .stream .stdout}
    Running Chroma using direct local API.
    Using DuckDB in-memory for database. Data will be transient.
:::
:::

::: {#f0ef85a6 .cell .markdown}
So that\'s creating the index. Then, we expose this index in a retriever
interface.
:::

::: {#13495c77 .cell .code execution_count="15"}
``` python
retriever = db.as_retriever()
```
:::

::: {#30c4e5c6 .cell .markdown}
Then, as before, we create a chain and use it to answer questions!
:::

::: {#3018f865 .cell .code execution_count="16"}
``` python
qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=retriever)
```
:::

::: {#032a47f8 .cell .code execution_count="17"}
``` python
query = "What did the president say about Ketanji Brown Jackson"
qa.run(query)
```

::: {.output .execute_result execution_count="17"}
    " The President said that Judge Ketanji Brown Jackson is one of the nation's top legal minds, a former top litigator in private practice, a former federal public defender, and from a family of public school educators and police officers. He said she is a consensus builder and has received a broad range of support from organizations such as the Fraternal Order of Police and former judges appointed by Democrats and Republicans."
:::
:::

::: {#9464690e .cell .markdown}
`VectorstoreIndexCreator` is just a wrapper around all this logic. It is
configurable in the text splitter it uses, the embeddings it uses, and
the vectorstore it uses. For example, you can configure it as below:
:::

::: {#4001bbc6 .cell .code execution_count="14"}
``` python
index_creator = VectorstoreIndexCreator(
    vectorstore_cls=Chroma, 
    embedding=OpenAIEmbeddings(),
    text_splitter=CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
)
```