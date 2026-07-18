# RAG Langchain

This project is a multimodal retrieval-augmented generation (RAG) application for financial documents. It ingests PDF files, converts them to Markdown while preserving image references, splits the content into meaningful chunks, builds a hybrid retrieval index, and answers user questions grounded in the indexed content.

The implementation uses LangChain, Chroma, Docling, sentence-transformers, BM25, and optional Cohere reranking to provide a practical document Q&A workflow for financial reports and similar documents.

## Features

- PDF ingestion and Markdown conversion with Docling
- Image-aware chunking that keeps image references attached to surrounding text
- Hybrid retrieval using BM25 + dense vector search
- Optional Cohere reranking for higher-quality retrieval
- A CLI for indexing documents and asking questions
- Sample financial PDFs included in the data folder

## Project structure

- [src/cli.py](src/cli.py): command-line entry point for indexing and querying
- [src/bootstrap.py](src/bootstrap.py): wires together the indexing and RAG services
- [src/config.py](src/config.py): environment-based configuration and directory setup
- [src/ingestion/](src/ingestion/): PDF parsing and Markdown loading
- [src/chunking/](src/chunking/): image-aware text chunking
- [src/infrastructure/](src/infrastructure/): persistence for chunks and vector storage
- [src/retrieval/](src/retrieval/): hybrid retriever and reranking logic
- [src/services/](src/services/): indexing service and RAG service
- [data/input/](data/input/): sample PDF documents
- [notebooks/financial_RAG.ipynb](notebooks/financial_RAG.ipynb): notebook example

## Requirements

- Python 3.11+
- pip
- Internet access for downloading models and dependencies

## Installation

1. Open a terminal in the project root:

   ```bash
   cd RAG_Langchain
   ```

2. Create and activate a virtual environment:

   On Windows:

   ```bash
   py -3.11 -m venv .venv
   .\.venv\Scripts\activate
   ```

   On macOS/Linux:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. Create your environment file:

   ```bash
   copy .env.example .env
   ```

   On macOS/Linux:

   ```bash
   cp .env.example .env
   ```

5. Edit the .env file and add your API keys.

## Environment variables

The application reads settings from the .env file. The default values are already defined in [.env.example](.env.example).

| Variable | Required | Description |
| --- | --- | --- |
| GROQ_API_KEY | Yes | API key for the Groq LLM used by the answer generation step |
| COHERE_API_KEY | No | API key for Cohere reranking; if omitted, reranking is disabled |
| HUGGING_FACE_TOKEN | No | Optional token for Hugging Face model access |
| EMBEDDING_MODEL | No | Embedding model used for dense retrieval |
| LLM_MODEL | No | LLM model name used by Groq |
| RERANK_MODEL | No | Cohere rerank model name |
| CHROMA_COLLECTION | No | Name of the Chroma collection |
| DENSE_TOP_K | No | Number of dense retrieval results |
| SPARSE_TOP_K | No | Number of sparse retrieval results |
| RERANK_TOP_N | No | Number of results to keep after reranking |
| CHUNK_SIZE | No | Size of each text chunk |
| CHUNK_OVERLAP | No | Overlap between adjacent chunks |

## Usage

### 1. Index documents

The indexing step parses PDFs and creates chunks plus a vector store.

If you want to index specific files:

```bash
python -m src.cli index --pdf "data/input/IPI Aug 2025 EN.pdf" "data/input/LMS Q2_2025_EN.pdf"
```

If you do not pass any file paths, the tool will automatically index PDF files from [data/input/](data/input/) and Markdown files from the processed output directory.

```bash
python -m src.cli index
```

### 2. Ask questions

After indexing, you can ask questions about the content:

```bash
python -m src.cli ask "What are the main financial highlights in these documents?"
```

To also display the retrieved chunks used to generate the answer:

```bash
python -m src.cli ask "What are the main financial highlights in these documents?" --show-context
```

## Output folders

After indexing, the project creates the following directories:

- [data/processed/](data/processed/): generated Markdown files from PDF parsing
- [data/chroma/](data/chroma/): persistent Chroma vector database
- [data/index/](data/index/): chunk storage in JSONL format

## How it works

1. PDFs are parsed into Markdown using Docling.
2. The Markdown content is loaded into LangChain documents.
3. The content is split into chunks while keeping image references attached to the surrounding description.
4. Chunks are saved locally and inserted into a Chroma vector store.
5. User questions are answered by retrieving relevant chunks using hybrid search and passing them to a Groq-based LLM.

## Notes

- The project is designed for document-based Q&A and works especially well with financial reports and long-form PDF documents.
- The first run may take some time because the embedding model and language model dependencies are downloaded.
- If you see errors related to missing API credentials, check your .env file and ensure the required keys are set.
- If the indexing step fails, confirm that the input files exist in [data/input/](data/input/).

## Example workflow

```bash
python -m src.cli index --pdf "data/input/IPI Aug 2025 EN.pdf" "data/input/LMS Q2_2025_EN.pdf"
python -m src.cli ask "Summarize the key points from these reports"
```

## Improvement ideas for answer quality on long documents

If answer quality is weak on longer documents, the first thing I would change is the chunking strategy. The current approach uses fairly large markdown chunks, which can make it harder for the retriever to isolate the most relevant passage when a question targets a very specific section. I would reduce chunk size and increase overlap so that important facts remain preserved in local context and nearby sentences are still available. This is especially useful for financial reports, where the answer may depend on a single table row, a short paragraph, or a chart caption rather than a broad section of text.

I would also strengthen the retrieval layer by tuning the hybrid search setup. A balanced combination of sparse retrieval and dense vector retrieval is already useful, but for long and heterogeneous documents it can still miss important information if the query is highly specific. I would experiment with a stronger dense retrieval configuration, adjust the top-k values, and add a re-ranking step so that the most relevant chunks are promoted before the language model generates an answer. Re-ranking is particularly valuable when the initial retrieval returns many related but not fully relevant passages.

In practice, I would start with smaller chunks, more overlap, and a slightly higher number of retrieved candidates, then apply re-ranking before sending the context to the model. This should improve grounding, reduce hallucination, and make the system more reliable for complex financial documents.

## Example Q&A samples

The following examples show the kind of questions this system can answer from indexed financial documents, along with the supporting context it uses:

### Question 1

**Question:** What is the IPI value for mining and quarrying in June 2025?  
**Answer:** The IPI value for mining and quarrying in June 2025 is 106.6.  
**Context:**  
[1]  
The image presents a bar chart illustrating the changes in various industries from June 2025 to August 2025. The chart is divided into four sections, each representing a different industry.  
- Water supply; sewerage, waste management and remediation activities  
- June 2025: 131.6  
- July 2025: 130.9  
- August 2025: 129.9  
- Electricity, gas, steam and air conditioning supply  
- June 2025: 151.1  
- July 2025: 146.2  
- August 2025: 140.1  
- Manufacturing  
- June 2025: 124.0  
- July 2025: 123.6  
- August 2025: 119.8  
- Mining and quarrying  
- June 2025: 106.6  
- July 2025: 104.5  
- August 2025: 102.7  
The chart provides a clear visual representation of the fluctuations in these industries over the three-month period, allowing for easy comparison and analysis of the data.  

Figure 2. IPI according to ISIC4  

[2]  
## IPI by main economic activities  
The index of oil activities in August 2025 increased by 8.3% compared to the same month of the previous year. The index of non-oil activities increased by 4.4%. Based on the month-on-month trend, the index for oil activities increased by 1.7%, and the index of non-oil activities increased by 0.7%.  
**Chart Description**  
The chart displays the Industrial Production Index (IPI) and Annual Growth Rate over a period spanning from August 2024 to August 2025. The data is presented in a bar graph format, with the IPI represented by dark blue bars and the Annual Growth Rate depicted by a light blue line.  
**Data Representation**  
| Month | IPI | Annual Growth Rate |  
| --- | --- | --- |  
| Aug-24 | 106.0 | 3.0% |  
| Sep-24 | 104.0 | 1.0% |  
| Oct-24 | 108.0 | 6.0% |  
| Nov-24 | 106.0 | 5.0% |  
| Dec-24 | 104.0 | 2.0% |  
| Jan-25 | 103.0 | 0.5% |  
| Feb-25 | 103.0 | 0.0% |  
| Mar-25 | 108.0 | 4.0% |  
| Apr-25 | 104.0 | 1.0% |  
| May-25 | 108.0 | 4.0% |  
| Jun-25 | 110.0 | 6.0% |  
| Jul-25 | 112.0 | 7.0% |  
| Aug-25 | 114.0 | 7.5% |  
**Key Observations**  
- The IPI values range from 103.0 to 114.0.  
- The Annual Growth Rate fluctuates between 0.0% and 7.5%.  
- The chart provides a visual representation of the trends in IPI and Annual Growth Rate over the specified time period.  
**Conclusion**  
The chart effectively illustrates the changes in IPI and Annual Growth Rate from August 2024 to August 2025, offering insights into the industrial production trends during this time frame.  

[3]  
## IPI increases by 7.1% in August 2025  
Preliminary results indicate a 7.1% increase in the Industrial Production Index (IPI) in August 2025 compared to the same month of the previous year (August 2024), supported by the rise in mining and quarrying activity, manufacturing activity, electricity, gas, steam, and air conditioning supply activity and water supply, sewerage and waste management and remediation activities. On a monthly basis, the index increased by 1.4%.

### Question 2

**Question:** What was the unemployment rate for Saudi male youth (15–24) in Q2 2025?  
**Answer:** The unemployment rate for Saudi male youth (15–24) in Q2 2025 was 11.5%.  
**Context:**  
[1]  
## Decrease of unemployment rate for Saudi youth  
In Q2 of 2025, Saudi female youth aged 15-24 experienced a 0.8 percentage points decrease in the employment to population ratio, reaching 13.8%. Additionally, there was a 1.0 percentage points decrease in the participation rate, reaching 17.4%. However, the unemployment rate decreased by 0.1 percentage points reaching 20.6% compared to the previous quarter of 2025.  
On the other hand, the employment to population ratio for Saudi male youth showed a 1.2 percentage points decrease, reaching 28.0%, and recorded a 1.4 percentage points decrease in the labor force participation rate, reaching 31.6%. The unemployment rate decreased by 0.1 percentage points reaching 11.5% compared to the previous quarter of 2025.  
The results concerning labor market indicators for the Saudi population (both males and females) in the core working age group (25-54 years) during the second quarter of 2025 showed a 2.6 percentage points decrease in employment to population ratio, reaching 63.3%, and a 2.3 percentage point decrease in the participation rate, reaching 67.3%. Also, the unemployment rate increased to reach 5.9% compared to the previous quarter of 2025.  
For Saudis aged 55 and above, the labor market indicators for Q2 of 2025 indicated a decrease in the labor force participation rate and the employment to population ratio and an increase in the unemployment rate compared to the previous quarter of 2025.  

[2]  
## Figure1. Main Labor Market Indicators for Saudis by Age Group Q1 2025 / Q2 2025  
The image presents a bar graph illustrating the employment statistics for Saudi youth aged 15-24 years, categorized by gender and quarter. The graph is divided into three sections: Unemployment rate, Employment-to-population ratio, and Labour force participation rate.  
**Unemployment Rate**  
- Male:  
- Q1: 11.6  
- Q2: 11.5  
- Female:  
- Q1: 20.7  
- Q2: 20.6  
- Total:  
- Q1: 14.8  
- Q2: 14.7  
**Employment-to-Population Ratio**  
- Male:  
- Q1: 29.2  
- Q2: 28.0  
- Female:  
- Q1: 14.6  
- Q2: 13.8  
- Total:  
- Q1: 21.9  
- Q2: 20.9  
**Labour Force Participation Rate**  
- Male:  
- Q1: 33.0  
- Q2: 31.6  
- Female:  
- Q1: 18.4  
- Q2: 17.4  
- Total:  
- Q1: 25.7  
- Q2: 24.6  
The graph provides a clear visual representation of the employment trends among Saudi youth, highlighting the differences between males and females across various quarters.  
Source: Estimated data from the Labor Force Survey according to population estimates for Q2 2025- General Authority for Statistics  

[3]  
## Unemployment rate for total population reaches 3.2% in Q2 of 2025  
According to estimates from the Labor Force Survey of the General Authority for Statistics, the overall unemployment rate (for Saudis and non-Saudis) recorded 3.2% in Q2 of 2025. This marked an increase of 0.4 percentage points compared to Q1 of 2025, and a yearly decrease of 0.1 percentage points compared to Q2 of 2024. The overall labor force participation rate (for Saudis and non-Saudis) reached 67.1%, showing a decrease of 1.1 percentage points compared to Q1 of 2025 and a yearly increase of 0.9 percentage points compared to Q2 of 2024.  
The Saudi unemployment rate in Q2 of 2025 reached 6.8%, showing a 0.5 percentage points increase compared to Q1 of 2025, and a yearly decrease of 0.3 percentage points compared to Q2 of 2024. On the other hand, the results indicated a decrease in the employment-to-population ratio for Saudis by 2.1 percentage points compared to Q1 of 2025, reaching 45.9%, and decreased by 1.3 percentage points compared to Q2 of 2024. The labor force participation rate for Saudis in Q2 of 2025 decreased by 2.1 percentage points compared to Q1 of 2025, reaching 49.2%, and a decrease of 1.6 percentage points compared to Q2 of 2024.

### Question 3

**Question:** What was the opening inventory in 2002?  
**Answer:** The opening inventory in 2002 was $156,657.  
**Context:**  
[1]  
## XYZ COMPANY LIMITED STATEMENT OF INCOME AND RETAINED EARNINGS  
FOR THE YEAR ENDED JUNE 30, 2002  
UNAUDITED - See "Notice to Reader"  
This table shows the company’s income statement for 2002 and 2001, including revenue, cost of sales, gross profit, operating expenses, income from operations, other income and expenses, net income, and changes in retained earnings for each year.  
| | 2002 | 2001 |  
| --- | --- | --- |  
| REVENUE | $ 1,104,786 | $ 1,133,736 |  
| COST OF SALES | | |  
| Opening inventory | 156,657 | 146,278 |  
| Delivery | 1,607 | 1,249 |  
| Purchases | 740,994 | 794,101 |  
| | 899,258 | 941,628 |  
| Closing inventory | 159,144 | 156,657 |  
| | 740,114 | 784,971 |  
| GROSS PROFIT | 364,672 | 348,765 |  
| OPERATING EXPENSES (schedule) | 286,817 | 339,905 |  
| INCOME FROM OPERATIONS | 77,855 | 8,860 |  
| OTHER INCOME (EXPENSES) | | |  
| Loss on disposal of property, plant and equipment | -- | (387) |  
| Gain on sale of investment | 16,149 | -- |  
| Miscellaneous | (1,101) | 337 |  
| | 15,048 | (50) |  
| NET INCOME BEFORE TAX | 92,903 | 8,810 |  
| INCOME TAX EXPENSE | 14,387 | -- |  
| NET INCOME | 78,516 | 8,810 |  
| (DEFICIT) - Beginning of Year | (61,350) | (54,160) |  
| DIVIDENDS | -- | (16,000) |  
| RETAINED EARNINGS (DEFICIT) - End of Year | $ 17,166 | $ (61,350) |  

[2]  
## NOTES TO THE FINANCIAL STATEMENTS FOR THE YEAR ENDED JUNE 30, 2002  

[3]  
## INVENTORY  
The inventory is valued at the lower of cost or market, with cost being determined on a first-in, first-out basis.  
No images found in the context.

### Question 4

**Question:** What were the total assets in 2002?  
**Answer:** The total assets in 2002 were $276,498.  
**Context:**  
[1]  
## XYZ COMPANY LIMITED BALANCE SHEET AS AT JUNE 30, 2002  
UNAUDITED - See "Notice to Reader"  
This table presents the company’s balance sheet for 2002 and 2001, summarizing assets, liabilities, and shareholders’ equity, including current assets, property, plant and equipment, total liabilities, long-term debt, and retained earnings, with totals shown for each year.  
| | 2002 | 2001 |  
| --- | --- | --- |  
| ASSETS | | |  
| CURRENT | | |  
| Cash | $ 11,552 | $ -- |  
| Accounts receivable | 42,970 | 50,595 |  
| Deposits and prepaid expenses | 2,942 | 2,688 |  
| Inventory | 159,144 | 156,657 |  
| | 216,608 | 209,940 |  
| PROPERTY, PLANT AND EQUIPMENT (Note 2) | 59,890 | 76,318 |  
| INVESTMENTS | -- | 45,001 |  
| | $ 276,498 | $ 331,259 |  
| LIABILITIES | | |  
| CURRENT | | |  
| Bank overdraft | $ -- | $ 9,474 |  
| Bank loan | -- | 60,000 |  
| Accounts payable and accrued liabilities | 82,053 | 91,343 |  
| Long-term debt - current portion | 25,200 | -- |  
| Income tax payable | 14,387 | -- |  
| | 121,640 | 160,817 |  
| DUE TO SHAREHOLDER (Note 3) | 51,591 | 231,791 |  
| LONG-TERM DEBT (Note 4) | 86,100 | -- |  
| | 259,331 | 392,608 |  
| SHAREHOLDER'S EQUITY | | |  
| STATED CAPITAL (Note 5) | 1 | 1 |  
| RETAINED EARNINGS (DEFICIT) | 17,166 | (61,350) |  
| | 17,167 | (61,349) |  
| | $ 276,498 | $ 331,259 |  
APPROVED  

[2]  
## XYZ COMPANY LIMITED STATEMENT OF CASH FLOW FOR THE YEAR ENDED JUNE 30, 2002  
UNAUDITED - See "Notice to Reader"  
This table presents the company’s statement of cash flows for 2002 and 2001, summarizing cash flows from operating, investing, and financing activities, along with the net change in cash resources and the beginning and ending cash balances for each year.  
| | 2002 | 2001 |  
| --- | --- | --- |  
| CASH FLOWS FROM OPERATING ACTIVITIES | | |  
| Net income for the year | $ 78,516 | $ 8,810 |  
| Adjustment for: | | |  
| Amortization | 17,854 | 16,856 |  
| Loss on disposal of property, plant and equipment | -- | 387 |  
| Gain on disposal of investment | (16,149) | -- |  
| Cash derived from operations | 80,221 | 26,053 |  
| Decrease (increase) in working capital items | | |  
| Accounts receivable | 7,625 | 23,380 |  
| Deposits and prepaid expenses | (254) | 688 |  
| Inventory | (2,487) | (904) |  
| Accounts payable and accrued liabilities | (9,290) | 34,543 |  
| Long-term debt - current portion | 25,200 | -- |  
| Income tax payable | 14,387 | 2,206 |  
| Cash flows from operating activities | 115,402 | 85,966 |  
| CASH FLOWS FROM INVESTING ACTIVITIES | | |  
| Acquisition of property, plant and equipment | (1,426) | (10,342) |  
| Proceeds from disposal of property, plant and equipment | -- | 3,113 |  
| Proceeds from disposal of investment | 61,150 | -- |  
| Dividends | -- | (16,000) |  
| Cash flows from investing activities | 59,724 | (23,229) |  
| CASH FLOWS FROM FINANCING ACTIVITIES | | |  
| Advances from (repayments to) shareholder | (180,200) | (150,000) |  
| Acquisition of (repayment of) long-term debt | 86,100 | -- |  
| | (94,100) | (150,000) |  
| NET INCREASE (DECREASE) IN CASH RESOURCES | 81,026 | (87,263) |  
| CASH (DEFICIENCY) RESOURCES - Beginning of Year | (69,474) | 17,789 |  
| CASH RESOURCES (DEFICIENCY) - End of Year | $ 11,552 | $ (69,474) |  
| Cash resources (deficiency) is comprised of: | | |  
| Cash | $ 11,552 | $ -- |  
| Bank overdraft | -- | (9,474) |  
| Bank loan | -- | (60,000) |  
| | $ 11,552 | $ (69,474) |  
The accompanying summary of significant accounting policies and notes are an integral part of these financial statements.  
XYZ COMPANY LIMITED  

[3]  
## NOTES TO THE FINANCIAL STATEMENTS FOR THE YEAR ENDED JUNE 30, 2002  
No images found in the context.
