# Find Equivalent job title terms.

The basic concept of this tool is to find a 'primitive' term for a job title using AI eg: we will replace senior software engineer for  software engineer  or data analyst for data scientist.
Very handy for automatization/scraping jobs.
The embeddings model used is [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). We could get better results using SOTA embedding models but those rely on Sentence-Transformers and that library is prone to have issues. So for this functional POC I decided to take it simple specially since the results are already very good.

python version: 3.11.5
