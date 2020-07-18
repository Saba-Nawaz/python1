from Bio import Entrez
Entrez.email = "A.N.Other@example.com"  # Always tell NCBI who you are
pmid = "19304878"
record = Entrez.read(Entrez.elink(dbfrom="pubmed", id=pmid))
print(record)
print(record[0]["LinkSetDb"])
print(record[0]["IdList"])