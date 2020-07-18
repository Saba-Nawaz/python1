#from Bio.Seq import Seq
#from Bio.Alphabet import IUPAC
#simple_seq = Seq("GATC")
#from Bio.SeqRecord import SeqRecord
#simple_seq_r = SeqRecord(simple_seq, id="AC12345")
#print(simple_seq_r)
#simple_seq_r.annotations["evidence"] = "None. I just made it up."
#simple_seq_r.annotations["evi"] = "None. I just ."
#print(simple_seq_r.annotations)
#m=Seq("GCTGCTCTGCAAACGTAACGA",IUPAC.ambiguous_dna)
#d=m[3:10]
#print(d)

#my_seq = Seq("GATCG", IUPAC.unambiguous_dna)
#this is used to count nucleotides oin the sequence
#m=("CGTGCGTCGTGTGCCAAAA".count("G"))
#print(m)
#for index, letter in enumerate(my_seq):
# print("%i %s" % (index, letter))

from Bio.Seq import Seq
from Bio.Alphabet import IUPAC
import Bio.Alphabet
my_seq=Seq("AGCTGCGTAA")
my_seq.alphabet
print(my_seq)
m=my_seq.complement()
print(m)
r=my_seq.reverse_complement()
print(r)
s=Seq('ACGTGCAACGT',Bio.Alphabet.IUPAC.unambiguous_dna)
for i in range(10):
    d=input("Enter DNA sequence")
    s=Seq(d,Bio.Alphabet.IUPAC.ambiguous_dna)
    print(s.transcribe())
    print(s.translate())