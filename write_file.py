seq=''
charge=-0.235
aa_charge={'A':2,'T':4,'C':6,'G':8}
with open('fasta.fasta') as fh:
    next(fh)
    fh.readline()
    for line in fh:
        seq+=line[:-2].upper()
for aa in seq:
    charge+=aa_charge.get(aa,0)
with open('out.txt','w') as file_out:
   d= file_out.write(str(charge))
   print(d)