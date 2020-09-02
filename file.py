#----        1st assignment--------------------------------------

#text = "X-DSPAM-Confidence:    0.8475"
#x= text.find(':')
#y = text[x+2:]
#z=float(y)
#print(z)


#  ----------------- 2nd assignment------------------------------------

#print("Hello, i am doing 2nd assignment of the coursera course")


#-------------------3rd assignment------------------------------------
#fname = input("Enter file name: ")
#fh = open(fname)
#for i in fh:
 #   j =i.strip()
  #  print(j.upper())


#-------------------4th assignment------------------------------------

fname = input("Enter file name: ")
fh = open(fname)
count =0
for line in fh:
  if not line.startswith("X-DSPAM-Confidence:"): continue
  x= line.find(':')

  y = line[x+1:]
  count = count + 1
  z=float(y)
  z=z+1

print(count)
a = z/count
print("Average spam confidence:",a)


#-------------------4thassignment------------------------------------
#fname = input("Enter file name: ")
#fh = open(fname)
#res = []
#for i in fh:
 #   i=i.split()
 #   if i not in res:
  #   res.append(i)
  #   res.sort()
#print(res)
