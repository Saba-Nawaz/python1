#here i use start(2) stop(9) and steps(3)
#x="Hello World"
#y=" To python"
#a=slice(2,9,3)
#print(x[a])
#here i use -indexing to find the position of the specific position of the char
#a=slice(-9,-3,3)
#print(x[a])
#here i perform concatination
#con=x+y
#print(con)
#here to replicate the string
#z=3*x
#print(z)
# creating an empty list
my_str=[]
rev_str=[]
count=0
my_str = input("Enter the string\n")
my_str = my_str.casefold()
rev_str = reversed(my_str)
if list(my_str) == list(rev_str):
  count=count+rev_str
  print(count)
else:
 my_str = input("Enter the string\n")
 my_str = my_str.casefold()
 rev_str = reversed(my_str)








