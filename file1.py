
num1=int(input("Enter any number\n"))
num2=int(input("Enter any number\n"))
print("\n****Addition*****")
sum=(int)(num1)+(int)(num2)
print("The sum of the number",sum)
print("\n****Difference*****")
diff=(int)(num1)-(int)(num2)
print("The difference of the number",diff)
print("\n****Product*****")
pro=(int)(num1)*(int)(num2)
print("The product of the number",pro)
print("\n****Factorial*****")
fac = 1
for i in range(1, num1 + 1):
       fac = fac * i
print("Factorial of ", num1, " is ", fac)
fact = 1
for i in range(1, num2 + 1):
       fact = fact * i
print("Factorial of ", num2, " is ", fact)
print("\n****Multiples*****")
mul=(int)(num1)*(int)(num2)
print("The product of the number",mul)
print("\n****Factors*****")
factor=[]
for i in range(1, num1 + 1):
       if num1 % i == 0:
              factor.append(i)
print("Factors of the 1st number")
print("Factors of {} = {}".format(num1, factor))
factors=[]
for i in range(1, num2 + 1):
       if num2 % i == 0:
              factors.append(i)
print("Factors of the 2nd number")
print("Factors of {} = {}".format(num2, factors))