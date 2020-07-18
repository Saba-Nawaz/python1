p=[1,4,5,2,'y',8]

# retrive data by negative indeing
print("hello",p[-1])

#print the whole list
print("i am complete list",p)

#string is converted into list and the count the every element int he given list
a="jghgubgffuknhbgfrfd"
l=list(a)
count=0
print("string is converted into list",l)
for i in l:
 print(count)
 print("single element of the list",i)
 count=count+1

 # multiply the list with the number to pre-populate the list
m=['python']*3
print("i am multiply with the list",m)

#adding  the list
k=['hello', 'python', 'what', 'are', 'u','doing']

#using append
k.append('hello')
print(k)

#using insert
k.insert(2,'hello hussan')
print(k)

#using extend
k.extend([5,3,55])
print(k)

#deleting the list
t=[1,2,3,4,5,634,23,5,3,43]

#using pop
t.pop()
print(t)
t.pop(5)
print(t)
t.remove(23)
print("removig list",t)

#coping the list
a=[648,5,6,34]
b=a[:]
print(a)
b.pop()
print("printing the list b",b)

#OR
import copy
c=[2,4,5,6]
x=copy.copy(c)
x.pop()
print("printing x:",x)
print("printing c",c)

#clear the list
z=[3,5,37,4,23,56,23,54]
print(z)
z.clear()
print("I am going to clear the list",z)

#print the index of an element of the list
x=[65,56,23,78,34,65,89,23,23]

#retrive the element of the list by using the indexing of the list
print("I am index of an element",x.index(89))

#countint the element of the list
print("I am counting the element of the list",x.count(23))

#sorting the list
x.sort()
print("i am sorting the list",x)

#revesse sorting of the given list
x.sort(reverse=True)
print("i am sorting the list in reverse order",x)

#here directly revere the list
x.reverse()
print("I am directly reverse the list",x)

#sorting the list according to their size
y=['hello', 'python', 'what', 'are', 'u','doing']
y.sort(key=len)
print("i am sorting the list according to their length",y)

#sorting the list according to their size in revese order
y.sort(key=len,reverse=True)
print("I am sorting the list according to their length in reverse order",y)

#maimum of the list
print("i am returning the maximum of the list",max(y))

#minimum of the list
print("returnning the minimum of the list",min(y))

#giving the range in the list
print(list(range(12,23)))
