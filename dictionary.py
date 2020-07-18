#simpley retriving the data from the dictionary
d={1:'java',2:'pyhton',3:'c',4:'c++',5:'DBS'}
#print("hello :{0}".format(d[5]))
print(d[3])

rgb=[('white','000'),('blue','001'),('green','008')]
c=dict(rgb)
print(c)

#adding the value in the dictionary
d[6]='genomics'
print(d)

#deleting the dictionary
del d[4]
print(d)
print(2 in d)

# accesing the vauess of the dictionary
for value in d.values():
    print("values",value)

    #accessing the keys of the dictionary
for key in d.keys():

    #counting the key of the dictionary
    key=key+1
print("keys",key)

#accessing the both key and the value of the dictionary
for key, value in d.items():
    print(key,value)

#creating the dictionary list
students=[]
students.append({'name':'saba','age':'20','class':'python'})
print(students)
print(students[0]['name'])

#dictionary contaninng list and dictionary
student={'name':'saba','subject':['python','java','ds','cb'],'info':{1:'hello',2:'python'}}
print(student['subject'][2])
print(student['info'][2])
print(len(student))
student.pop('name')
print("poping",student)

#enter string to get the points
point={'a':1,'e':1,'l':1,'d':2,'g':2,'b':3,'c':3,'f':4,'h':4,'k':5,'x':10}
s=input("enter the strion to get the points")
sum=0
for i in s:
    for w in point:
        if i==w:
            sum=sum+point[w]

print(sum)
from collections import OrderedDict
seq={'A':'Ale','g':'glr','c':'cyc','e':'glu'}
print('{0}'.format(seq['c']))
o=OrderedDict()
o['s']='s'
o['l']='l'
print(OrderedDict())
