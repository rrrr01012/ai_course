a = [[0.1,0.2,0.9,0.5,0.6,0.8,0.1,0.8],[0.9,0.2,0.9,0.5,0.6,0.8,0.1,0.8]]

c=[]
d=[]
e=[]
for i in range(0,len(a)):
    b=[]
    for j in range(0,8):
        if(a[i][j]>=0.8):
            b.append(1)
        else:
            b.append(0)
    c.append(b)   
print(c)     

for i in range(0,len(c)):
    d=[]
    for j in range(0,8):
        if(c[i][j] == 1):
            d.append(j)

    e.append(d)
print(e)

