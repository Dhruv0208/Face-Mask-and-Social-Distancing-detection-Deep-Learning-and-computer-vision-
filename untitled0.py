import math
approximate_e = 0
previous_value = 0
facto = 1
i = 1
delta = 1
difference=0
a = round(math.exp(1),4)
while(difference != delta):
    previous_value = approximate_e
    facto = facto*i
    div = 1/facto
    x = round(div,4)
    if(i==1):
        approximate_e = approximate_e + x + 1
    else:
        approximate_e = approximate_e + x
    difference = round(approximate_e,4) - a
print("smallest k value is: ", i)
print("the value of e is: ", approximate_e)    
    

