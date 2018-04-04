# Cheat sheet

## List
* A list of comma-separated values (items) between square brackets. The items of a list are ordered and can be accessed by indices.
* Tuples and Lists are ordered. Dictionaries are unordered. 
```
  - squares = [1, 4, 9, 16, 25] 
  - squares.count(1)  #1 counts frequency   
  - letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g'] 
  - nest lists :x = [['a', 'b', 'c'], [2, 1, 3]] 
  - csquares = squares  # Transfers pointers of squares to csquares. List access by index
  - csquares = list(squares) # make a copy of squares. suqres and csquares are independent.
  ```
  
  

## List Methods
```
fruits = ['orange', 'apple', 'pear', 'banana', 'kiwi', 'apple', 'banana']
fruits.count('apple')
fruits.index('banana')
fruits.index('banana', 4)  # Find next banana starting a position 4
fruits.reverse()
fruits.append('grape')
fruits.sort()
sorted(fruits): will not affect fruits List order
add one list to other using : '+', or using extend()
```
## Tuple
```
tup1= (12, 34, 56);
tup1[0]:  12  #access tuple by index
```
## Functions
```
* def lastc(s): return s[-1] # returns last character of string 
* def lastc(s): return s[-1] myList=[(2, 5), (1, 2), (4, 4), (2, 3), (2, 1)]   sorted(myList,key=lastc)
 ref: https://developers.google.com/edu/python/sorting
* def ask_ok(prompt, retries=4, reminder='Please try again!'):   can pass one argument,2 or all the argumnets: 
    - ask_ok('OK to overwrite the file?', 2) or 
    - ask_ok('Do you really want to quit?') or 
    - ask_ok('OK to overwrite the file?', 2, 'Come on, only yes or no!')
* Fctn: def parrot(voltage, state='a stiff', action='voom', type='Norwegian Blue'):
       ways to call: using positional and keywords. One value is mandatory above
		parrot(1000)                                          # 1 positional argument
		parrot(voltage=1000)                                  # 1 keyword argument
		parrot(voltage=1000000, action='VOOOOOM')             # 2 keyword arguments
		parrot(action='VOOOOOM', voltage=1000000)             # 2 keyword arguments
		parrot('a million', 'bereft of life', 'jump')         # 3 positional arguments
		parrot('a thousand', state='pushing up the daisies')  # 1 positional, 1 keyword
* Multiple arguments: def cheeseshop1(kind, *arguments):   call:  cheeseshop1("Limburger", "It's very runny, sir.","It's really very,   VERY runny, sir.")
* Mutiple dict as arguments '**': def cheeseshop2(kind, **keywords): print(kw, ":", keywords[kw])	; 
        call: cheeseshop2("Limburger", shopkeeper="Michael Palin",client="John Cleese",sketch="Cheese Shop Sketch")
```
## List as stack
```
stack = [3, 4, 5]
stack.append(6)  #  [3,4,5,6]  like push
stack.pop() # [3,4,5]  pops out 6 last added
```

## Queue
```
from collections import deque
queue = deque(["Eric", "John", "Michael"])
queue.popleft()  
```

## For loop
```
for i in range(5):
for i in range(4,9,2):
for i in A: # A is List then i refers to element in A not the index
```
```
Multiplication: 2*3
Power: 2**3 (2 cube)
```

## Trim
```
string.strip() # removes begining and ending spaces by default
string.strips('0') # removes if character '0' is present at the beginning and end of string. Generally, strip([chars]):   all combinations of its chars are stripped from both ends.
```

## SET
```
{},
unordered collection;
no duplicate elements;
create an empty set you have to use set(), not {};
membership testing using 'in';
Unique letters in set s1 = set('aab');
set operations: '-','|','&','^'
```

## Dictionaries
```
Unordered
keys unique
tel = {'sape': 4139, 'guido': 4127, 'jack': 4098}   ;tel.keys(),tel.values(),tel.items()
dict to set of tuples: set(tel.items())
access values: for k, v in tel.items():   print (k,v)
access values: for key in tel: print( key, tel[key])
delete : del tel['guido']
```


## Error
```
ValueError('invalid user response')
IOError:
stderr: import sys ; sys.stderr.write('problem reading:' + filename)
```

## Print
```
defualt ends with newline
print("Python" , end = '@')  : ends with @ and continues with  next print  
print("-" * 40)
```


## packing/unpacking argumnets
```
'*', '**'
t1 = [10, 20, 30]; 
  print(t1)  # print list
  print(*t1)  # print unpacked list
a1 = [0, 1, 2, 3]; a2 = [4, 5, 6, 7]; print(*zip(a1,a2)) # unpack tuples 
```

## Lambda Functions
```
g = lambda x: x**2  # a single expression defines for input x,  compute x**2
  call:g(4)
g = lambda x,y: x**2 + y**2
  call:g(4,3)
Filter:  filter():returns object
   list_a = [1, 2, 3, 4, 5, 6, 7, 8]  ; result = list(filter(lambda x: x%2 == 0, list_a ))
Map: result = list(map(lambda x: x**2, list_a ))
Reduce (output scalar value, can be used in place of for loop ):import functools    functools.reduce(lambda x,y: x+y, list_a)
 ```
 
## File
```
f = open('hello.txt', 'w')  ;f.write('Hello-1, world!\n') ;f.close();
f.read() #read 
f = open('hello.txt', 'r') # read mode 
```

## OS Utility
```
import os
dir = 'resources'
filenames = os.listdir(dir)
for filename in filenames:
    print(filename)  
    print(os.path.join(dir, filename))  # path
    print(os.path.abspath(os.path.join(dir, filename)))  # absolute path 
```
## Counter
```
 for i, line in enumerate(f):  // i is counter	attched to f
 ```
 # Numpy
 ## Working with arrays #[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14]
``` 
 import numpy as np
 array: a = np.arange(15)  #[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14]
 access array: a[index]
 matrix: a = np.arange(15).reshape(3, 5) # 3*5
 zeros: np.zeros( (3,4) )
 ones: np.ones( (3,4), dtype=int )
 linspace: b=np.linspace( 0, 2, 9 ) # 9 members from 0 to 2
 PI: np.pi
 Sin: np.sin(x)
 ```
 ## random array
 ```
       np.random.random(10)  #Random [0,1)
       a =  np.random.rand(4,5)  # 2d 4*5  over unifom distribution [0, 1)
       a0= a[:,4]   # Produces a array of one dimensional using 4th column
	   a0 = a[0, :] # Produces a array of one dimensional using 0th row
	   a[1:3,0:2] #  accessing rows and columns  [1,3), [0,2)
	   np.random.randint(1,6) # retunrs random [1,6)
	   np.random.randint(2, size=10) #array([1, 0, 0, 0, 1, 1, 0, 0, 1, 0])
	   np.random.choice(sa)  #select random number from the array/list  sa 
	   np.random.choice(sa,(2,3)) #random choice of shape 2*3
	   np.random.choice(array, 5, p=[0.5, 0.1, 0.1, 0.3]) #selects by considering probabilities p 
 	   rng = np.random.RandomState(10) # used to generate same sequence of random numbers just like np.random.seed(number)
	      rng = np.random.RandomState(10)   print(rng.randint(10))  # it prints the same number 
		  v = rng.normal(mu,sigma,1000)   # Normal generates random numbers(1000) with mean of mu and standard deviation of sigma .
          print (np.mean(v))   # prints mean which is mu here 
          print (np.std(v))   # prints std deviation that is sigma here 
```          
## arrange	   
```
x = np.arange(10)  
       x[2:5]  # prints from index [2,5)
	   x[:3]   # prints from [0,3)
	   x[1:7:2] # access starting at 1 with stride 2 upto < 7
	   x[9:4:-2] # access starting at 9 with stride -2  
	   x[::] # implicitly x[0:10:1]
	   x[::-2]  #implicitly x[9:0:-2]
	   x[0:-1] # access from first to last 
```     
	   
## array
```
   a = np.array( [20,30,40,50] ) 
      multiplication: c = a**2
	  c = a[a<40] #numbers less than 40 to c
	  c = a < 35 #Relational [ True  True False False]
 ```
## array operations
```
      A = np.array( [[1,1], [0,1]] )  # represents 2d array
	  B = np.array( [[2,0], [3,4]] )   
	        C = A * B    #output [[2 0] [0 4]]
			C = A.dot(B) # Matrix multiplication
			C = np.dot(A, B) # Matrix multiplication using numpy
			MA = np.mat(A) # MA represents Mattrix
	  A = np.array([ [11, 12, 13], [21, 22, 23], [31, 32, 33] ])
	  B = np.array([1, 2, 3])
	        print(A+B) # output = [[12 14 16] [22 24 26] [32 34 36]]
	  B = np.array([[1, 2, 3],] * 3) #output 3*3 array [[1 2 3] [1 2 3] [1 2 3]]
	  np.array([[1, 2, 3],] * 3).transpose() #Transpose of array
	  access multiple values by indices: C[[0, 2, 3, 1, 4, 1]]
   ```
	  
## Broadcast
```
 * The smaller array is “broadcast” across the larger array so that they have compatible shapes
 broadcasting describes how numpy treats arrays with different shapes during arithmetic operation
      A = np.array([ [11, 12, 13], [21, 22, 23], [31, 32, 33] ])
  	  B = np.array([1, 2, 3])
	  # output of form print(A + B[:, np.newaxis])  [[12 13 14][23 24 25] [34 35 36]]
  ```
## Tile
```
    A = np.array([ [11, 12, 13], [21, 22, 23], [31, 32, 33] ]) 
    B = np.tile(np.array([1, 2, 3]), (3, 1))
    print(B)    #[[1 2 3][1 2 3][1 2 3]]   # in (3,1) 3 is number of copies, 1 indicates array is not repeated. 2 (place of 1)indicates 
 ```   
## array repeated twice 
```
    print(A+B)  #[[12 14 16][22 24 26][32 34 36]]   Direct addition
    print(A*B)  #[[11 24 39][21 44 69] [31 64 99]] Direct multiplication
```   
## Linear algebra
```
Inverse:  np.linalg.inv(a)	 
Solve for X:  AX=Y        x = np.linalg.solve(a, y) 
```
		
## StacK/Floor/Split
```
VstacK C = np.vstack((A,B)) #Verticle stack
Hstack: C = np.hstack((A,B)) #Horizantal stack 

Floor:a = np.floor(10*np.random.random((2,12)))

Split:
      x,y,z = np.hsplit(a,3)   # Split a into 3
	  x,y,z = np.hsplit(a,(3,4))   # Split a after the third and the fourth column
      x,y = np.vsplit(a,2) #Verucle split into 2 of equal size 
```
	  
## Histogram
```
# Plot simple histogram
# Histograms shows the frequency of occurence of data in the particular interval
# density histogram is defined so that the area of each rectangle equals the relative frequency of the corresponding class, and the area of the entire histogram equals 1.
		import numpy as np
		import matplotlib.pyplot as plt	  
		sa = np.array( [1, 2, 3, 4, 5] )
		data = [ np.random.choice(sa) for _ in range(1000) ]
		binedges = np.array( [1, 2, 3, 4, 5, 6] ) # bin edges considers intervals as  [1,2), [2,3) ...
		hist, _ = np.histogram(data, bins=binedges) # hist: count of elements in each interval , '_' refers to bin edges
		plt.hist(data, bins=binedges)
		plt.show()

		np.diff(binedges) #command to show difference of intervals from binedges 
		hist, _ = np.histogram(data, bins=binedges,density=True)	
		             density : bool, optional
									If ``False``, the result will contain the number of samples in
									each bin. If ``True``, the result is the value of the
									probability *density* function at the bin, normalized such that
									the *integral* over the range is 1. Note that the sum of the
									histogram values will not be equal to 1 unless bins of unity
									width are chosen; it is not a probability *mass* function
	    hist.sum() # total number of samples
```

## MLab 
```
Matlab bridge that lets Matlab look like a normal python library. Plotting pdf	
reference:  https://mathinsight.org/probability_density_function_idea 
https://stats.stackexchange.com/questions/133369/the-total-area-underneath-a-probability-density-function-is-1-relative-to-wh   
# in short area under pdf for one particular interval indicates the probability for vaues to be in that specified interval.  
		Sample:
            import matplotlib.mlab as mlab   # for plotting pdf
			plt.hist(v, normed=True, bins=bin_edges)
			xmin, xmax = plt.xlim()  # xaxis mini and max values 
			print(xmin,xmax)
			x = np.linspace(xmin, xmax, 100)
			p = mlab.normpdf(x, mu, sigma)
			plt.plot(x, p, 'r', linewidth=2) #plot line in red clolor 
			plt.show()	
			
		plt.xlim(0, 10000) #sets x axis 
```
```
Sum:np.sum(isum) #sum	 
```

## Sparse Matrix
```
# matrix that has number of entries as zeroes is referred to as sparse 
# Coordinate Format (COO)
   data[i] is value at (row[i], col[i]) position
   duplicates entries are summed together:
		import numpy as np
		import scipy.sparse as sps           
		row = np.array([0, 3, 1, 0])
		col = np.array([0, 3, 1, 2])
		data = np.array([4, 5, 7, 9])
		mtx = sps.coo_matrix((data, (row, col)), shape=(4, 4))
		mtx     
		mtx.todense()
		#output:matrix([[4, 0, 9, 0],[0, 7, 0, 0],[0, 0, 0, 0],[0, 0, 0, 5]])

# Compressed Sparse Row Format 
    three NumPy arrays: indices, indptr, data
		data = np.array([1, 2, 3, 4, 5, 6])
		indices = np.array([0, 2, 2, 0, 1, 2])
		indptr = np.array([0, 2, 3, 6])
		mtx = sps.csr_matrix((data, indices, indptr), shape=(3, 3))
		mtx.todense()	
        #output: matrix([[1, 0, 2],[0, 0, 3],[4, 5, 6]])		

# Create RHS and solve A x = b for x:
b = rand(1000)
x = spsolve(A, b)	
x_ = solve(A.toarray(), b)
err = norm(x-x_)
```

# Usefull Tips

```
cretaing list: squares = [x**2 for x in range(10)]
               Ex2: [(x, y) for x in [1,2,3] for y in [3,1,4] if x != y]
			   
apply one function to all elements in the list: [abs(x) for x in vec]	
		   
Zip: x = [1, 2, 3] y = [4, 5, 6] zipped = zip(x, y)	(object)	list(zipped) = [(1, 4), (2, 5), (3, 6)]	 

List to set : set(list)  
Set to list: List(set)

string start with :  x.startswith('x')

2 List--->List of tuples(zip())---> dict (dict())

newaxis is used to increase the dimension 
   xs = np.array([1, 2, 3, 4, 5])
      #xs.shape (5,)  1D
   xs[:, np.newaxis].shape 
      #(5, 1) 2D
   xs[np.newaxis, :].shape
      #(1, 5)  
   xs[:, np.newaxis, np.newaxis].shape
      #(5, 1, 1) 3D
	  
One use of Underscore:
    [ np.random.randint(1, 6) for _ in range(10) ]  #prints list of random numbers in [1,6)
    https://stackoverflow.com/questions/5893163/what-is-the-purpose-of-the-single-underscore-variable-in-python
	
import matplotlib.mlab as mlab   # for plotting probability distribution function 
```

# usefull pieces of code
 ```
# print lines from a file  
# with for file streams/unmanaged resources 
import os
dir = 'resources'
filename = os.path.join(dir, 'test_house.csv')
print(filename)
with open(filename) as f:
    for i, line in enumerate(f):
        print(line,i)
        if i > 1:
            break
f.close()

# read from file
    dir = 'resources'
    filename = os.path.join(dir, 'ttest.csv')
    try:
        f = open(filename, 'r')
        text = f.read()
        print(text)
        f.close()
    except IOError:
        sys.stderr.write('problem reading:' + filename)
    print('After handling exception')


# sort dict by value
tel = {'sape': 4139, 'guido': 4127, 'jack': 4098}
sorted(tel.items(),key=lambda x: (x[1],x[0]))
 
# Boolean Mask
C = np.array([123,188,190,99,77,88,100])
A = np.array([4,7,2,8,6,9,5])
R = C[A<=5]
print(R) #output: [123 190 100]


# Boolean Indexing
import numpy as np
A = np.array([4, 7, 3, 4, 2, 8])
print(A == 4)
[ True False False  True False False]

# function that is called with a parameter p, which is a probabilty value between 0 and 1. 
# The function returns a 1 with a probability of p, and zeros with probability of (1-p).
def prob(a):
    x=np.random.random()
    if x< a:
        return 1
    else:
        return 0
n=1000
sum(prob(0.2) for _ in range(n))/n

# dummy data generation 
	1.data = [ np.random.choice(array) for _ in range(1000) ]
	2.data =np.random.randint(10,size=1000)
```
