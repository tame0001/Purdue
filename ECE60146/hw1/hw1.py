class Sequence(object):
    '''
    The base class. The array is defined in this class.
    This class is iterable so all children inherit this.
    '''
    def __init__(self, array) -> None:
        self.array:list = array # store array at base class
        self.index = -1 # index for iteration
    
    def __len__(self): # for len(..)
        return len(self.array)
    
    def __iter__(self): # for make the class iterable
        return self

    def __next__(self):
        self.index += 1 # move index
        if self.index < len(self.array): 
            # check if index in in valid range
            return self.array[self.index]
        else:
            raise StopIteration

    def __gt__(self, other): # for comparison 
        count = 0
        if len(self) != len(other):
            # check if two arrays are the same length
            raise ValueError('Two arrays are not equal in length!')
        for x, y in zip(self, other):
            if x > y: 
                count += 1
        return count
    
class Fibonacci(Sequence):
    '''
    The Fibonacci class. This class takes 2 parameters at initation.
    '''
    def __init__(self, first_value, second_value) -> None:
        super().__init__([first_value, second_value])
        
    def __call__(self, length) -> list:
        # the first two are defined during initiation.
        # only add the remaining 
        for _ in range(length-2):
            # sum of the last two elements
            self.array.append(self.array[-1]+self.array[-2])
        if length < len(self.array): 
            # for the case that call with length that less than current length
            self.array = self.array[:length]
        return self.array

FS = Fibonacci(first_value=1, second_value=2)
FS(length=5)
# create a Fibonacci class's instance and with 5 elements

print(len(FS)) # lenght is 5
print([n for n in FS]) # the instance is iterable

class Prime(Sequence):
    '''
    This class only shows prime numbers. 
    '''
    def __init__(self) -> None:
        # Initiation with an empty array
        super().__init__([])

    def __call__(self, length) -> list:
        prime_candidate = 2 # the first prime number
        while len(self.array) < length: 
            # loop until meet the required lenght
            while not self.is_prime(prime_candidate):
                # loop to find the next prime
                prime_candidate += 1
            self.array.append(prime_candidate)
            prime_candidate += 1
        if length < len(self.array):
            # for the case that call with length that less than current length
            self.array = self.array[:length]
        
        return self.array
    
    def is_prime(self, prime_candidate) -> bool:
        # testing until its square root is enough to prove
        for i in range(2, int(prime_candidate**0.5)+1):
            if prime_candidate % i == 0:
                return False
        return True

PS = Prime()
PS(length=8)
# create a Prime class's instance and with 8 elements

print(len(PS)) # length is 8
print([n for n in PS]) # the instance is iterable

# create an instance from each class with 8 elements
FS = Fibonacci(first_value=1, second_value=2)
FS(length=8)
PS = Prime()
PS(length=8)
# compare elements in both class
print(FS > PS)

PS(length=5)
# modify Prime instance to 5 element
# the error is raised
print(FS > PS)