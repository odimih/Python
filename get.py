key = 'Sentence'
value = 'hello world'
num=22
formatted = f'{key:<10} is {value:<20}! Number: {num:.3f}'
print(formatted)

fruits = ['apple', 'banana', 'cherry']
for ind, fruit in enumerate(fruits, start=4):
    print(f'{ind:<5} {fruit:<10}')
