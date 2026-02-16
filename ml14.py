# from urllib.parse import parse_qs
# my_values = parse_qs('red=5&blue=0&green=',keep_blank_values=True)
# print(repr(my_values))

# print('Red:', my_values.get('red'))
# print('Blue:', my_values.get('blue'))
# print('Green:', my_values.get('green'))
# print('Opacity:', my_values.get('opacity'))

def headline(text: str, align: bool = True) -> str:
    if align:
        return f"{text}\n{'-' * 22}"
    else:
        return f" {text} ".center(50, "o")

# print(headline(34, align="false"))
open_file = open('flush_demo.py', 'w')
string = "import time\nfor num in range(10, 0, -1):\n\tprint(str(num)+' ', end='', flush=True)\n\ttime.sleep(1)\nprint('\\nCountdown finished!')"
print(string, file=open_file)
open_file.close()
