from somelib import g

def f(a):
	if a == 123:
		return f(10)
	else:
		return somelib.g(a)

def f2():
	return f(123)

class Example():
	def __init__(self):
		self.a = 1
	def make_f(self):
		return f(self.a)

class Example2():
	def __init__(self):
		self.my_example = Example1()
	def make_f2():
		return f2(self.my_example.make_f())