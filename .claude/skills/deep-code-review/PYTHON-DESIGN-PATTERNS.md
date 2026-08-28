
# Creational Patterns
## Abstract Factory
Lets you produce families of related objects without specifying their concrete classes.
```python
class Button: pass
class WinButton(Button): pass
class MacButton(Button): pass

class GUIFactory:
    def create_button(self) -> Button: ...

class WinFactory(GUIFactory):
    def create_button(self): return WinButton()

class MacFactory(GUIFactory):
    def create_button(self): return MacButton()
```
## Builder
Lets you construct complex objects step by step. The pattern allows you to produce different types and representations of an object using the same construction code.
```python
class House:
    def __init__(self):
        self.parts = []

class HouseBuilder:
    def __init__(self): self.house = House()
    def add_walls(self): self.house.parts.append("walls"); return self
    def add_roof(self): self.house.parts.append("roof"); return self
    def build(self): return self.house

house = HouseBuilder().add_walls().add_roof().build()
```
## Factory Method
Provides an interface for creating objects in a superclass, but allows subclasses to alter the type of objects that will be created.
```python
class Dog:
    def speak(self): return "Woof"

class Cat:
    def speak(self): return "Meow"

class AnimalFactory:
    def create_animal(self, kind):
        return Dog() if kind == "dog" else Cat()
```
## Prototype
Lets you copy existing objects without making your code dependent on their classes.
```python
class Prototype:
    def __init__(self, data): self.data = data
    def clone(self): return copy.deepcopy(self)

a = Prototype([1, 2, 3])
b = a.clone()
```
## Singleton
Lets you ensure that a class has only one instance, while providing a global access point to this instance.
```python
class Singleton:
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
```
# Structural Patterns
## Adapter
Allows objects with incompatible interfaces to collaborate.
```python
class EuroPlug:
    def voltage(self): return 230

class USAdapter:
    def __init__(self, euro_plug): self.euro_plug = euro_plug
    def voltage(self): return 110

us_socket = USAdapter(EuroPlug())
```
## Bridge
Lets you split a large class or a set of closely related classes into two separate hierarchies—abstraction and implementation—which can be developed independently of each other.
```python
class Renderer:
    def render(self, shape): ...

class VectorRenderer(Renderer):
    def render(self, shape): return f"Vector: {shape}"

class Circle:
    def __init__(self, renderer): self.renderer = renderer
    def draw(self): return self.renderer.render("circle")
```
## Composite
Lets you compose objects into tree structures and then work with these structures as if they were individual objects.
```python
class Leaf:
    def total(self): return 1

class Composite:
    def __init__(self): self.children = []
    def add(self, child): self.children.append(child)
    def total(self): return sum(c.total() for c in self.children)
```
## Decorator
Lets you attach new behaviors to objects by placing these objects inside special wrapper objects that contain the behaviors.
```python
def bold(func):
    def wrapper(): return f"<b>{func()}</b>"
    return wrapper

@bold
def text(): return "Hello"
```
## Facade
Provides a simplified interface to a library, a framework, or any other complex set of classes.
```python
class CPU:
    def start(self): return "CPU started"

class Memory:
    def load(self): return "Memory loaded"

class Computer:
    def __init__(self): self.cpu, self.mem = CPU(), Memory()
    def start(self): return f"{self.cpu.start()}, {self.mem.load()}"
```
## Flyweight
Lets you fit more objects into the available amount of RAM by sharing common parts of state between multiple objects instead of keeping all of the data in each object.
```python
class TreeType:
    _types = {}
    def __new__(cls, name):
        if name not in cls._types:
            cls._types[name] = super().__new__(cls)
        return cls._types[name]
```
## Proxy
Lets you provide a substitute or placeholder for another object. A proxy controls access to the original object, allowing you to perform something either before or after the request gets through to the original object.
```python
class RealService:
    def request(self): return "Real response"

class Proxy:
    def __init__(self): self._service = None
    def request(self):
        if self._service is None:
            self._service = RealService()
        return self._service.request()
```
# Behavioral Patterns
Chain of Responsibility

Lets you pass requests along a chain of handlers. Upon receiving a request, each handler decides either to process the request or to pass it to the next handler in the chain.
```python
class Handler:
    def __init__(self, nxt=None): self.next = nxt
    def handle(self, request):
        if self.next: return self.next.handle(request)
        return f"Unhandled: {request}"

class AuthHandler(Handler):
    def handle(self, request):
        if request == "auth": return "Authenticated"
        return super().handle(request)
```
## Command
Turns a request into a stand-alone object that contains all information about the request. This transformation lets you pass requests as a method arguments, delay or queue a request's execution, and support undoable operations.
```python
class Command:
    def execute(self): ...

class PrintCommand(Command):
    def __init__(self, text): self.text = text
    def execute(self): print(self.text)

PrintCommand("Hello").execute()
```
## Iterator
Lets you traverse elements of a collection without exposing its underlying representation (list, stack, tree, etc.).
```python
class Range:
    def __init__(self, n): self.n = n
    def __iter__(self): return iter(range(self.n))

for i in Range(3): print(i)
```
## Mediator
Lets you reduce chaotic dependencies between objects. The pattern restricts direct communications between the objects and forces them to collaborate only via a mediator object.
```python
class Mediator:
    def notify(self, sender, event): print(f"{sender} -> {event}")

class Button:
    def __init__(self, mediator): self.mediator = mediator
    def click(self): self.mediator.notify("Button", "clicked")
```

## Memento
Lets you save and restore the previous state of an object without revealing the details of its implementation.
```python
class Editor:
    def __init__(self): self.text = ""
    def save(self): return self.text
    def restore(self, snapshot): self.text = snapshot

e = Editor(); e.text = "Hi"
snap = e.save()
e.text = "Bye"
e.restore(snap)
```

## Observer
Lets you define a subscription mechanism to notify multiple objects about any events that happen to the object they're observing.
```python
class Subject:
    def __init__(self): self.observers = []
    def subscribe(self, obs): self.observers.append(obs)
    def notify(self, event):
        for obs in self.observers: obs(event)

Subject().subscribe(lambda e: print(f"Got: {e}"))
```

## State
Lets an object alter its behavior when its internal state changes. It appears as if the object changed its class.
```python
class State:
    def handle(self): ...

class OnState(State):
    def handle(self): return "Turning off"

class OffState(State):
    def handle(self): return "Turning on"

class Light:
    def __init__(self): self.state = OffState()
    def press(self): return self.state.handle()
```

## Strategy
Lets you define a family of algorithms, put each of them into a separate class, and make their objects interchangeable.
```python
class Strategy:
    def execute(self, a, b): ...

class Add(Strategy):
    def execute(self, a, b): return a + b

class Multiply(Strategy):
    def execute(self, a, b): return a * b
```

## Template Method
Defines the skeleton of an algorithm in the superclass but lets subclasses override specific steps of the algorithm without changing its structure.
```python
class Algorithm:
    def run(self):
        self.step1()
        self.step2()
    def step1(self): print("Default step1")
    def step2(self): ...

class MyAlgorithm(Algorithm):
    def step2(self): print("Custom step2")
```
## Visitor
Lets you separate algorithms from the objects on which they operate.
```python
class Visitor:
    def visit(self, node): print(f"Visiting {node}")

class Node:
    def accept(self, visitor): visitor.visit(self)

Node().accept(Visitor())
```
